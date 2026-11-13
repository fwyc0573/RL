# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare checkpoint logprobs in vLLM, native scoring, and native training."""

import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.algorithms.single_controller_utils.setup import (
    _maybe_inject_megatron_train_iters,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from tests.manual.mopd.native_forward_entrypoint import select_observed_workers
from tests.manual.mopd.replay_logprobs import error_stats


def select_prompts(dump: Path, identities: Path) -> list[dict[str, Any]]:
    """Select the longest recorded sequence from each source group."""
    with identities.open() as stream:
        groups = {
            index: json.loads(line)["task_id"]["taskset"]
            for index, line in enumerate(stream)
        }
    selected = {}
    with dump.open() as stream:
        for line in stream:
            row = json.loads(line)
            for index, metadata in enumerate(row["metadata"]):
                group = groups[metadata["prompt_idx"]]
                length = row["input_lengths"][index]
                if group not in selected or length > selected[group]["source_length"]:
                    selected[group] = {
                        "source_group": group,
                        "prompt_idx": metadata["prompt_idx"],
                        "source_sample_id": row["sample_id"][index],
                        "source_length": length,
                        "prompt_ids": row["prompt_ids"][index],
                    }
    if not selected:
        raise ValueError("The native dump has no samples")
    return [selected[group] for group in sorted(selected)]


def training_forward_report(
    directory: Path, batch: BatchedDataDict[Any]
) -> dict[str, Any]:
    """Match observed native training tokens to the fresh checkpoint responses."""
    remaining: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, length in enumerate(batch["input_lengths"].tolist()):
        remaining[tuple(batch["input_ids"][index, :length].tolist())].append(index)
    samples = []
    issues = []
    generation_errors = []
    scorer_errors = []
    for path in sorted(directory.glob("forward-*.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["kind"] != "training_forward":
                    continue
                for observed_index, length in enumerate(row["input_lengths"]):
                    ids = tuple(row["input_ids"][observed_index][:length])
                    if not remaining[ids]:
                        issues.append("unexpected_or_duplicate_sequence")
                        continue
                    index = remaining[ids].pop(0)
                    mask = batch["token_mask"][index, 1:length].bool()
                    observed_mask = torch.tensor(
                        row["token_mask"][observed_index][1:length]
                    ).bool()
                    if not torch.equal(mask, observed_mask) or not mask.any():
                        issues.append(f"sample_{index}_response_mask_mismatch")
                        continue
                    logprobs = torch.tensor(
                        row["next_token_logprobs"][observed_index][: length - 1]
                    )
                    generation_error = (
                        logprobs - batch["generation_logprobs"][index, 1:length]
                    )[mask]
                    scorer_error = (logprobs - batch["prev_logprobs"][index, 1:length])[
                        mask
                    ]
                    if not (
                        torch.isfinite(generation_error).all()
                        and torch.isfinite(scorer_error).all()
                    ):
                        issues.append(f"sample_{index}_nonfinite_logprobs")
                        continue
                    generation_errors.append(generation_error)
                    scorer_errors.append(scorer_error)
                    samples.append(
                        {
                            "sample_index": index,
                            "rank": row["rank"],
                            "length": length,
                            "logits_dtype": row["logits_dtype"],
                            "generation_error": error_stats(generation_error),
                            "scorer_error": error_stats(scorer_error),
                        }
                    )
    missing = [index for indices in remaining.values() for index in indices]
    report = {
        "expected_samples": len(batch["input_lengths"]),
        "observed_samples": len(samples),
        "missing_samples": missing,
        "issues": issues,
        "samples": samples,
        "verdict": "UNVERIFIED",
    }
    if generation_errors:
        report["generation_error"] = error_stats(torch.cat(generation_errors))
        report["scorer_error"] = error_stats(torch.cat(scorer_errors))
        if not missing and not issues:
            report["verdict"] = "MEASURED"
    return report


def main() -> None:
    """Use native checkpoint loading and the existing non-colocated refit transport."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = MasterConfig(
        **OmegaConf.to_container(
            parse_hydra_overrides(load_config(args.config), overrides),
            resolve=True,
        )
    )
    _maybe_inject_megatron_train_iters(config)
    generation_config = config.policy["generation"]
    if (
        generation_config["backend"] != "vllm"
        or generation_config["colocated"]["enabled"]
    ):
        raise ValueError("Checkpoint replay uses non-colocated vLLM generation")
    world_size = (
        config.policy["megatron_cfg"]["tensor_model_parallel_size"]
        * config.policy["megatron_cfg"]["pipeline_model_parallel_size"]
        * config.policy["megatron_cfg"]["context_parallel_size"]
    )
    if world_size != config.cluster.gpus_per_node or config.cluster.num_nodes != 2:
        raise ValueError(
            "Allocate two nodes: one complete policy replica and one generation node"
        )
    args.checkpoint.relative_to("/mnt/codesign-exp/ycfeng")
    weights = args.checkpoint / "policy/weights"
    if not (weights / "iter_0000000/.metadata").is_file():
        raise FileNotFoundError("A published native checkpoint is required")
    samples = select_prompts(args.dump, args.identities)
    output = Path(config.logger.log_dir) / "checkpoint-refit-logprobs.json"
    output.relative_to("/mnt/codesign-exp/ycfeng")
    observation_dir = output.parent / "checkpoint-training-forward"
    os.environ["MOPD_OBSERVATION_DIR"] = str(observation_dir)
    select_observed_workers()
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    generation_config = configure_generation_config(
        copy.deepcopy(generation_config), tokenizer
    )
    generation_config["model_name"] = config.policy["model_name"]
    init_ray()
    train_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[world_size],
        num_gpus_per_node=config.cluster.gpus_per_node,
        name="checkpoint_replay_policy",
    )
    generation_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[config.cluster.gpus_per_node],
        num_gpus_per_node=config.cluster.gpus_per_node,
        name="checkpoint_replay_generation",
    )
    policy = Policy(
        cluster=train_cluster,
        config=config.policy,
        tokenizer=tokenizer,
        name_prefix="checkpoint_replay",
        weights_path=str(weights),
        init_optimizer=True,
        init_reference_model=False,
    )
    generation = VllmGeneration(cluster=generation_cluster, config=generation_config)
    try:
        synchronizer = create_weight_synchronizer(
            policy=policy,
            generation=generation,
            generation_backend="vllm",
            colocated=False,
            train_cluster=train_cluster,
            inference_cluster=generation_cluster,
        )
        synchronizer.init_communicator()
        policy.sync_params_before_refit()
        synchronizer.sync_weights()
        if synchronizer.is_stale:
            raise RuntimeError("The checkpoint refit did not complete")
        policy.prepare_for_lp_inference()
        report = {
            "checkpoint": str(args.checkpoint),
            "dump": str(args.dump),
            "comparison": "Reloaded native Megatron scorer minus vLLM fresh generation after checkpoint refit.",
            "samples": [],
        }
        errors = []
        training_rows = []
        for sample in samples:
            prompt = sample["prompt_ids"]
            generated = generation.generate(
                BatchedDataDict(
                    {
                        "input_ids": torch.tensor([prompt], dtype=torch.long),
                        "input_lengths": torch.tensor([len(prompt)], dtype=torch.long),
                    }
                )
            )
            length = int(generated["unpadded_sequence_lengths"][0])
            ids = generated["output_ids"][0, :length].cpu()
            if not torch.equal(ids[: len(prompt)], torch.tensor(prompt)):
                raise ValueError("Generation changed the supplied prompt tokens")
            actual = policy.get_logprobs(
                BatchedDataDict(
                    {
                        "input_ids": ids.unsqueeze(0),
                        "input_lengths": torch.tensor([length]),
                    }
                )
            )["logprobs"][0, :length].cpu()
            recorded = generated["logprobs"][0, :length].cpu()
            error = (actual - recorded)[len(prompt) :]
            if not error.numel() or not torch.isfinite(error).all():
                raise ValueError(
                    "The response logprob comparison is empty or non-finite"
                )
            errors.append(error)
            token_mask = torch.zeros((1, length), dtype=torch.float32)
            token_mask[:, len(prompt) :] = 1
            training_rows.append(
                BatchedDataDict(
                    {
                        "input_ids": ids.unsqueeze(0),
                        "input_lengths": torch.tensor([length]),
                        "token_mask": token_mask,
                        "sample_mask": torch.ones(1),
                        "advantages": torch.ones((1, length)),
                        "prev_logprobs": actual.unsqueeze(0),
                        "generation_logprobs": recorded.unsqueeze(0),
                    }
                )
            )
            report["samples"].append(
                {
                    **{
                        key: value
                        for key, value in sample.items()
                        if key != "prompt_ids"
                    },
                    "prompt_length": len(prompt),
                    "length": length,
                    **error_stats(error),
                }
            )
            report["overall"] = error_stats(torch.cat(errors))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                "MOPD_CHECKPOINT_REFIT_PARITY",
                json.dumps(report["samples"][-1]),
                flush=True,
            )
        batch = BatchedDataDict.from_batches(
            training_rows,
            pad_value_dict={"input_ids": generation_config["_pad_token_id"]},
        )
        policy.prepare_for_training()
        loss_fn = ClippedPGLossFn(
            ClippedPGLossConfig(
                disable_ppo_ratio=True,
                reference_policy_kl_penalty=0,
                enable_torch_compile=False,
            )
        )
        native_metrics = policy.train(batch, loss_fn, gbs=len(samples), mbs=1)
        report["diagnostic_training"] = {
            "objective": "ClippedPGLossFn used to exercise the native model.train() forward; formal full-vocabulary OPD is validated separately.",
            "loss": float(native_metrics["loss"]),
            "grad_norm": None
            if native_metrics["grad_norm"] is None
            else float(native_metrics["grad_norm"]),
            "weights_observed": "All fresh responses and native training forwards use the reloaded checkpoint before this independent diagnostic optimizer attempt.",
        }
        report["actual_training_forward"] = training_forward_report(
            observation_dir, batch
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            "MOPD_CHECKPOINT_TRAINING_FORWARD",
            json.dumps(report["actual_training_forward"]),
            flush=True,
        )
    finally:
        generation.shutdown()
        policy.shutdown()
        generation_cluster.shutdown()
        train_cluster.shutdown()


if __name__ == "__main__":
    main()
