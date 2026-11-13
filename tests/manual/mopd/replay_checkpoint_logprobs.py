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
"""Reload native checkpoint weights, refit vLLM, and compare fresh response logprobs."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

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
        init_optimizer=False,
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
    finally:
        generation.shutdown()
        policy.shutdown()
        generation_cluster.shutdown()
        train_cluster.shutdown()


if __name__ == "__main__":
    main()
