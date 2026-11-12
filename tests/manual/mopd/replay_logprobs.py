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
"""Replay recorded token sequences through native Megatron logprob forwards."""

import argparse
import copy
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.algorithms.single_controller_utils.setup import (
    _maybe_inject_megatron_train_iters,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def error_stats(error: torch.Tensor) -> dict[str, float | int]:
    """Report response-token errors without changing the recorded mask."""
    error = error.double().flatten()
    absolute = error.abs()
    quantiles = absolute.quantile(torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64))
    return {
        "tokens": error.numel(),
        "signed_mean": error.mean().item(),
        "mae": absolute.mean().item(),
        "p50": quantiles[0].item(),
        "p95": quantiles[1].item(),
        "p99": quantiles[2].item(),
        "max": absolute.max().item(),
    }


def main() -> None:
    """Use Policy.get_logprobs on identical saved token IDs for each CP size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--context-parallel-sizes", type=int, nargs="+", required=True)
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    loaded = parse_hydra_overrides(load_config(args.config), overrides)
    config = MasterConfig(**OmegaConf.to_container(loaded, resolve=True))
    _maybe_inject_megatron_train_iters(config)
    output = Path(config.logger.log_dir) / "replay-logprobs.json"
    output.relative_to("/mnt/codesign-exp/ycfeng")
    rows = [json.loads(line) for line in args.dump.read_text().splitlines()]
    samples = []
    for row in rows:
        for index, length in enumerate(row["input_lengths"]):
            ids = row["token_ids"][index][:length]
            mask = torch.tensor(row["token_loss_mask"][index][:length]) > 0
            mask[0] = False
            assert (
                mask.any()
                and ids[: len(row["prompt_ids"][index])] == row["prompt_ids"][index]
            )
            samples.append(
                {
                    "sample_id": row["sample_id"][index],
                    "input_ids": ids,
                    "mask": mask,
                    "generation": torch.tensor(
                        row["generation_logprobs"][index][:length]
                    ),
                    "previous": torch.tensor(row["prev_logprobs"][index][:length]),
                }
            )
    assert samples
    init_ray()
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    results = {
        "dump": str(args.dump),
        "model": config.policy["model_name"],
        "variants": [],
    }
    first_logprobs = None
    for cp_size in args.context_parallel_sizes:
        policy_config = copy.deepcopy(config.policy)
        policy_config["megatron_cfg"]["context_parallel_size"] = cp_size
        world_size = (
            cp_size
            * policy_config["megatron_cfg"]["tensor_model_parallel_size"]
            * policy_config["megatron_cfg"]["pipeline_model_parallel_size"]
        )
        assert 0 < world_size <= config.cluster.gpus_per_node
        cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[world_size],
            num_gpus_per_node=config.cluster.gpus_per_node,
            name=f"mopd_replay_cp{cp_size}",
        )
        policy = Policy(
            cluster=cluster,
            config=policy_config,
            tokenizer=tokenizer,
            name_prefix=f"mopd_replay_cp{cp_size}",
            init_optimizer=False,
            init_reference_model=False,
        )
        try:
            policy.prepare_for_lp_inference()
            logprobs = []
            sample_results = []
            for sample in samples:
                length = len(sample["input_ids"])
                batch = BatchedDataDict(
                    {
                        "input_ids": torch.tensor(
                            [sample["input_ids"]], dtype=torch.long
                        ),
                        "input_lengths": torch.tensor([length], dtype=torch.long),
                    }
                )
                actual = policy.get_logprobs(batch)["logprobs"][0, :length].cpu()
                assert torch.isfinite(actual).all()
                logprobs.append(actual[sample["mask"]])
                sample_results.append(
                    {
                        "sample_id": sample["sample_id"],
                        "length": length,
                        "response_logprobs": actual[sample["mask"]].tolist(),
                        "generation_error": error_stats(
                            (actual - sample["generation"])[sample["mask"]]
                        ),
                        "recorded_scorer_error": error_stats(
                            (actual - sample["previous"])[sample["mask"]]
                        ),
                    }
                )
            joined = torch.cat(logprobs)
            variant = {
                "tensor_model_parallel_size": policy_config["megatron_cfg"][
                    "tensor_model_parallel_size"
                ],
                "pipeline_model_parallel_size": policy_config["megatron_cfg"][
                    "pipeline_model_parallel_size"
                ],
                "context_parallel_size": cp_size,
                "sequence_packing": policy_config["sequence_packing"]["enabled"],
                "apply_rope_fusion": policy_config["megatron_cfg"]["apply_rope_fusion"],
                "samples": sample_results,
                "against_first_variant": None
                if first_logprobs is None
                else error_stats(joined - first_logprobs),
            }
            if first_logprobs is None:
                first_logprobs = joined
            results["variants"].append(variant)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2) + "\n")
            print("MOPD_REPLAY_VARIANT", json.dumps(variant), flush=True)
        finally:
            policy.shutdown()
            cluster.shutdown()


if __name__ == "__main__":
    main()
