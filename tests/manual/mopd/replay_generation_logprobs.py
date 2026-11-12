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
"""Compare vLLM prompt logprobs with saved generation on fixed token sequences."""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nemo_rl.models.generation.vllm.patches import _apply_vllm_patches
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def error_stats(error: torch.Tensor) -> dict[str, float | int]:
    """Report the entire recorded response mask."""
    values = error.double().flatten()
    absolute = values.abs()
    quantiles = absolute.quantile(torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64))
    return {
        "tokens": values.numel(),
        "signed_mean": values.mean().item(),
        "mae": absolute.mean().item(),
        "p50": quantiles[0].item(),
        "p95": quantiles[1].item(),
        "p99": quantiles[2].item(),
        "max": absolute.max().item(),
    }


def main() -> None:
    """Use native LLM.generate prompt scores for all recorded response tokens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(
        parse_hydra_overrides(load_config(args.config), overrides), resolve=True
    )
    generation = config["policy"]["generation"]
    vllm_config = generation["vllm_cfg"]
    if vllm_config["fp32_lm_head"]:
        raise ValueError("This BF16 replay requires the original BF16 generation head")
    output = Path(config["logger"]["log_dir"]) / "replay-generation-logprobs.json"
    output.relative_to("/mnt/codesign-exp/ycfeng")
    rows = [json.loads(line) for line in args.dump.read_text().splitlines()]
    maximum_length = max(length for row in rows for length in row["input_lengths"])
    _apply_vllm_patches(sys.executable, extra_env_vars=[])
    # The vLLM engine requires a GPU; CLI and config reads can run on the CPU host.
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=config["policy"]["model_name"],
        tokenizer=config["policy"]["tokenizer"]["name"],
        tensor_parallel_size=vllm_config["tensor_parallel_size"],
        pipeline_parallel_size=vllm_config["pipeline_parallel_size"],
        dtype=vllm_config["precision"],
        gpu_memory_utilization=vllm_config["gpu_memory_utilization"],
        enforce_eager=vllm_config["enforce_eager"],
        trust_remote_code=True,
        distributed_executor_backend="ray",
        max_model_len=maximum_length + 1,
        logprobs_mode=vllm_config["logprobs_mode"],
        **generation["vllm_kwargs"],
    )
    parameters = SamplingParams(
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        top_k=generation["top_k"],
        max_tokens=1,
        prompt_logprobs=0,
        detokenize=False,
    )
    report = {
        "dump": str(args.dump),
        "model": config["policy"]["model_name"],
        "scored_sequence_length_max": maximum_length,
        "engine_max_model_len": maximum_length + 1,
        "comparison": "Synchronous native vLLM prompt scoring versus recorded async generation; the one new output token is unused.",
        "samples": [],
    }
    errors = []
    for row in rows:
        for index, length in enumerate(row["input_lengths"]):
            ids = row["token_ids"][index][:length]
            mask = torch.tensor(row["token_loss_mask"][index][:length]) > 0
            mask[0] = False
            assert (
                mask.any()
                and ids[: len(row["prompt_ids"][index])] == row["prompt_ids"][index]
            )
            result = engine.generate(
                {"prompt_token_ids": ids}, parameters, use_tqdm=False
            )[0]
            assert result.prompt_token_ids == ids
            positions = mask.nonzero().flatten().tolist()
            actual = torch.tensor(
                [
                    result.prompt_logprobs[position][ids[position]].logprob
                    for position in positions
                ]
            )
            assert torch.isfinite(actual).all()
            recorded = torch.tensor(row["generation_logprobs"][index][:length])[mask]
            previous = torch.tensor(row["prev_logprobs"][index][:length])[mask]
            errors.append(actual - recorded)
            report["samples"].append(
                {
                    "sample_id": row["sample_id"][index],
                    "length": length,
                    "response_logprobs": actual.tolist(),
                    "generation_error": error_stats(actual - recorded),
                    "recorded_scorer_error": error_stats(actual - previous),
                }
            )
            report["overall_generation_error"] = error_stats(torch.cat(errors))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                "MOPD_VLLM_PROMPT_REPLAY", json.dumps(report["samples"][-1]), flush=True
            )


if __name__ == "__main__":
    main()
