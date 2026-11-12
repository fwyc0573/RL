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
"""Compare actual eager decode and full prefill using the same generated tokens."""

import argparse
import copy
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from tests.manual.mopd import generation_layer_outputs, replay_generation_logprobs


def main() -> None:
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
    assert (
        vllm_config["enforce_eager"] and vllm_config["logprobs_mode"] == "raw_logprobs"
    )
    if vllm_config.get("env_vars"):
        os.environ.update(vllm_config["env_vars"])
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    destination = Path(config["logger"]["log_dir"])
    destination.relative_to("/mnt/codesign-exp/ycfeng")
    samples = {}
    for row in map(json.loads, args.dump.read_text().splitlines()):
        for index, length in enumerate(row["input_lengths"]):
            samples[tuple(row["token_ids"][index][:length])] = (row, index)

    # vLLM is only needed after the CPU-readable configuration and input checks.
    import vllm

    generated_rows = []
    report = {"config": config, "source_dump": str(args.dump), "samples": []}

    class DecodeLLM(vllm.LLM):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if "enable_prefix_caching" in vllm_config:
                kwargs["enable_prefix_caching"] = vllm_config["enable_prefix_caching"]
            super().__init__(*args, **kwargs)

        def generate(self, prompts: Any, parameters: Any, **kwargs: Any) -> Any:
            row, index = samples[tuple(prompts["prompt_token_ids"])]
            sid = row["sample_id"][index]
            prompt = row["prompt_ids"][index]
            maximum = generation["max_new_tokens"]
            positions = list(range(len(prompt) - 1, len(prompt) + maximum - 1))
            self.apply_model(
                partial(
                    generation_layer_outputs.start_forward_capture,
                    sample_id=sid,
                    positions=positions,
                    destination=str(destination / "decode-layers"),
                )
            )
            actual_parameters = vllm.SamplingParams(
                temperature=generation["temperature"],
                top_p=generation["top_p"],
                top_k=generation["top_k"],
                max_tokens=maximum,
                logprobs=0,
                detokenize=False,
            )
            decoded = super().generate(
                {"prompt_token_ids": prompt}, actual_parameters, use_tqdm=False
            )[0]
            self.apply_model(
                partial(
                    generation_layer_outputs.save_capture,
                    destination=str(destination / "decode-last-layers"),
                )
            )
            response = decoded.outputs[0]
            tokens = prompt + list(response.token_ids)
            actual = torch.tensor(
                [
                    entry[token].logprob
                    for token, entry in zip(
                        response.token_ids, response.logprobs, strict=True
                    )
                ]
            )
            assert len(actual) > 0 and torch.isfinite(actual).all()
            selected = list(range(len(prompt) - 1, len(tokens) - 1))
            self.apply_model(
                partial(
                    generation_layer_outputs.start_forward_capture,
                    sample_id=sid,
                    positions=selected,
                    destination=str(destination / "prefill-layers"),
                )
            )
            prefilled = super().generate(
                {"prompt_token_ids": tokens}, parameters, use_tqdm=False
            )[0]
            self.apply_model(
                partial(
                    generation_layer_outputs.save_capture,
                    destination=str(destination / "prefill-last-layers"),
                )
            )
            previous = torch.tensor(
                [
                    prefilled.prompt_logprobs[position][tokens[position]].logprob
                    for position in range(len(prompt), len(tokens))
                ]
            )
            assert torch.isfinite(previous).all()
            item = {
                "sample_id": sid,
                "prompt_length": len(prompt),
                "length": len(tokens),
                "response_token_ids": list(response.token_ids),
                "decode_logprobs": actual.tolist(),
                "prefill_logprobs": previous.tolist(),
                "prefill_minus_decode": replay_generation_logprobs.error_stats(
                    previous - actual
                ),
            }
            report["samples"].append(item)
            joined = torch.cat(
                [
                    torch.tensor(sample["prefill_logprobs"])
                    - torch.tensor(sample["decode_logprobs"])
                    for sample in report["samples"]
                ]
            )
            report["overall"] = replay_generation_logprobs.error_stats(joined)
            observed = {
                "idx": row["idx"],
                "step": row["step"],
                "sample_id": [sid],
                "input_lengths": [len(tokens)],
                "metadata": [copy.deepcopy(row["metadata"][index])],
                "token_ids": [tokens],
                "token_loss_mask": [[0.0] * len(prompt) + [1.0] * len(actual)],
                "generation_logprobs": [[0.0] * len(prompt) + actual.tolist()],
                "prev_logprobs": [[0.0] * len(prompt) + previous.tolist()],
                "prompt_ids": [prompt],
            }
            observed["metadata"][0]["diagnostic_source"] = (
                "actual eager decode, followed by full prefill of identical generated tokens"
            )
            generated_rows.append(observed)
            (destination / "actual-decode-comparison.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            (destination / "actual-decode-tokens.jsonl").write_text(
                "".join(json.dumps(sample) + "\n" for sample in generated_rows)
            )
            print("MOPD_ACTUAL_DECODE_COMPARISON", json.dumps(item), flush=True)
            return super().generate(prompts, parameters, **kwargs)

    vllm.LLM = DecodeLLM
    replay_generation_logprobs.main()


if __name__ == "__main__":
    main()
