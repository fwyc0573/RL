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
"""Observe eager vLLM layers on the native fixed-token prompt replay."""

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from tests.manual.mopd import generation_layer_outputs, replay_generation_logprobs


def main() -> None:
    """Use native model hooks at every position predicting a recorded response."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = parse_hydra_overrides(load_config(args.config), overrides)
    if not config.policy.generation.vllm_cfg.enforce_eager:
        raise ValueError(
            "Layer hooks require policy.generation.vllm_cfg.enforce_eager=true"
        )
    destination = str(Path(config.logger.log_dir) / "layer-outputs")
    samples = {}
    for row in map(json.loads, args.dump.read_text().splitlines()):
        for index, length in enumerate(row["input_lengths"]):
            mask = torch.tensor(row["token_loss_mask"][index][:length]) > 0
            mask[0] = False
            samples[tuple(row["token_ids"][index][:length])] = (
                row["sample_id"][index],
                (mask.nonzero().flatten() - 1).tolist(),
            )
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    # vLLM is only needed after the CPU-readable configuration and input checks.
    import vllm

    class LayerOutputLLM(vllm.LLM):
        def generate(self, prompts: Any, *args: Any, **kwargs: Any) -> Any:
            sample_id, positions = samples[tuple(prompts["prompt_token_ids"])]
            self.apply_model(
                partial(
                    generation_layer_outputs.start_capture,
                    sample_id=sample_id,
                    positions=positions,
                )
            )
            result = super().generate(prompts, *args, **kwargs)
            self.apply_model(
                partial(generation_layer_outputs.save_capture, destination=destination)
            )
            return result

    vllm.LLM = LayerOutputLLM
    replay_generation_logprobs.main()


if __name__ == "__main__":
    main()
