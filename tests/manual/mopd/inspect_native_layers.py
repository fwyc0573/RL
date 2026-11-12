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
"""Capture native layer outputs at the positions predicting saved response tokens."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from nemo_rl.distributed import ray_actor_environment_registry, worker_groups
from nemo_rl.models.policy import lm_policy
from tests.manual.mopd import replay_logprobs
from tests.manual.mopd.native_forward_entrypoint import NATIVE_WORKER

OBSERVED_WORKER = "tests.manual.mopd.layer_output_worker.LayerOutputWorker"


class LayerOutputWorkerBuilder(worker_groups.RayWorkerBuilder):
    """Select the layer observer in the existing native worker environment."""

    def __init__(self, ray_actor_class_fqn: str, *args: Any, **kwargs: Any) -> None:
        if ray_actor_class_fqn == NATIVE_WORKER:
            ray_actor_class_fqn = OBSERVED_WORKER
        super().__init__(ray_actor_class_fqn, *args, **kwargs)


def main() -> None:
    """Use the fixed-token native replay with detached layer observations."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dump", type=Path, required=True)
    args, _ = parser.parse_known_args()
    rows = [json.loads(line) for line in args.dump.read_text().splitlines()]
    samples = {}
    for row in rows:
        for index, length in enumerate(row["input_lengths"]):
            mask = torch.tensor(row["token_loss_mask"][index][:length]) > 0
            mask[0] = False
            samples[tuple(row["token_ids"][index][:length])] = (
                row["sample_id"][index],
                mask.nonzero().flatten() - 1,
            )
    environment = Path(os.environ["NEMO_RL_VENV_DIR"]) / NATIVE_WORKER / "bin/python"
    if not environment.is_file():
        raise FileNotFoundError(environment)
    ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY[OBSERVED_WORKER] = str(
        environment
    )
    worker_groups.RayWorkerBuilder = LayerOutputWorkerBuilder
    lm_policy.RayWorkerBuilder = LayerOutputWorkerBuilder
    native_logprobs = lm_policy.Policy.get_logprobs

    def capture_logprobs(policy: lm_policy.Policy, data: Any, **kwargs: Any) -> Any:
        length = int(data["input_lengths"][0])
        sample_id, positions = samples[tuple(data["input_ids"][0, :length].tolist())]
        data["capture_sample_id"] = [sample_id]
        data["capture_positions"] = [positions.tolist()]
        return native_logprobs(policy, data, **kwargs)

    lm_policy.Policy.get_logprobs = capture_logprobs
    replay_logprobs.main()


if __name__ == "__main__":
    main()
