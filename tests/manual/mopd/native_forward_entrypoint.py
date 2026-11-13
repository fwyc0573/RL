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
"""Run the native controller with observations inside its Megatron forwards."""

import os
from pathlib import Path
from typing import Any

from examples import run_grpo_single_controller
from nemo_rl.distributed import ray_actor_environment_registry, worker_groups
from nemo_rl.models.policy import lm_policy

NATIVE_WORKER = (
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
)
OBSERVED_WORKER = "tests.manual.mopd.native_forward_worker.NativeForwardWorker"


class NativeForwardWorkerBuilder(worker_groups.RayWorkerBuilder):
    """Select the observing subclass for native Megatron workers only."""

    def __init__(self, ray_actor_class_fqn: str, *args: Any, **kwargs: Any) -> None:
        if ray_actor_class_fqn == NATIVE_WORKER:
            ray_actor_class_fqn = OBSERVED_WORKER
        super().__init__(ray_actor_class_fqn, *args, **kwargs)


def select_observed_workers() -> None:
    """Select the observing subclass in the mounted Megatron environment."""
    environment = Path(os.environ["NEMO_RL_VENV_DIR"]) / NATIVE_WORKER / "bin/python"
    if not environment.is_file():
        raise FileNotFoundError(environment)
    ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY[OBSERVED_WORKER] = str(
        environment
    )
    worker_groups.RayWorkerBuilder = NativeForwardWorkerBuilder
    lm_policy.RayWorkerBuilder = NativeForwardWorkerBuilder


def main() -> None:
    """Install test-local worker selection and use the ordinary entrypoint."""
    select_observed_workers()
    run_grpo_single_controller.main()


if __name__ == "__main__":
    main()
