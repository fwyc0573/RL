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
"""Exercise native checkpoint save and resume within one GPU allocation."""

import argparse
import subprocess
from pathlib import Path

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def main() -> None:
    """Run the ordinary controller twice with an unchanged scheduler horizon."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = parse_hydra_overrides(load_config(args.config), overrides)
    if config["checkpointing"]["checkpoint_must_save_by"] is None:
        raise ValueError(
            "Set checkpointing.checkpoint_must_save_by for the initial native run"
        )
    if not config["checkpointing"]["enabled"]:
        raise ValueError(
            "Checkpoint save and resume require checkpointing.enabled=true"
        )

    entrypoint = (
        Path(__file__).resolve().parents[3] / "examples/run_grpo_single_controller.py"
    )
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-u",
        str(entrypoint),
        "--config",
        str(args.config),
    ]
    print("MOPD_CHECKPOINT_PHASE initial", flush=True)
    subprocess.run([*command, *overrides], check=True)
    print("MOPD_CHECKPOINT_PHASE resume", flush=True)
    subprocess.run(
        [*command, *overrides, "checkpointing.checkpoint_must_save_by=null"],
        check=True,
    )
    print("MOPD_CHECKPOINT_PHASE complete", flush=True)


if __name__ == "__main__":
    main()
