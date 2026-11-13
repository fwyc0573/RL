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
"""Restore a recorded native scheduler after extending the controller horizon."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from megatron.bridge.training.config import (
    ConfigContainer,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from megatron.bridge.training.optim import _get_scheduler
from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def main() -> None:
    """Exercise native scheduler calculation and loading without model tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-state", type=Path, required=True)
    parser.add_argument("--train-iters", type=int, required=True)
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(
        parse_hydra_overrides(load_config(args.config), overrides), resolve=True
    )
    policy = config["policy"]
    recorded = json.loads(args.native_state.read_text())
    saved_schedulers = [
        entry["state"][0]["opt_param_scheduler"]
        for entry in recorded["native_state"]
        if entry["key"].startswith("common_state/")
    ]
    if not saved_schedulers or any(
        state != saved_schedulers[0] for state in saved_schedulers
    ):
        raise ValueError("The native state must contain identical saved schedulers")
    saved = saved_schedulers[0]
    groups = next(
        entry["state"][0]["param_groups"]
        for entry in recorded["native_state"]
        if ".optimizer/" in entry["key"]
    )
    optimizer = SimpleNamespace(param_groups=copy.deepcopy(groups))
    scheduler_config = SchedulerConfig(**policy["megatron_cfg"]["scheduler"])
    training_config = TrainingConfig(
        train_iters=args.train_iters,
        global_batch_size=policy["train_global_batch_size"],
    )
    ConfigContainer._calculate_scheduler_steps(
        SimpleNamespace(train=training_config, scheduler=scheduler_config)
    )
    optimizer_config = OptimizerConfig(
        lr=policy["megatron_cfg"]["optimizer"]["lr"],
        min_lr=policy["megatron_cfg"]["optimizer"]["min_lr"],
    )
    scheduler = _get_scheduler(optimizer_config, scheduler_config, optimizer)
    print(
        "MOPD_NATIVE_SCHEDULER_BEFORE_RESTORE",
        json.dumps({"configured": scheduler.state_dict(), "saved": saved}),
        flush=True,
    )
    scheduler.load_state_dict(copy.deepcopy(saved))
    assert scheduler.state_dict() == saved, "Native scheduler state changed on restore"
    before = [(group["lr"], group["weight_decay"]) for group in optimizer.param_groups]
    scheduler.step(increment=training_config.global_batch_size)
    assert scheduler.num_steps == saved["num_steps"] + training_config.global_batch_size
    after = [(group["lr"], group["weight_decay"]) for group in optimizer.param_groups]
    if saved["lr_decay_style"] == "constant" and saved["wd_incr_style"] == "constant":
        assert after == before, "The constant LR or weight decay changed after resume"
    print(
        "MOPD_NATIVE_SCHEDULER_RESTORE_AND_NEXT_STEP_PASS",
        json.dumps(
            {
                "native_checkpoint": recorded["checkpoint"],
                "saved_copies": len(saved_schedulers),
                "restored_num_steps": saved["num_steps"],
                "next_num_steps": scheduler.num_steps,
                "lr_decay_steps": scheduler.lr_decay_steps,
                "wd_incr_steps": scheduler.wd_incr_steps,
                "parameter_group_lr_wd": after,
                "tensor_payloads_read": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
