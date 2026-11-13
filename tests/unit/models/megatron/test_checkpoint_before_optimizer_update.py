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

from unittest.mock import patch

import pytest
import torch


@pytest.mark.mcore
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chained", [False, True])
def test_checkpoint_before_first_update(dtype: torch.dtype, chained: bool) -> None:
    # Megatron and TE are optional outside the mcore test lane.
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer
    from transformer_engine.pytorch.optimizers import FusedAdam

    from nemo_rl.models.megatron.setup import initialize_optimizer_state_for_checkpoint

    parameter = torch.nn.Parameter(torch.tensor([1.25, -0.5], dtype=dtype))
    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    with patch("torch.tensor", side_effect=cpu_tensor):
        adam = FusedAdam(
            [parameter],
            master_weights=True,
            use_decoupled_grad=True,
            store_param_remainders=True,
        )
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.optimizer = adam
    optimizer.config = OptimizerConfig(
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        use_precision_aware_optimizer=True,
    )
    optimizer.is_stub_optimizer = False
    optimizer.model_param_group_index_map = {parameter: (0, 0)}
    wrapper = ChainedOptimizer([optimizer]) if chained else optimizer
    before = parameter.detach().clone()
    initial_groups = adam.state_dict()["param_groups"]

    initialize_optimizer_state_for_checkpoint(wrapper)
    states = optimizer._get_main_param_and_optimizer_states(parameter)

    assert set(states) == {"param", "exp_avg", "exp_avg_sq"}
    if dtype == torch.bfloat16:
        assert states["param"].dtype == torch.int16
        assert torch.count_nonzero(states["param"]) == 0
    else:
        torch.testing.assert_close(states["param"], before.float(), rtol=0, atol=0)
    assert torch.count_nonzero(states["exp_avg"]) == 0
    assert torch.count_nonzero(states["exp_avg_sq"]) == 0
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert adam.state_dict()["param_groups"] == initial_groups

    states["exp_avg"].fill_(0.25)
    states["exp_avg_sq"].fill_(0.5)
    if dtype == torch.float16:
        states["param"].fill_(1.00001)
    adam.param_groups[0]["step"] = 7
    saved = {key: value.clone() for key, value in states.items()}
    initialize_optimizer_state_for_checkpoint(wrapper)
    after = optimizer._get_main_param_and_optimizer_states(parameter)
    for key, value in saved.items():
        torch.testing.assert_close(after[key], value, rtol=0, atol=0)
        assert after[key].data_ptr() == states[key].data_ptr()
    assert adam.param_groups[0]["step"] == 7
