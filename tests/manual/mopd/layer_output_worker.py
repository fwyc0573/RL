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
"""Record selected native layer outputs while replaying a saved sequence."""

import os
from functools import partial
from pathlib import Path
from typing import Any

import ray
import torch
from megatron.core import parallel_state

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class LayerOutputWorker(MegatronPolicyWorkerImpl):
    """Observe the embedding, decoder layers, and final normalized hidden states."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._positions: torch.Tensor | None = None
        self._input_length = 0
        self._layer_outputs: dict[str, dict[str, torch.Tensor]] = {}
        owner = self._resolve_output_layer_owner()
        owner.embedding.register_forward_hook(partial(self._record_output, "embedding"))
        for index, layer in enumerate(owner.decoder.layers):
            layer.register_forward_hook(partial(self._record_output, f"layer_{index}"))
            if index < 2:
                attention = layer.self_attention
                attention.linear_qkv.register_forward_hook(
                    partial(self._record_output, f"layer_{index}_qkv")
                )
                attention.core_attention.register_forward_pre_hook(
                    partial(self._record_attention_input, f"layer_{index}")
                )
                attention.core_attention.register_forward_hook(
                    partial(self._record_attention_output, f"layer_{index}_attention")
                )
                attention.linear_proj.register_forward_hook(
                    partial(self._record_output, f"layer_{index}_attention_projection")
                )
                layer.mlp.linear_fc1.register_forward_hook(
                    partial(self._record_output, f"layer_{index}_mlp_projection")
                )
                layer.mlp.linear_fc2.register_forward_pre_hook(
                    partial(self._record_input, f"layer_{index}_mlp_activation")
                )
                layer.mlp.linear_fc2.register_forward_hook(
                    partial(self._record_output, f"layer_{index}_mlp_output")
                )
        owner.rotary_pos_emb.register_forward_hook(
            partial(self._record_attention_output, "rope_angles")
        )
        owner.decoder.final_layernorm.register_forward_hook(
            partial(self._record_output, "final_norm")
        )

    def _record_output(
        self,
        stage: str,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: torch.Tensor | tuple[torch.Tensor, Any],
    ) -> None:
        del module, inputs
        if self._positions is None:
            return
        hidden = output[0] if isinstance(output, tuple) else output
        length = hidden.shape[0]
        offset = 0
        if length != self._input_length:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if length * tp_size != self._input_length:
                raise ValueError(
                    "Layer sequence shape does not match the replayed input"
                )
            offset = parallel_state.get_tensor_model_parallel_rank() * length
        positions = self._positions
        selected = positions[(positions >= offset) & (positions < offset + length)]
        self._layer_outputs[stage] = {
            "positions": selected.clone(),
            "hidden_states": hidden[selected.to(hidden.device) - offset, 0]
            .detach()
            .cpu(),
        }

    def _record_input(
        self, stage: str, module: torch.nn.Module, inputs: tuple[Any, ...]
    ) -> None:
        self._record_output(stage, module, (), inputs[0])

    def _record_attention_rows(self, stage: str, tensor: torch.Tensor) -> None:
        if self._positions is None:
            return
        if tensor.shape[0] != self._input_length:
            raise ValueError("Attention capture requires the full CP1 sequence")
        selected = self._positions.to(tensor.device)
        self._layer_outputs[stage] = {
            "positions": self._positions.clone(),
            "hidden_states": tensor[selected].reshape(len(selected), -1).detach().cpu(),
        }

    def _record_attention_input(
        self, stage: str, module: torch.nn.Module, inputs: tuple[Any, ...]
    ) -> None:
        del module
        self._record_attention_rows(stage + "_q_after_rope", inputs[0])
        self._record_attention_rows(stage + "_k_after_rope", inputs[1])

    def _record_attention_output(
        self,
        stage: str,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        self._record_attention_rows(stage, output)

    def get_logprobs(self, *, data: BatchedDataDict[Any], **kwargs: Any) -> Any:
        """Run the native scorer and retain its selected hidden-state rows."""
        self._positions = torch.tensor(data.pop("capture_positions")[0])
        sample_id = data.pop("capture_sample_id")[0]
        self._input_length = int(data["input_lengths"][0])
        self._layer_outputs = {}
        try:
            result = super().get_logprobs(data=data, **kwargs)
            destination = Path(os.environ["MOPD_LAYER_OUTPUT_DIR"])
            destination.relative_to("/mnt/codesign-exp/ycfeng")
            destination.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "sample_id": sample_id,
                    "input_length": self._input_length,
                    "rank": torch.distributed.get_rank(),
                    "stages": self._layer_outputs,
                },
                destination / f"{sample_id}-rank-{torch.distributed.get_rank()}.pt",
            )
            return result
        finally:
            self._positions = None
