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
"""Capture selected eager vLLM decoder outputs through its model inspection API."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch


@dataclass
class LayerOutputs:
    """One active sample's selected model states inside a vLLM worker."""

    sample_id: str
    positions: torch.Tensor
    stages: dict[str, dict[str, torch.Tensor]]
    handles: list[Any]
    current_positions: torch.Tensor | None
    current_stage: str | None


_CAPTURE: LayerOutputs | None = None


def _record(
    capture: LayerOutputs,
    stage: str,
    positions: torch.Tensor,
    hidden: torch.Tensor,
) -> None:
    selected = torch.isin(positions, capture.positions.to(positions.device))
    if selected.any():
        capture.stages[stage] = {
            "positions": positions[selected].detach().cpu(),
            "hidden_states": hidden[selected].detach().cpu(),
        }


def _record_layer(
    capture: LayerOutputs,
    stage: str,
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    del module
    positions, _, _ = inputs
    capture.current_positions = positions
    hidden, residual = output
    combined = (hidden.float() + residual.float()).to(hidden.dtype)
    _record(capture, stage, positions, combined)


def _record_layer_input(
    capture: LayerOutputs,
    stage: str,
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
) -> None:
    del module
    positions, hidden, residual = inputs
    capture.current_positions = positions
    capture.current_stage = stage
    if stage == "layer_0":
        assert residual is None
        # Fused normalization can mutate input storage during the layer call.
        _record(capture, "embedding", positions, hidden)


def _record_tensor(
    capture: LayerOutputs,
    stage: str,
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
    output: torch.Tensor | tuple[torch.Tensor, Any],
) -> None:
    del module, inputs
    assert capture.current_positions is not None
    tensor = output[0] if isinstance(output, tuple) else output
    _record(capture, stage, capture.current_positions, tensor)


def _record_rope_input(
    capture: LayerOutputs,
    stage: str,
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
) -> None:
    if capture.current_stage != stage:
        return
    positions, query, key = inputs
    _record(capture, stage + "_q_before_rope", positions, query)
    _record(capture, stage + "_k_before_rope", positions, key)
    _record(capture, "rope_cos_sin", positions, module.cos_sin_cache[positions])


def _record_rope_output(
    capture: LayerOutputs,
    stage: str,
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    del module
    if capture.current_stage != stage:
        return
    positions = inputs[0]
    _record(capture, stage + "_q_after_rope", positions, output[0])
    _record(capture, stage + "_k_after_rope", positions, output[1])


def _record_norm(
    capture: LayerOutputs,
    stage: str,
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> None:
    del module, inputs
    assert capture.current_positions is not None
    hidden = output[0] if isinstance(output, tuple) else output
    _record(capture, stage, capture.current_positions, hidden)
    if isinstance(output, tuple):
        _record(capture, stage + "_residual", capture.current_positions, output[1])


def _record_norm_input(
    capture: LayerOutputs,
    stage: str,
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
) -> None:
    del module
    assert capture.current_positions is not None
    _record(capture, stage + "_hidden", capture.current_positions, inputs[0])
    if len(inputs) == 2 and inputs[1] is not None:
        _record(
            capture, stage + "_residual_input", capture.current_positions, inputs[1]
        )


def start_capture(
    model: torch.nn.Module, *, sample_id: str, positions: list[int]
) -> None:
    """Attach hooks for the next eager native engine request on this worker."""
    global _CAPTURE
    assert _CAPTURE is None
    capture = LayerOutputs(sample_id, torch.tensor(positions), {}, [], None, None)
    for index, layer in enumerate(model.model.layers):
        capture.handles.append(
            layer.register_forward_pre_hook(
                partial(_record_layer_input, capture, f"layer_{index}")
            )
        )
        if index < 2:
            attention = layer.self_attn
            for suffix, norm in [
                ("input_norm", layer.input_layernorm),
                ("post_attention_norm", layer.post_attention_layernorm),
            ]:
                stage = f"layer_{index}_{suffix}"
                capture.handles.extend(
                    [
                        norm.register_forward_pre_hook(
                            partial(_record_norm_input, capture, stage)
                        ),
                        norm.register_forward_hook(
                            partial(_record_norm, capture, stage)
                        ),
                    ]
                )
            capture.handles.extend(
                [
                    attention.qkv_proj.register_forward_hook(
                        partial(_record_tensor, capture, f"layer_{index}_qkv")
                    ),
                    attention.rotary_emb.register_forward_pre_hook(
                        partial(_record_rope_input, capture, f"layer_{index}")
                    ),
                    attention.rotary_emb.register_forward_hook(
                        partial(_record_rope_output, capture, f"layer_{index}")
                    ),
                    attention.attn.register_forward_hook(
                        partial(_record_tensor, capture, f"layer_{index}_attention")
                    ),
                    attention.o_proj.register_forward_hook(
                        partial(
                            _record_tensor,
                            capture,
                            f"layer_{index}_attention_projection",
                        )
                    ),
                    layer.mlp.gate_up_proj.register_forward_hook(
                        partial(
                            _record_tensor, capture, f"layer_{index}_mlp_projection"
                        )
                    ),
                    layer.mlp.act_fn.register_forward_hook(
                        partial(
                            _record_tensor, capture, f"layer_{index}_mlp_activation"
                        )
                    ),
                    layer.mlp.down_proj.register_forward_hook(
                        partial(_record_tensor, capture, f"layer_{index}_mlp_output")
                    ),
                ]
            )
        capture.handles.append(
            layer.register_forward_hook(
                partial(_record_layer, capture, f"layer_{index}")
            )
        )
    capture.handles.append(
        model.model.norm.register_forward_hook(
            partial(_record_norm, capture, "final_norm")
        )
    )
    _CAPTURE = capture


def _save_forward(
    capture: LayerOutputs,
    destination: str,
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> None:
    del module, inputs, output
    positions = capture.current_positions
    assert positions is not None
    if not torch.isin(positions, capture.positions.to(positions.device)).any():
        return
    final_positions = capture.stages["final_norm"]["positions"]
    path = Path(destination)
    path.relative_to("/mnt/codesign-exp/ycfeng")
    path.mkdir(parents=True, exist_ok=True)
    rank = torch.distributed.get_rank()
    torch.save(
        {"sample_id": capture.sample_id, "rank": rank, "stages": capture.stages},
        path
        / f"{capture.sample_id}-position-{int(final_positions.max())}-rank-{rank}.pt",
    )


def start_forward_capture(
    model: torch.nn.Module,
    *,
    sample_id: str,
    positions: list[int],
    destination: str,
) -> None:
    """Save each selected eager forward through an importable engine callback."""
    start_capture(model, sample_id=sample_id, positions=positions)
    assert _CAPTURE is not None
    _CAPTURE.handles.append(
        model.model.norm.register_forward_hook(
            partial(_save_forward, _CAPTURE, destination)
        )
    )


def save_capture(model: torch.nn.Module, *, destination: str) -> None:
    """Retain selected states and remove the completed request's hooks."""
    del model
    global _CAPTURE
    assert _CAPTURE is not None
    capture = _CAPTURE
    path = Path(destination)
    path.relative_to("/mnt/codesign-exp/ycfeng")
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sample_id": capture.sample_id,
            "rank": torch.distributed.get_rank(),
            "stages": capture.stages,
        },
        path / f"{capture.sample_id}-rank-{torch.distributed.get_rank()}.pt",
    )
    for handle in capture.handles:
        handle.remove()
    _CAPTURE = None
