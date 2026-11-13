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
"""Observe actual native training logits and same-forward teacher reconstruction."""

import json
import os
import socket
from pathlib import Path
from typing import Any, Callable

import ray
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import _linear_forward

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn
from nemo_rl.algorithms.opd import opd_full_payload_field, opd_full_teacher_index_field
from nemo_rl.algorithms.loss.utils import reconstruct_opd_full_teacher_logits
from nemo_rl.distributed.model_utils import (
    ChunkedDistributedReverseKLToFixedLogits,
    _compute_distributed_log_softmax,
    get_next_token_logprobs_from_logits,
)
from nemo_rl.models.megatron import train
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class NativeForwardWorker(MegatronPolicyWorkerImpl):
    """Add detached observations while preserving native loss and optimizer calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._observation_dir = Path(os.environ["MOPD_OBSERVATION_DIR"])
        self._observation_dir.relative_to("/mnt/codesign-exp/ycfeng")
        self._training_observations = 0
        self._teacher_observations = 0
        native_prepare = train.prepare_loss_input
        native_teacher_processor = train.TeacherFullPayloadPostProcessor.__call__

        def observe_training(
            logits: torch.Tensor,
            data: dict[str, Any],
            loss_fn: LossFunction,
            **prepare_kwargs: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            prepared = native_prepare(logits, data, loss_fn, **prepare_kwargs)
            if loss_fn.input_type == LossInputType.OPD_FULL:
                assert isinstance(loss_fn, ClippedPGLossFn)
                self._observe_objective_gradient(logits, data, loss_fn, prepare_kwargs)
            if loss_fn.input_type in (LossInputType.OPD_FULL, LossInputType.LOGPROB):
                with torch.no_grad():
                    if loss_fn.input_type == LossInputType.LOGPROB:
                        logprobs = prepared[0]["next_token_logprobs"].detach()
                    else:
                        logprobs = get_next_token_logprobs_from_logits(
                            input_ids=data["input_ids"],
                            next_token_logits=logits.detach(),
                            vocab_parallel_rank=prepare_kwargs["vocab_parallel_rank"],
                            vocab_parallel_group=prepare_kwargs["vocab_parallel_group"],
                            context_parallel_group=prepare_kwargs[
                                "context_parallel_group"
                            ],
                            sampling_params=None,
                            chunk_size=self.cfg["logprob_chunk_size"],
                        )
                    self._training_observations += 1
                    if (
                        parallel_state.get_tensor_model_parallel_rank() == 0
                        and parallel_state.get_context_parallel_rank() == 0
                    ):
                        self._write_observation(
                            {
                                "kind": "training_forward",
                                "rank": torch.distributed.get_rank(),
                                "call": self._training_observations,
                                "logits_dtype": str(logits.dtype),
                                "input_ids": data["input_ids"].cpu().tolist(),
                                "token_mask": data["token_mask"].cpu().tolist(),
                                "sample_mask": data["sample_mask"].cpu().tolist(),
                                "input_lengths": data["input_lengths"].cpu().tolist(),
                                "next_token_logprobs": logprobs.cpu().tolist(),
                                "prev_logprobs": data["prev_logprobs"].cpu().tolist(),
                                "generation_logprobs": data["generation_logprobs"]
                                .cpu()
                                .tolist(),
                            }
                        )
            return prepared

        def observe_teacher(
            processor: train.TeacherFullPayloadPostProcessor, **processor_kwargs: Any
        ) -> Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]]:
            native_processor = native_teacher_processor(processor, **processor_kwargs)
            if processor.payload != "hidden_states":
                return native_processor

            def compare(
                output_tensor: torch.Tensor,
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                result = native_processor(output_tensor)
                with torch.no_grad():
                    owner = self._resolve_output_layer_owner()
                    weight = owner.output_layer.weight
                    if weight is None:
                        weight = owner.shared_embedding_or_output_weight()
                    hidden = (
                        processor_kwargs["hidden_states"].transpose(0, 1).contiguous()
                    )
                    reconstructed = reconstruct_opd_full_teacher_logits(
                        hidden,
                        teacher_payload="hidden_states",
                        student_logits=output_tensor,
                        vocab_parallel_rank=parallel_state.get_tensor_model_parallel_rank(),
                        context_parallel_group=None,
                        teacher_output_layer_weight_by_index={
                            0: weight.detach().to(dtype=output_tensor.dtype)
                        },
                    )
                    native_gemm_logits = _linear_forward(
                        hidden, weight.detach(), None, output_tensor.dtype
                    )
                    native_gemm_error = native_gemm_logits - output_tensor
                    logprobs_fn = processor._logprobs_post_processor(
                        data_dict=processor_kwargs["data_dict"],
                        input_ids=processor_kwargs["input_ids"],
                        cu_seqlens_padded=processor_kwargs["cu_seqlens_padded"],
                        original_seq_length=processor_kwargs["original_seq_length"],
                    )
                    _, reconstructed_outputs = logprobs_fn(reconstructed)
                    error = reconstructed.float() - output_tensor.float()
                    tp_group = parallel_state.get_tensor_model_parallel_group()
                    distribution_error = _compute_distributed_log_softmax(
                        reconstructed.float(), group=tp_group
                    ) - _compute_distributed_log_softmax(
                        output_tensor.float(), group=tp_group
                    )
                    with torch.enable_grad():
                        probe_logits = output_tensor.detach().float().requires_grad_()
                        reference_loss = ChunkedDistributedReverseKLToFixedLogits.apply(
                            probe_logits,
                            output_tensor.detach(),
                            self.cfg["logprob_chunk_size"],
                            tp_group,
                            False,
                        )
                        reconstructed_loss = (
                            ChunkedDistributedReverseKLToFixedLogits.apply(
                                probe_logits,
                                reconstructed.detach(),
                                self.cfg["logprob_chunk_size"],
                                tp_group,
                                False,
                            )
                        )
                        reference_gradient = torch.autograd.grad(
                            reference_loss.sum(), probe_logits, retain_graph=True
                        )[0]
                        reconstructed_gradient = torch.autograd.grad(
                            reconstructed_loss.sum(), probe_logits, retain_graph=True
                        )[0]
                        native_gemm_loss = (
                            ChunkedDistributedReverseKLToFixedLogits.apply(
                                probe_logits,
                                native_gemm_logits.detach(),
                                self.cfg["logprob_chunk_size"],
                                tp_group,
                                False,
                            )
                        )
                        native_gemm_gradient = torch.autograd.grad(
                            native_gemm_loss.sum(), probe_logits, retain_graph=True
                        )[0]
                    gradient_error = reconstructed_gradient - reference_gradient
                    native_gemm_gradient_error = (
                        native_gemm_gradient - reference_gradient
                    )
                    self._write_observation(
                        {
                            "kind": "teacher_logit_shard",
                            "call": self._teacher_observations + 1,
                            "rank": torch.distributed.get_rank(),
                            "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
                            "cp_rank": parallel_state.get_context_parallel_rank(),
                            "shape": list(output_tensor.shape),
                            "mae": error.abs().mean().item(),
                            "max": error.abs().max().item(),
                            "elements": error.numel(),
                            "full_logprob_mae": distribution_error.abs().mean().item(),
                            "full_logprob_max": distribution_error.abs().max().item(),
                            "loss_error_max": (reconstructed_loss - reference_loss)
                            .abs()
                            .max()
                            .item(),
                            "reference_loss_max": reference_loss.abs().max().item(),
                            "reconstructed_loss_max": reconstructed_loss.abs()
                            .max()
                            .item(),
                            "hidden_dtype": str(hidden.dtype),
                            "native_head_weight_dtype": str(weight.dtype),
                            "native_gemm_logit_mae": native_gemm_error.abs()
                            .mean()
                            .item(),
                            "native_gemm_logit_max": native_gemm_error.abs()
                            .max()
                            .item(),
                            "native_gemm_gradient_error_max": native_gemm_gradient_error.abs()
                            .max()
                            .item(),
                            "gradient_error_max": gradient_error.abs().max().item(),
                            "gradient_error_norm": gradient_error.norm().item(),
                            "reference_gradient_norm": reference_gradient.norm().item(),
                            "gradient_relative_error": None
                            if reference_gradient.norm().item() == 0
                            else (
                                gradient_error.norm() / reference_gradient.norm()
                            ).item(),
                        }
                    )
                    self._teacher_observations += 1
                    if (
                        parallel_state.get_tensor_model_parallel_rank() == 0
                        and parallel_state.get_context_parallel_rank() == 0
                    ):
                        data = processor_kwargs["data_dict"]
                        self._write_observation(
                            {
                                "kind": "teacher_same_forward",
                                "rank": torch.distributed.get_rank(),
                                "call": self._teacher_observations,
                                "head_dtype": str(weight.dtype),
                                "fp32_lm_head": bool(
                                    self.cfg["megatron_cfg"].get("fp32_lm_head")
                                ),
                                "direct_logits_dtype": str(output_tensor.dtype),
                                "local_logits_shape": list(output_tensor.shape),
                                "local_all_position_logit_mae": error.abs()
                                .mean()
                                .item(),
                                "local_all_position_logit_max": error.abs()
                                .max()
                                .item(),
                                "input_ids": data["input_ids"].cpu().tolist(),
                                "input_lengths": data["input_lengths"].cpu().tolist(),
                                "direct_logprobs": result[1]["logprobs"].cpu().tolist(),
                                "reconstructed_logprobs": reconstructed_outputs[
                                    "logprobs"
                                ]
                                .cpu()
                                .tolist(),
                            }
                        )
                return result

            return compare

        train.prepare_loss_input = observe_training
        train.TeacherFullPayloadPostProcessor.__call__ = observe_teacher

    def _observe_objective_gradient(
        self,
        logits: torch.Tensor,
        data: dict[str, Any],
        loss_fn: ClippedPGLossFn,
        prepare_kwargs: dict[str, Any],
    ) -> None:
        """Compare native and reconstructed targets with actual training masks."""
        full_cfg = loss_fn.opd_full
        payload = data[opd_full_payload_field(full_cfg)]
        index = data[opd_full_teacher_index_field(full_cfg)]
        heads = prepare_kwargs["teacher_output_layer_weight_by_index"]
        tp_group = prepare_kwargs["vocab_parallel_group"]
        with torch.no_grad():
            reconstructed = reconstruct_opd_full_teacher_logits(
                payload,
                teacher_payload=full_cfg.teacher_payload,
                student_logits=logits,
                vocab_parallel_rank=prepare_kwargs["vocab_parallel_rank"],
                context_parallel_group=prepare_kwargs["context_parallel_group"],
                teacher_output_layer_weight_by_index=heads,
                teacher_index=index,
            )
            payload = torch.nn.functional.pad(
                payload, (0, 0, 0, reconstructed.shape[1] - payload.shape[1])
            )
            native_target = torch.empty_like(reconstructed)
            for teacher_index in torch.unique(index).tolist():
                mask = index == teacher_index
                native_target[mask] = _linear_forward(
                    payload[mask].contiguous(),
                    heads[teacher_index].to(dtype=payload.dtype),
                    None,
                    reconstructed.dtype,
                )
            valid_tokens = (
                data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
            ).sum()
            valid_sequences = data["sample_mask"].sum()
        next_token_width = data["input_ids"].shape[1] - 1
        objectives = []
        gradients = []
        with torch.enable_grad():
            probe = logits.detach().float().requires_grad_()
            for target in (native_target, reconstructed):
                divergence = ChunkedDistributedReverseKLToFixedLogits.apply(
                    probe, target.detach(), full_cfg.chunk_size, tp_group, False
                )
                objective, _ = loss_fn(
                    data=data,
                    global_valid_seqs=valid_sequences,
                    global_valid_toks=valid_tokens,
                    opd_full_divergence=divergence[:, :next_token_width],
                )
                objectives.append(objective.item())
                gradients.append(torch.autograd.grad(objective, probe)[0])
        error = gradients[1] - gradients[0]
        self._write_observation(
            {
                "kind": "training_objective_gradient",
                "call": self._training_observations + 1,
                "rank": torch.distributed.get_rank(),
                "shape": list(logits.shape),
                "teacher_index": index.cpu().tolist(),
                "payload_dtype": str(payload.dtype),
                "reconstructed_dtype": str(reconstructed.dtype),
                "objective_valid_tokens": valid_tokens.item(),
                "objective_valid_sequences": valid_sequences.item(),
                "objective_normalization": "observed microbatch token count",
                "objective_losses": objectives,
                "objective_reference_gradient_norm": gradients[0].norm().item(),
                "objective_reconstructed_gradient_error_max": error.abs().max().item(),
                "objective_reconstructed_gradient_error_norm": error.norm().item(),
            }
        )

    def _write_observation(self, record: dict[str, Any]) -> None:
        """Record one detached comparison with a nonblocking I/O soft gate."""
        record.update(model=self.cfg["model_name"], pid=os.getpid())
        try:
            self._observation_dir.mkdir(parents=True, exist_ok=True)
            path = (
                self._observation_dir
                / f"forward-{socket.gethostname()}-{os.getpid()}.jsonl"
            )
            with path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError as error:
            print(
                "MOPD_FORWARD_OBSERVATION_UNVERIFIED", type(error).__name__, flush=True
            )
