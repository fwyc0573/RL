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
"""Check native FP16 scaler backoff for token sums before final normalization."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed.nn.functional
from megatron.bridge.training.config import OptimizerConfig
from megatron.core.optimizer.grad_scaler import DynamicGradScaler
from omegaconf import OmegaConf

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.opd import OnPolicyDistillationFullConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import ChunkedDistributedReverseKLToFixedLogits
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def backward(*, scale: float, normalize_first: bool, tokens: int) -> dict:
    """Run the native full RKL through a shared FP16 gradient boundary on CPU."""
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(disable_ppo_ratio=True, reference_policy_kl_penalty=0.0),
        opd_full=OnPolicyDistillationFullConfig(enabled=True),
    )
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    narrow = parameter.to(torch.float16)
    logits = torch.stack([narrow.float(), torch.zeros_like(narrow).float()])
    logits = logits.reshape(1, 1, 2).expand(1, tokens, 2).contiguous()
    teacher = torch.tensor([-3.0, 3.0]).reshape(1, 1, 2).expand_as(logits)
    batch = BatchedDataDict(
        {"token_mask": torch.ones(1, tokens + 1), "sample_mask": torch.ones(1)}
    )
    with (
        patch("torch.distributed.all_reduce", return_value=None),
        patch(
            "torch.distributed.nn.functional.all_reduce",
            side_effect=lambda tensor, *args, **kwargs: tensor,
        ),
    ):
        divergence = ChunkedDistributedReverseKLToFixedLogits.apply(
            logits, teacher, 1024, None, False
        )
        loss, _ = loss_fn(
            data=batch,
            global_valid_seqs=torch.tensor(1.0),
            global_valid_toks=torch.tensor(float(tokens) if normalize_first else 1.0),
            opd_full_divergence=divergence,
        )
        (loss * scale).backward()
    gradient = parameter.grad.item()
    finite = bool(torch.isfinite(parameter.grad))
    return {
        "loss": loss.item(),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradient": finite,
        "normalized_gradient": (
            gradient / scale / (1 if normalize_first else tokens) if finite else None
        ),
    }


def check_scaler(*, floor: float, tokens: int) -> dict:
    """Exercise native backoff without CUDA allocations or model checkpoints."""
    native_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs) -> torch.Tensor:
        return native_tensor(*args, **dict(kwargs, device="cpu"))

    with patch("torch.tensor", side_effect=cpu_tensor):
        scaler = DynamicGradScaler(
            initial_scale=1.0,
            min_scale=floor,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1000,
            hysteresis=1,
        )
    attempts = []
    for _ in range(3):
        scale = scaler.scale.item()
        result = backward(scale=scale, normalize_first=False, tokens=tokens)
        attempts.append(dict(result, scale=scale))
        scaler.update(not result["finite_gradient"])
    return {"floor": floor, "attempts": attempts, "next_scale": scaler.scale.item()}


def main() -> None:
    """Validate the recipe's native scaler floor with the observed overflow pattern."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-state", type=Path)
    args = parser.parse_args()
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(args.config), resolve=True)
    policy = config["policy"]
    optimizer = OptimizerConfig(**policy["megatron_cfg"]["optimizer"])
    token_ceiling = policy["max_total_sequence_length"]
    expected_floor = 1.0 / (policy["train_global_batch_size"] * token_ceiling)
    reference = backward(scale=1.0, normalize_first=True, tokens=token_ceiling)
    old = check_scaler(floor=1.0, tokens=token_ceiling)
    configured = check_scaler(floor=optimizer.min_loss_scale, tokens=token_ceiling)
    result = {
        "config": str(args.config),
        "python": sys.version,
        "torch": torch.__version__,
        "reference": reference,
        "old_floor": old,
        "configured_floor": configured,
        "expected_floor": expected_floor,
        "scope": "Native full RKL on two vocabulary entries, a shared FP16 boundary, and the configured token ceiling; not an original-model replay.",
    }
    if args.native_state is not None:
        native_state = json.loads(args.native_state.read_text())
        saved_scalers = [
            row["state"][0]
            for row in native_state["native_state"]
            if ".grad_scaler/" in row["key"]
        ]
        native_tensor = torch.tensor

        def cpu_tensor(*values, **kwargs) -> torch.Tensor:
            return native_tensor(*values, **dict(kwargs, device="cpu"))

        restored = []
        for saved in saved_scalers:
            with patch("torch.tensor", side_effect=cpu_tensor):
                scaler = DynamicGradScaler(
                    initial_scale=optimizer.initial_loss_scale,
                    min_scale=optimizer.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=optimizer.loss_scale_window,
                    hysteresis=optimizer.hysteresis,
                )
            with (
                patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor),
                patch("torch.cuda.current_device", return_value=0),
            ):
                scaler.load_state_dict(
                    {
                        "scale": torch.tensor(saved["scale"]),
                        "growth_tracker": saved["growth_tracker"],
                        "hysteresis_tracker": saved["hysteresis_tracker"],
                    }
                )
            assert scaler.scale.tolist() == saved["scale"]
            assert scaler.min_scale.item() == expected_floor
            scaler.update(found_inf=True)
            restored.append(
                {"saved_scale": saved["scale"], "next_scale": scaler.scale.item()}
            )
        assert len(restored) == 8
        result["native_restore"] = {
            "source": str(args.native_state),
            "scalers": restored,
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert reference["finite_gradient"]
    assert all(not attempt["finite_gradient"] for attempt in old["attempts"])
    assert optimizer.min_loss_scale == expected_floor
    assert configured["attempts"][1]["finite_gradient"]
    assert (
        abs(
            configured["attempts"][1]["normalized_gradient"]
            - reference["normalized_gradient"]
        )
        < 0.001
    )
    print("MOPD_FP16_RAW_SUM_SCALER_PASS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
