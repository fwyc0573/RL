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
"""Check full-vocabulary reconstruction and gradients on initial HF checkpoints.

This reference control does not exercise Megatron, vLLM, or payload transport.
"""

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo_rl.algorithms.loss.utils import reconstruct_opd_full_teacher_logits
from nemo_rl.distributed.model_utils import ChunkedDistributedReverseKLToFixedLogits


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Summarize signed and absolute element errors in FP32."""
    error = (actual.float() - expected.float()).flatten()
    absolute = error.abs()
    quantiles = absolute.quantile(torch.tensor([0.5, 0.95, 0.99], device=error.device))
    return {
        "signed_mean": error.mean().item(),
        "mae": absolute.mean().item(),
        "p50": quantiles[0].item(),
        "p95": quantiles[1].item(),
        "p99": quantiles[2].item(),
        "max": absolute.max().item(),
    }


def check_gradient(student: torch.Tensor, teacher: torch.Tensor) -> dict:
    """Compare the native full-vocabulary CUDA kernel with direct autograd."""
    actual_student = student.detach().clone().requires_grad_(True)
    actual = ChunkedDistributedReverseKLToFixedLogits.apply(
        actual_student, teacher, 3, None, False
    ).mean()
    actual.backward()
    reference_student = student.detach().clone().requires_grad_(True)
    student_logprobs = reference_student.float().log_softmax(-1)
    teacher_logprobs = teacher.float().log_softmax(-1)
    reference = (
        (student_logprobs.exp() * (student_logprobs - teacher_logprobs)).sum(-1).mean()
    )
    reference.backward()
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual_student.grad, reference_student.grad, rtol=1e-4, atol=1e-5
    )
    assert torch.isfinite(actual_student.grad).all()
    reference_value = reference.item()
    absolute_error = abs(actual.item() - reference_value)
    return {
        "loss": actual.item(),
        "reference_loss": reference_value,
        "loss_absolute_error": absolute_error,
        "loss_relative_error": (
            absolute_error / abs(reference_value) if reference_value else None
        ),
        "logit_gradient_norm": actual_student.grad.float().norm().item(),
        "reference_logit_gradient_norm": reference_student.grad.float().norm().item(),
        "gradient_error": error_stats(actual_student.grad, reference_student.grad),
    }


def load_model(path: str) -> torch.nn.Module:
    """Load a frozen checkpoint with the PyTorch SDPA reference backend."""
    model = (
        AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    model.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--probe-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-tokens", type=int, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, required=True)
    args = parser.parse_args()
    assert args.response_tokens > 0 and args.max_prompt_tokens > 0
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    rendezvous = Path(tempfile.gettempdir()) / f"mopd-payload-{os.getpid()}"
    dist.init_process_group(
        "nccl", init_method=rendezvous.as_uri(), rank=0, world_size=1
    )
    recipe = yaml.safe_load(args.recipe.read_text())
    student_path = recipe["policy"]["model_name"]
    teachers = recipe["on_policy_distillation"]["teacher_model_by_agent_name"]
    tokenizer = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
    prompts = {}
    for line in args.probe_data.read_text().splitlines():
        row = json.loads(line)
        group = row["task_id"]["taskset"]
        messages = row["task_input"]["responses_create_params"]["input"]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if group not in prompts or len(ids) < len(prompts[group]["ids"]):
            prompts[group] = {"ids": ids, "task_id": row["task_id"]}
    assert set(prompts) == set(teachers)
    assert all(
        len(prompt["ids"]) <= args.max_prompt_tokens for prompt in prompts.values()
    )
    student = load_model(student_path)
    cases = {}
    with torch.inference_mode():
        for group, prompt in prompts.items():
            input_ids = torch.tensor([prompt["ids"]], device="cuda")
            token_ids = student.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                max_new_tokens=args.response_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            start = input_ids.shape[1] - 1
            assert token_ids.shape[1] > input_ids.shape[1]
            logits = student(token_ids, use_cache=False).logits[:, start:-1]
            cases[group] = {
                "token_ids": token_ids.cpu(),
                "start": start,
                "student_logits": logits.cpu(),
                "task_id": prompt["task_id"],
            }
    del student, logits
    gc.collect()
    torch.cuda.empty_cache()
    results = {
        "scope": "HF initial-checkpoint control; Megatron/vLLM/transport unverified",
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "models": {},
    }
    paths = {"student": student_path, **teachers}
    for name, path in paths.items():
        case = cases[next(iter(teachers)) if name == "student" else name]
        model = load_model(path)
        token_ids = case["token_ids"].cuda()
        start = case["start"]
        student_logits = case["student_logits"].cuda()
        with torch.inference_mode():
            output = model(token_ids, use_cache=False, output_hidden_states=True)
            direct = output.logits[:, start:-1].contiguous()
            hidden = output.hidden_states[-1][:, start:-1].contiguous()
            head = model.get_output_embeddings().weight
            reconstructed = reconstruct_opd_full_teacher_logits(
                hidden,
                teacher_payload="hidden_states",
                student_logits=student_logits,
                vocab_parallel_rank=0,
                context_parallel_group=None,
                teacher_output_layer_weight_by_index={0: head},
                teacher_index=torch.zeros(1, dtype=torch.long, device="cuda"),
            )
            direct_lp = direct.float().log_softmax(-1)
            reconstructed_lp = reconstructed.float().log_softmax(-1)
            fp32_head_lp = F.linear(hidden.float(), head.float()).log_softmax(-1)
            targets = token_ids[:, start + 1 :].unsqueeze(-1)
            selected_error = error_stats(
                reconstructed_lp.gather(-1, targets), direct_lp.gather(-1, targets)
            )
            assert selected_error["mae"] <= 0.02
            result = {
                "path": path,
                "task_id": case["task_id"],
                "prompt_tokens": start + 1,
                "response_tokens": targets.shape[1],
                "vocab_size": direct.shape[-1],
                "selected_logprob_error": selected_error,
                "full_distribution_logprob_error": error_stats(
                    reconstructed_lp, direct_lp
                ),
                "bf16_vs_fp32_head_logprob_error": error_stats(direct_lp, fp32_head_lp),
            }
        result["direct_gradient"] = check_gradient(student_logits, direct.clone())
        result["reconstructed_gradient"] = check_gradient(
            student_logits, reconstructed.clone()
        )
        results["models"][name] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(json.dumps({"model": name, **result}), flush=True)
        del model, output, head, direct, reconstructed, hidden
        gc.collect()
        torch.cuda.empty_cache()
    results["status"] = "PASS"
    args.output.write_text(json.dumps(results, indent=2))
    dist.destroy_process_group()
    print("MODEL_PAYLOAD_REFERENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
