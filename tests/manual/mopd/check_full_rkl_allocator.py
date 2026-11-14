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
"""Exercise native full-RKL backward with deliberately fragmented CUDA memory."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from nemo_rl.distributed.model_utils import ChunkedDistributedReverseKLToFixedLogits


def memory() -> dict[str, int]:
    """Read allocated, cached, and non-releasable CUDA bytes."""
    stats = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "inactive_split": stats["inactive_split_bytes.all.current"],
        "free": free,
        "total": total,
        "allocation_retries": stats["num_alloc_retries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--fragment-pairs", type=int, required=True)
    parser.add_argument("--half-block-bytes", type=int, required=True)
    parser.add_argument("--free-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "python": sys.version,
        "torch": torch.__version__,
        "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "inputs": vars(args) | {"output": str(args.output)},
        "limitation": "Synthetic allocation history; uses the failing native backward, not model replay.",
    }
    if args.device == "cuda":
        torch.cuda.set_device(0)
        assert "H800" in torch.cuda.get_device_name()
        result["gpu"] = torch.cuda.get_device_name()
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group(
            "gloo", init_method=f"file://{directory}/rendezvous", rank=0, world_size=1
        )
        student = torch.zeros(
            (1, args.sequence_length, args.vocab_size),
            device=args.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher = torch.zeros_like(student)
        with torch.no_grad():
            teacher[..., 0] = 1
        loss = ChunkedDistributedReverseKLToFixedLogits.apply(
            student, teacher, args.chunk_size, dist.group.WORLD, False
        )
        retained, holes = [], []
        if args.device == "cuda":
            torch.cuda.empty_cache()
            for _ in range(args.fragment_pairs):
                block = torch.empty(
                    2 * args.half_block_bytes, device="cuda", dtype=torch.uint8
                )
                del block
                retained.append(
                    torch.empty(args.half_block_bytes, device="cuda", dtype=torch.uint8)
                )
                holes.append(
                    torch.empty(args.half_block_bytes, device="cuda", dtype=torch.uint8)
                )
            free, _ = torch.cuda.mem_get_info()
            ballast_bytes = free - args.free_bytes
            assert ballast_bytes > 0
            retained.append(
                torch.empty(ballast_bytes, device="cuda", dtype=torch.uint8)
            )
            holes.clear()
            torch.cuda.synchronize()
            result["before_backward"] = memory()
        try:
            loss.sum().backward()
        except torch.OutOfMemoryError as error:
            result["status"] = "OOM"
            result["error"] = str(error)
        else:
            result["status"] = "PASS"
            expected = torch.full(
                (args.vocab_size,),
                1 / args.vocab_size**2,
                device=args.device,
                dtype=torch.float32,
            )
            expected[0] -= 1 / args.vocab_size
            max_error = 0.0
            for start in range(0, args.sequence_length, args.chunk_size):
                gradient = student.grad[:, start : start + args.chunk_size]
                assert torch.isfinite(gradient).all()
                max_error = max(max_error, (gradient - expected).abs().max().item())
            result["max_gradient_error"] = max_error
            assert max_error < 1e-6
        if args.device == "cuda":
            result["after_backward"] = memory()
            retained.clear()
            torch.cuda.empty_cache()
            result["empty_cache_completed"] = True
        dist.destroy_process_group()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
