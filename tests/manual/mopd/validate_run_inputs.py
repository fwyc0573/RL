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
"""Validate formal MOPD inputs on CPU and check their identity before model startup."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from prepare_inputs import prepare_conversations, verify_models
from safetensors import safe_open

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def file_identity(path: Path) -> dict[str, int]:
    """Record inexpensive change indicators for an already validated file."""
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def check(report_path: Path) -> None:
    """Check the exact prepared files without repeating weight or dataset scans."""
    report = json.loads(report_path.read_text())
    if report["status"] != "PASS":
        raise ValueError("The CPU input validation did not pass")
    for model in report["models"].values():
        root = Path(model["path"])
        root.relative_to("/mnt/codesign-exp/ycfeng")
        current = {
            str(path)
            for path in root.iterdir()
            if path.is_file() and path.suffix in {".json", ".jinja", ".safetensors"}
        }
        validated = {name for name in report["files"] if Path(name).parent == root}
        if current != validated:
            raise ValueError(f"Model structure changed since CPU validation: {root}")
    for name, recorded in report["files"].items():
        path = Path(name)
        path.relative_to("/mnt/codesign-exp/ycfeng")
        if file_identity(path) != recorded["identity"]:
            raise ValueError(f"Input changed since CPU validation: {path}")
    print("MOPD_PREPARED_INPUTS_CHECK_PASS", len(report["files"]), flush=True)


def validate(args: argparse.Namespace) -> None:
    """Scan all source tensors and reproduce the complete prepared datasets on CPU."""
    register_omegaconf_resolvers()
    recipe = load_config(args.config)
    models = verify_models(recipe)
    previous = json.loads(args.preparation.read_text())
    files = {}
    elements = 0
    tensors = 0
    for role, model in models.items():
        expected = previous["models"][role]
        if (
            model["config"] != expected["config"]
            or model["weight_files"] != expected["weight_files"]
        ):
            raise ValueError(f"Model differs from approved preparation: {role}")
        root = Path(model["path"])
        root.relative_to("/mnt/codesign-exp/ycfeng")
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix not in {
                ".json",
                ".jinja",
                ".safetensors",
            }:
                continue
            before = file_identity(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if path.suffix == ".safetensors":
                with safe_open(path, framework="pt", device="cpu") as weights:
                    for key in weights.keys():
                        value = weights.get_slice(key)
                        shape = value.get_shape()
                        if not shape:
                            chunks = [weights.get_tensor(key)]
                            for chunk in chunks:
                                if not torch.isfinite(chunk).all().item():
                                    raise ValueError(
                                        f"Nonfinite model tensor: {path}:{key}"
                                    )
                                elements += chunk.numel()
                        else:
                            for start in range(0, shape[0], args.tensor_rows):
                                chunk = value[start : start + args.tensor_rows]
                                if not torch.isfinite(chunk).all().item():
                                    raise ValueError(
                                        f"Nonfinite model tensor: {path}:{key}:{start}"
                                    )
                                elements += chunk.numel()
                        tensors += 1
            if file_identity(path) != before:
                raise ValueError(f"Input changed during validation: {path}")
            files[str(path)] = {"identity": before, "sha256": digest.hexdigest()}
        print("MOPD_CPU_MODEL_VALIDATED", role, flush=True)

    datasets, source_report = prepare_conversations(
        source_dir=args.source_dir,
        manifest_path=args.data_manifest,
        groups=list(recipe["on_policy_distillation"]["teacher_model_by_agent_name"]),
        seed=previous["seed"],
    )
    for split, expected_rows in datasets.items():
        path = args.preparation.parent / f"{split}.jsonl"
        before = file_identity(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            count = 0
            for count, (line, expected_row) in enumerate(
                zip(stream, expected_rows, strict=True), 1
            ):
                digest.update(line)
                if json.loads(line) != expected_row:
                    raise ValueError(f"Prepared dataset differs: {path}:{count}")
        if file_identity(path) != before:
            raise ValueError(f"Dataset changed during validation: {path}")
        files[str(path)] = {"identity": before, "sha256": digest.hexdigest()}
    report = {
        "status": "PASS",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "validation": "Full CPU safetensor integrity/finite scan and exact dataset reproduction; no native publisher READY evidence was reused.",
        "models": models,
        "tensor_count": tensors,
        "finite_elements": elements,
        "datasets": source_report,
        "files": files,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    check(args.report)


def main() -> None:
    """Run full CPU preparation or a nonblocking worker identity check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--tensor-rows", type=int)
    args = parser.parse_args()
    if args.check:
        try:
            check(args.report)
        except (OSError, ValueError, KeyError) as error:
            print("MOPD_PREPARED_INPUTS_UNVERIFIED", str(error), flush=True)
        return
    if any(
        value is None
        for value in (
            args.config,
            args.preparation,
            args.source_dir,
            args.data_manifest,
            args.tensor_rows,
        )
    ):
        parser.error(
            "Full validation requires config, preparation, source-dir, data-manifest and tensor-rows"
        )
    if args.tensor_rows < 1:
        parser.error("tensor-rows must be positive")
    validate(args)


if __name__ == "__main__":
    main()
