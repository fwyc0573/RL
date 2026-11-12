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
"""Verify the source MOPD assets and prepare fresh-rollout Gym task inputs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

import yaml


def verify_models(recipe: dict) -> dict:
    """Compare complete tokenizer mappings and record the six model configs."""
    paths = {"student": recipe["policy"]["model_name"]}
    paths.update(recipe["on_policy_distillation"]["teacher_model_by_agent_name"])
    reference = None
    report = {}
    for role, model_path in paths.items():
        path = Path(model_path)
        config = json.loads((path / "config.json").read_text())
        tokenizer = json.loads((path / "tokenizer.json").read_text())
        tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
        semantic_tokenizer = {
            "vocab": tokenizer["model"]["vocab"],
            "merges": [
                tuple(merge.split(" ")) if isinstance(merge, str) else tuple(merge)
                for merge in tokenizer["model"]["merges"]
            ],
            "added_tokens": tokenizer["added_tokens"],
            "normalizer": tokenizer["normalizer"],
            "pre_tokenizer": tokenizer["pre_tokenizer"],
            "decoder": tokenizer["decoder"],
        }
        if reference is None:
            reference = semantic_tokenizer
        differences = [
            key for key, value in semantic_tokenizer.items() if value != reference[key]
        ]
        if differences:
            raise ValueError(f"{role} tokenizer differs from student: {differences}")
        weights = sorted(path.glob("*.safetensors"))
        if not weights or any(weight.stat().st_size == 0 for weight in weights):
            raise ValueError(f"Missing or empty safetensors weights: {path}")
        template_path = path / "chat_template.jinja"
        report[role] = {
            "path": str(path),
            "config": config,
            "vocabulary_and_bpe_match_student": True,
            "merges": len(semantic_tokenizer["merges"]),
            "post_processor": tokenizer["post_processor"],
            "chat_template": tokenizer_config.get("chat_template"),
            "chat_template_file": (
                template_path.read_text() if template_path.exists() else None
            ),
            "weight_files": {weight.name: weight.stat().st_size for weight in weights},
        }
    return report


def prepare_conversations(
    *, source_dir: Path, manifest_path: Path, groups: list[str], seed: int
) -> tuple[dict[str, list[dict]], dict]:
    """Preserve source identities, reject split overlap, and balance group slots."""
    manifest = json.loads(manifest_path.read_text())
    expected = {row["path"]: row for row in manifest["files"]}
    pools = {split: {} for split in ("train", "probe")}
    identities = {split: set() for split in ("train", "probe", "reserve")}
    prompts = {split: set() for split in ("train", "probe", "reserve")}
    report = {"sources": {}, "seed": seed}
    sources = [(group, split) for group in groups for split in ("train", "probe")]
    sources.append(("math", "reserve"))
    for group, split in sources:
        relative_path = f"{group}/{split}.json"
        path = source_dir / relative_path
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected[relative_path]["sha256"]:
            raise ValueError(f"Source differs from the approved manifest: {path}")
        conversations = json.loads(payload)
        if len(conversations) != expected[relative_path]["items"]:
            raise ValueError(f"Source count differs from manifest: {path}")
        converted = []
        for index, conversation in enumerate(conversations):
            if [message["role"] for message in conversation] != ["user", "assistant"]:
                raise ValueError(f"Unexpected conversation roles: {path}:{index}")
            user = conversation[0]
            metadata = user["meta"]
            if metadata["group"] != group:
                raise ValueError(f"Wrong group in {path}:{index}")
            identity = (
                group,
                metadata["source"],
                metadata.get("subset"),
                str(metadata["source_row"]),
            )
            identities[split].add(identity)
            prompts[split].add(user["content"])
            converted.append(
                {
                    "task_id": {"taskset": group, "task_id": f"{split}/{index}"},
                    "task_input": {
                        "responses_create_params": {
                            "input": [{"role": "user", "content": user["content"]}]
                        },
                        "task_data": {
                            "source_file": relative_path,
                            "source_index": index,
                            "metadata": metadata,
                        },
                    },
                }
            )
        if split in pools:
            pools[split][group] = converted
        report["sources"][relative_path] = {"sha256": digest, "rows": len(converted)}
    for left, right in itertools.combinations(identities, 2):
        shared_ids = identities[left] & identities[right]
        shared_prompts = prompts[left] & prompts[right]
        if shared_ids or shared_prompts:
            raise ValueError(
                f"{left}/{right} overlap: {len(shared_ids)} source IDs, "
                f"{len(shared_prompts)} exact prompts"
            )
    rng = random.Random(seed)
    output = {}
    for split, group_pools in pools.items():
        slots_per_group = max(map(len, group_pools.values()))
        balanced = {}
        for group, pool in group_pools.items():
            shuffled = pool.copy()
            rng.shuffle(shuffled)
            balanced[group] = list(
                itertools.islice(itertools.cycle(shuffled), slots_per_group)
            )
        # Each group has equal probability under the native shuffled dataloader.
        output[split] = list(itertools.chain.from_iterable(zip(*balanced.values())))
        report[split] = {
            "rows": len(output[split]),
            "slots_per_group": slots_per_group,
            "unique_source_rows": {
                group: len(pool) for group, pool in group_pools.items()
            },
        }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text())
    models = verify_models(recipe)
    groups = list(recipe["on_policy_distillation"]["teacher_model_by_agent_name"])
    datasets, report = prepare_conversations(
        source_dir=args.source_dir,
        manifest_path=args.data_manifest,
        groups=groups,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in datasets.items():
        with (args.output_dir / f"{split}.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["models"] = models
    (args.output_dir / "preparation.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(args.output_dir),
                "train": report["train"],
                "probe": report["probe"],
                "models": list(models),
            }
        )
    )


if __name__ == "__main__":
    main()
