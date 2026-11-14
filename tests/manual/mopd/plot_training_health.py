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
"""Plot observed MOPD updates from native scalar events and sample summaries."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Write a figure and its exact scalar values for completed training attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--parity-directory", type=Path, nargs="+", required=True)
    parser.add_argument("--through-step", type=int, required=True)
    parser.add_argument("--mae-limit", type=float, required=True)
    parser.add_argument("--grad-clip", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events: dict[str, dict[int, tuple[float, float]]] = {}
    for path in args.metrics:
        for tag, observations in json.loads(path.read_text()).items():
            series = events.setdefault(tag, {})
            for observation in observations:
                step = observation["step"]
                event = (observation["wall_time"], observation["value"])
                if step not in series or event[0] > series[step][0]:
                    series[step] = event
    metrics = {
        tag: {step: event[1] for step, event in series.items()}
        for tag, series in events.items()
    }
    steps = sorted(
        step
        for step in metrics["train/optimizer_update_successful"]
        if step <= args.through_step
    )
    assert steps and steps[-1] == args.through_step
    assert steps == list(range(steps[0], args.through_step + 1))
    summaries = {}
    for directory in args.parity_directory:
        for path in sorted(directory.glob("*.json")):
            summary = json.loads(path.read_text())
            if summary["step"] in steps:
                assert summary["step"] not in summaries, path
                summaries[summary["step"]] = summary
    assert set(summaries) == set(steps)
    groups = sorted(summaries[steps[0]]["by_group"])

    tags = {
        "optimizer_success": "train/optimizer_update_successful",
        "loss": "train/loss",
        "full_rkl": "train/opd_full_reverse_kl",
        "loss_scale": "train/loss_scale",
        "valid_samples": "train/num_valid_samples",
        "response_tokens": "train/global_valid_toks",
        "step_seconds": "timing/train/total_step_time",
        "generation_seconds": "timing/train/exposed_generation",
        "policy_seconds": "timing/train/policy_training",
    }
    rows: list[dict[str, int | float | str | None]] = []
    successes = 0
    for step in steps:
        row: dict[str, int | float | str | None] = {"step": step}
        row.update({name: metrics[tag][step] for name, tag in tags.items()})
        assert all(math.isfinite(metrics[tag][step]) for tag in tags.values())
        flag = metrics["train/optimizer_update_successful"][step]
        assert flag in (0, 1)
        successes += int(flag)
        row["successful_updates"] = successes
        # A skipped native optimizer step emits no norm; preserve that gap.
        row["grad_norm"] = metrics["train/grad_norm"][step] if flag else None
        if flag:
            assert math.isfinite(metrics["train/grad_norm"][step])
        summary = summaries[step]
        assert sorted(summary["by_group"]) == groups
        assert summary["overall"]["tokens"] == row["response_tokens"]
        assert len(summary["samples"]) == row["valid_samples"]
        row["source_job"] = summary["job"]
        row["scorer_vllm_mae"] = summary["overall"]["mae"]
        row["scorer_vllm_max"] = summary["overall"]["max"]
        for group in groups:
            samples = [s for s in summary["samples"] if s["group"] == group]
            row[f"{group}_mae"] = summary["by_group"][group]["mae"]
            row[f"{group}_response_tokens_mean"] = sum(
                s["tokens"] for s in samples
            ) / len(samples)
        rows.append(row)

    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    series_axes = (
        (axes[0, 0], (("loss", "Full-vocabulary RKL loss"),), "Loss (nats/token)"),
        (axes[0, 1], (("grad_norm", "Pre-clipping gradient norm"),), "Gradient norm"),
        (axes[1, 0], (("successful_updates", "Successful updates"),), "Count"),
        (
            axes[1, 1],
            tuple((f"{group}_mae", group) for group in groups),
            "Scorer/vLLM MAE (nats/token)",
        ),
        (
            axes[2, 0],
            tuple((f"{group}_response_tokens_mean", group) for group in groups),
            "Mean response tokens per sample",
        ),
        (
            axes[2, 1],
            (
                ("step_seconds", "Total step"),
                ("generation_seconds", "Exposed generation"),
                ("policy_seconds", "Policy training"),
            ),
            "Seconds",
        ),
    )
    skipped = [row["step"] for row in rows if row["optimizer_success"] == 0]
    for axis, columns, ylabel in series_axes:
        for column, label in columns:
            axis.plot(steps, [row[column] for row in rows], marker=".", label=label)
        for step in skipped:
            axis.axvline(step, color="red", alpha=0.25, linestyle=":")
        axis.set_xlabel("Controller step (red dotted line = skipped update)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0, 1].axhline(args.grad_clip, color="black", linestyle="--", linewidth=1)
    axes[1, 1].axhline(args.mae_limit, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_yscale("log")
    figure.suptitle(
        f"MOPD observed training: {successes} successes / {len(steps)} attempts\n"
        "Native metrics and independent scorer/vLLM comparison"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140)
    plt.close(figure)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "figure": str(args.output),
                "csv": str(csv_path),
                "first_step": steps[0],
                "last_step": steps[-1],
                "successful_updates": successes,
                "attempts": len(steps),
                "skipped_steps": skipped,
            }
        )
    )


if __name__ == "__main__":
    main()
