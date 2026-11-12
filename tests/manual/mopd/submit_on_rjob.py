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
"""Submit the native MOPD Ray entrypoint through local personal RJobBackend."""

import argparse
import json
import os
import shlex
from pathlib import Path

from steptron.exp.base_exp import ResourceConfig
from steptron.utils.stepmind import spawn_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-entrypoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--cpus-per-node", type=int, required=True)
    parser.add_argument("--memory-gb-per-node", type=int, required=True)
    parser.add_argument("--startup-timeout", type=int, required=True)
    parser.add_argument("--min-worker-port", type=int, required=True)
    parser.add_argument("--max-worker-port", type=int, required=True)
    parser.add_argument("--training-timeout", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--credential-dir", type=Path, required=True)
    parser.add_argument("--base-environment", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--cuda-toolkit", type=Path, required=True)
    parser.add_argument("--cuda-compat", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    args, overrides = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[3]
    args.output_root.relative_to("/mnt/codesign-exp/ycfeng")
    for path in (
        args.config,
        args.base_environment / "bin/python",
        args.uv,
        args.cuda_toolkit / "bin/nvcc",
        args.cuda_compat / "libcuda.so.1",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    credentials = {
        "BRAINPP_ACCESS_KEY": (args.credential_dir / "accesskey_id")
        .read_text()
        .strip(),
        "BRAINPP_SECRET_KEY": (args.credential_dir / "accesskey_secret")
        .read_text()
        .strip(),
    }
    if not all(credentials.values()):
        raise ValueError("Personal RJob credentials must be nonempty")
    os.environ.update(credentials, STEPMIND_BACKEND="rjob", EXP_ID=args.experiment_id)
    argv = [
        str(args.uv),
        "run",
        "--no-sync",
        "python",
        "-u",
        str(repo / "tests/manual/mopd/run_on_rjob.py"),
        "--config",
        str(args.config.resolve()),
        "--output-root",
        str(args.output_root),
        "--nodes",
        str(args.nodes),
        "--gpus-per-node",
        str(args.gpus_per_node),
        "--cpus-per-node",
        str(args.cpus_per_node),
        "--startup-timeout",
        str(args.startup_timeout),
        "--min-worker-port",
        str(args.min_worker_port),
        "--max-worker-port",
        str(args.max_worker_port),
        "--training-timeout",
        str(args.training_timeout),
        *overrides,
    ]
    if args.training_entrypoint is not None:
        if not args.training_entrypoint.is_file():
            raise FileNotFoundError(args.training_entrypoint)
        argv += ["--training-entrypoint", str(args.training_entrypoint.resolve())]
    shell = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo))}",
            f"export PYTHONPATH={shlex.quote(str(repo))}",
            "export PATH="
            + shlex.quote(
                f"{args.uv.parent}:{args.base_environment / 'bin'}:{args.cuda_toolkit / 'bin'}"
            )
            + ':"$PATH"',
            "export LD_LIBRARY_PATH="
            + shlex.quote(str(args.cuda_compat))
            + "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}",
            "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv",
            "exec " + shlex.join(argv),
        ]
    )
    command = shlex.join(["bash", "-lc", shell])
    cfg = ResourceConfig(
        cpu=args.cpus_per_node,
        gpu=args.gpus_per_node,
        mem_gb=args.memory_gb_per_node,
        replica=args.nodes,
        image=args.image,
        positive_tags=["H800"],
        extra_requirements=[],
        mounts=["juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp"],
        envs={
            "DISTRIBUTED_JOB": "true",
            "MC_DISABLE_METACACHE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_PROJECT_ENVIRONMENT": str(args.base_environment),
            "CUDA_HOME": str(args.cuda_toolkit),
            "CUDA_PATH": str(args.cuda_toolkit),
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            **credentials,
        },
        command="{COMMAND}",
        preemptible=args.preemptible,
    )
    os.chdir("/data/ycfeng")
    print(
        "MOPD_RJOB_REQUEST",
        json.dumps(
            {
                "nodes": cfg.replica,
                "gpus_per_node": cfg.gpu,
                "cpus_per_node": cfg.cpu,
                "memory_gb_per_node": cfg.mem_gb,
                "charged_group": "codesign",
                "preemptible": cfg.preemptible,
                "image": cfg.image,
                "source": os.getcwd(),
                "mounts": cfg.mounts,
                "custom_resources": cfg.custom_resources,
                "command": command,
                "environment_keys": sorted(cfg.envs),
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return
    workers = spawn_tasks(
        cfg,
        command=command,
        charged_group="codesign",
        use_image=True,
        code_mount_point="/data/ycfeng",
    )
    print("MOPD_RJOB_TERMINAL", workers.rjob.meta.name, workers.poll(), flush=True)


if __name__ == "__main__":
    main()
