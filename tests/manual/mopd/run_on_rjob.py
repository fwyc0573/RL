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
"""Join RJob replicas into Ray and run the native MOPD Single-Controller."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import ray
import torch


def write_json(path: Path, value: dict) -> None:
    """Publish one small operational receipt atomically."""
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_evidence(path: Path, value: dict) -> None:
    """Record auxiliary evidence without blocking native training."""
    try:
        write_json(path, value)
    except OSError as error:
        print("MOPD_EVIDENCE_UNVERIFIED", str(path), type(error).__name__, flush=True)


def wait_json(path: Path, *, timeout: int) -> dict:
    """Wait for the head's native startup or completion receipt."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(2)
    raise TimeoutError(f"RJob head did not publish {path} within {timeout}s")


def retain_ray_logs(local: Path, destination: Path) -> None:
    """Retain bounded Ray logs without changing the native workload outcome."""
    try:
        destination.mkdir(parents=True, exist_ok=True)
        secrets = [
            value
            for key, value in os.environ.items()
            if value
            and any(
                word in key.upper()
                for word in ("TOKEN", "SECRET", "ACCESS_KEY", "PASSWORD")
            )
        ]
        logs = local / "ray" / "session_latest" / "logs"
        for pattern in ("raylet.out", "raylet.err", "gcs_server.out", "worker-*.err"):
            for source in logs.glob(pattern):
                with source.open("rb") as stream:
                    stream.seek(max(0, source.stat().st_size - 1024 * 1024))
                    text = stream.read().decode(errors="replace")
                for secret in secrets:
                    text = text.replace(secret, "<REDACTED>")
                (destination / source.name).write_text(text)
    except OSError as error:
        print("MOPD_RAY_LOG_ARCHIVE_UNVERIFIED", type(error).__name__, flush=True)


def wait_cluster(head: dict, *, nodes: int, gpus_per_node: int, timeout: int) -> None:
    """Wait for the resources that the native training configuration requires."""
    ray.init(address=f"{head['ip']}:{head['port']}", log_to_driver=False)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            active = [node for node in ray.nodes() if node["Alive"]]
            if len(active) == nodes and all(
                node["Resources"].get("GPU", 0) == gpus_per_node for node in active
            ):
                if len({node["NodeManagerAddress"] for node in active}) != nodes:
                    raise ValueError("RJob replicas did not receive distinct GPU nodes")
                print(
                    "MOPD_RAY_READY",
                    json.dumps({"nodes": nodes, "gpus": nodes * gpus_per_node}),
                    flush=True,
                )
                return
            time.sleep(2)
    finally:
        ray.shutdown()
    raise TimeoutError("Ray did not register all requested RJob GPU resources")


def terminate(signum: int, frame: object) -> None:
    """Let normal finalization retain evidence for a platform SIGTERM."""
    raise SystemExit(128 + signum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-entrypoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--cpus-per-node", type=int, required=True)
    parser.add_argument("--startup-timeout", type=int, required=True)
    parser.add_argument("--min-worker-port", type=int, required=True)
    parser.add_argument("--max-worker-port", type=int, required=True)
    parser.add_argument("--training-timeout", type=int, required=True)
    args, overrides = parser.parse_known_args()
    args.output_root.relative_to("/mnt/codesign-exp/ycfeng")
    rank = int(os.environ["KUBEBRAIN_REPLICA"])
    if int(os.environ["KUBEBRAIN_REPLICA_TOTAL"]) != args.nodes:
        raise ValueError(
            "RJob replica count differs from the configured Ray node count"
        )
    if torch.cuda.device_count() != args.gpus_per_node:
        raise ValueError(
            "RJob GPU allocation differs from the configured GPUs per node"
        )
    if any(
        "H800" not in torch.cuda.get_device_name(i) for i in range(args.gpus_per_node)
    ):
        raise ValueError("This MOPD validation requires H800 workers")
    device_capabilities = sorted(
        {torch.cuda.get_device_capability(i) for i in range(args.gpus_per_node)}
    )
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST",
        " ".join(f"{major}.{minor}" for major, minor in device_capabilities),
    )
    job = os.environ["KUBEBRAIN_RJOB_NAME"]
    run = args.output_root / job
    run.mkdir(parents=True, exist_ok=True)
    local = Path("/tmp") / f"nemo-mopd-{job}"
    local.mkdir(parents=True, exist_ok=True)
    for key, name in {
        "TMPDIR": "tmp",
        "HF_HOME": "huggingface",
        "HF_MODULES_CACHE": "hf-modules",
        "XDG_CACHE_HOME": "cache",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
        "CUDA_CACHE_PATH": "cuda",
        "RAY_TMPDIR": "ray",
    }.items():
        cache = local / name
        cache.mkdir(exist_ok=True)
        os.environ[key] = str(cache)
    for key in list(os.environ):
        if key.startswith("OTEL_"):
            os.environ.pop(key)
    os.environ.update(
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        RAY_USAGE_STATS_ENABLED="0",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
        NEMO_RL_VENV_DIR="/data/ycfeng/envs/nemo-mopd-actors",
        NEMO_GYM_VENV_DIR="/data/ycfeng/envs/nemo-mopd-gym-servers",
        NRL_MEGATRON_CHECKPOINT_DIR=str(
            args.output_root / "initial-megatron-checkpoints"
        ),
    )
    ip = socket.gethostbyname(socket.gethostname())
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "ray.scripts.scripts",
        "start",
        "--node-ip-address",
        ip,
        "--num-gpus",
        str(args.gpus_per_node),
        "--num-cpus",
        str(args.cpus_per_node),
        "--disable-usage-stats",
        "--min-worker-port",
        str(args.min_worker_port),
        "--max-worker-port",
        str(args.max_worker_port),
    ]
    outcome = {"status": "failed", "rank": rank, "job": job}
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        write_evidence(
            run / f"runtime-{rank}.json",
            {
                "python": sys.version,
                "executable": sys.executable,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_device_capabilities": device_capabilities,
                "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
                "devices": [
                    torch.cuda.get_device_name(i) for i in range(args.gpus_per_node)
                ],
                "hostname": socket.gethostname(),
                "ip": ip,
                "platform_nccl": {
                    k: v for k, v in os.environ.items() if k.startswith("NCCL_")
                },
            },
        )
        if rank == 0:
            head = {
                "ip": ip,
                "port": int(os.environ["PORT_AUTO0"]),
                "dashboard_port": int(os.environ["PORT_AUTO1"]),
            }
            command += [
                "--head",
                "--port",
                str(head["port"]),
                "--dashboard-host",
                "0.0.0.0",
                "--dashboard-port",
                str(head["dashboard_port"]),
                "--temp-dir",
                str(local / "ray"),
            ]
            subprocess.run(command, check=True, timeout=args.startup_timeout)
            write_json(run / "head.json", head)
            wait_cluster(
                head,
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
                timeout=args.startup_timeout,
            )
            os.environ["RAY_ADDRESS"] = f"{head['ip']}:{head['port']}"
            os.environ["NRL_RAY_READY_EPOCH"] = str(time.time())
            training_command = [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-u",
                str(args.training_entrypoint)
                if args.training_entrypoint is not None
                else "examples/run_grpo_single_controller.py",
                "--config",
                str(args.config),
                f"cluster.num_nodes={args.nodes}",
                f"cluster.gpus_per_node={args.gpus_per_node}",
                f"logger.log_dir={run / 'logs'}",
                f"checkpointing.checkpoint_dir={run / 'checkpoints'}",
                "+env.nemo_gym.skip_venv_if_present=true",
                *overrides,
            ]
            write_evidence(run / "command.json", {"argv": training_command})
            training_environment = dict(os.environ)
            # Ray workers inherit these platform settings from their own raylet.
            # Broadcasting the head's device names breaks heterogeneous nodes.
            for key in ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA"):
                training_environment.pop(key, None)
            subprocess.run(
                training_command,
                check=True,
                timeout=args.training_timeout,
                env=training_environment,
            )
            outcome["status"] = "native_execution_succeeded"
            print("MOPD_NATIVE_EXECUTION_FINISHED", str(run), flush=True)
        else:
            head = wait_json(run / "head.json", timeout=args.startup_timeout)
            command += ["--address", f"{head['ip']}:{head['port']}"]
            subprocess.run(command, check=True, timeout=args.startup_timeout)
            finished = wait_json(
                run / "finished.json",
                timeout=args.startup_timeout + args.training_timeout,
            )
            if finished["status"] != "native_execution_succeeded":
                raise RuntimeError("The native MOPD head failed; inspect native logs")
            outcome["status"] = "native_execution_succeeded"
    finally:
        if rank == 0:
            write_evidence(run / "finished.json", outcome)
        retain_ray_logs(local, run / "ray" / f"rank-{rank}")
        write_evidence(run / f"replica-{rank}.json", outcome)
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
