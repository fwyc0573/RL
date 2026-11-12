## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Preserve native failure, terminal RCA and CPU verification of hardware-derived architecture environment. |

# Native Megatron architecture environment

The native control exp-1112-165702-849504 Failed at2026-11-12T09:12:55Z. The initiating business traceback at09:12:47Z is `Policy.__init__`, lm_policy.py:328: `RuntimeError: TORCH_CUDA_ARCH_LIST is not set`. Head replica UID7fc7f601-7db2-4d70-95f7-2bafde46b61f, RJob UID134dafb7-6caa-4a76-96c1-ac0a4ff6e4ef; creator i-fengyicheng, H800/codesign, nonpreemptible. Container Error/exit1/restart0 at09:12:51Z. StopTimestamp09:12:54Z and any-task fail-policy annotation are subsequent cleanup. Launcher cleanup17:12:56.753HKT followed the failure. No agent manual stop was requested. Missing audited initiating manual-stop actor/reason/time remain UNKNOWN.

## Commands and environment

Native Python3.13.14 `/data/ycfeng/envs/nemo-mopd-py313`, Torch2.13.0+cu130, executed using `/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --no-sync python`. CPU check script: `/data/ycfeng/tmp/nemo-mopd-20261112/run_native_cpu_validation.py`; probe script: `probe_cuda_arch_guard.py` in the same directory. Exact argument vectors are archived in evidence/cuda-arch-cpu-validation.json. The probe executes the actual worker prologue up to cloud I/O with CUDA discovery mocked to8 H800s/capability(9,0), then actual native Policy.__init__; cluster.world_size=0 deliberately stops before worker allocation.

Personal submit dry-run uses Python3.10.6 `/data/ycfeng/tmp/stepmind-env/bin/python`, `tests/manual/mopd/submit_on_rjob.py`, evidence/self-diagnostic-submit-args.json and `--dry-run`. It allocates no job. The exact previous job was re-read with the personal Python client and still Failed. Credentials are never printed or archived.

Ruff0.9.9 commands use `uv tool run --offline --from ruff==0.9.9 ruff check` and `ruff format --check` on both native RJob scripts. Set UV_CACHE_DIR to the task cache and UV_DEFAULT_INDEX=`http://mirrors.i.basemind.com/pypi/simple/` to address the existing cached index.

## Results

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Missing architecture native guard | Exact RuntimeError, exit2 | Exact RuntimeError, exit2 | PASS |
| Worker-derived architecture | Pass architecture guard, reach deliberate zero-cluster error | Value9.0, World size(0) insufficient, exit0 | PASS |
| Preserve explicit architecture | Preserve9.0+PTX | Value9.0+PTX retained, exit0 | PASS |
| Personal submit dry-run | H800/codesign3x8, same image/local source/volume, no submission | Explicit request emitted, exit0 | PASS |
| Ruff/format | Both scripts clean | All checks passed;2already formatted | PASS |
| GPU constructor and actor creation | Native constructor guard passes and8trainer workers created | exp-1112-173036-554443:8workers initialized in68.25s | PASS |
| GPU training/refit | Actual weights/refit/updates verified | Initialization continues;0updates observed | UNVERIFIED |

Two probe implementation errors are retained: omitted OmegaConf resolver registration and a patch.dict context that restored the environment before constructor inspection. Corrected checks execute the actual native guard and preserve only mocked platform input. Ruff's first default-public-index call timed out; default-public offline had no matching cached index. Reusing the existing corporate cache in offline mode passed. These are verification-tool errors, not new training failures; no runtime dependencies were installed.

## Root-cause change and evidence limits

`tests/manual/mopd/run_on_rjob.py` now derives missing TORCH_CUDA_ARCH_LIST from allocated device capabilities before Ray startup, preserves explicit values, and records capability/value in its existing runtime receipts. No native guard, objective, tolerance or dependency was changed. Actual H800 receipt/actor propagation and native trainer initialization require a GPU rerun; CPU mocked discovery does not establish those outcomes.

RCA exact-object, live/historical Event and incident query_range reads succeeded. The09:11–09:14Z log query returned355/limit3000 rows, recovered the full initiating traceback, and was not limit-truncated. Existing DeepScan count0. Both all-operation audits timed out18s; do not record them as empty. Historical ConfigMap warning09:07:18Z preceded successful engine initialization. Selected exact evidence: evidence/self-diagnostic-4-terminal.json. Raw queries/logs stay under `/data/ycfeng/tmp/nemo-mopd-20261112/rca-self-native-4/`.

Four TP2 vLLM engines completed loading/KV cache/CUDA graphs; previous NCCL failure did not recur. All3nodes used ens20np0, so this run cannot prove the earlier heterogeneous-node case. Initial refit, actual151936-vocabulary trainer forward, loss/grad norm, successful updates and all parity checks remain unverified.

## Corrected GPU execution

exp-1112-173036-554443 UIDe7d21741-1753-4bec-8dd8-70d133f6f998 is Running at09:36:06Z, queue approval09:30:39.748817614Z,3x8H800, creator i-fengyicheng, codesign/nonpreemptible, local NFS100.96.128.195:/data/ycfeng:/data/ycfeng. Launcher71916 and monitor32270 are alive. All3actual runtime receipts show device capability[[9,0]], TORCH_CUDA_ARCH_LIST9.0 and the pinned native interpreter/Torch. A read-only Ray dashboard inspection completed SUCCEEDED;12existing generation processes/actors retained9.0. Native trainer creation reached8workers initialized in68.25s, passing the original constructor failure location. Model loading, initial refit and successful updates are still awaiting evidence. Small evidence: native-cuda-architecture-5.json and native-cuda-constructor-5.json.
