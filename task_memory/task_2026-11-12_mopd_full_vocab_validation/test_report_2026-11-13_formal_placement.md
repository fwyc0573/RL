## Modification History

| Date | Change |
| --- | --- |
| 2026-11-13 | Record rejected smaller GPU layouts, current input readiness, and the retained formal queue state. |

# Formal placement and current input readiness

The current formal run remains unverified on GPU. This report settles the proposed resource replacement and the completed input identity check, not training acceptance.

## Configuration and environment

- Retained run: `exp-1113-151718-896703`, UID `9f0eb021-6bda-44fa-af2f-5469dd08b02b`, personal creator `i-fengyicheng`, H800/codesign, 7 replicas × 8 GPUs, 32 CPU / 768 GB per replica.
- Source: `100.96.128.195:/data/ycfeng:/data/ycfeng` on the current local machine. All cloud accesses remain under `/mnt/codesign-exp/ycfeng`.
- Native CPU config interpreter: `/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python`, Python3.13.14. Personal platform interpreter: `/data/ycfeng/tmp/stepmind-env/bin/python`, Python3.10.6.
- Native full-vocabulary training dimensions remain 151936 vocabulary entries, 256 fresh trajectories/update, 65536-token ceiling, and 200 actual successful optimizer updates. Student remains TP2/PP4/CP1, FP16 with FP32 gradient reduction.

## Checks and observed results

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Current input identity | Six unchanged model roles, current metadata, deliberately changed identity rejected, no repeated full tensor scan | `exp-1113-155805-262829`:41files; student/math/code/stem/logic/if; negative controlPASS; business report08:02:00Z | PASS |
| Input worker terminal state | Expected creator/source/image; native Succeeded; container exit0 and no restart | Succeeded08:04:53Z; exit0/restart0; imageIDd29740f438d49f543989925ccee2e1e0c6a739f2254c9acc7430363a570a6177; launcher85510exit0 | PASS |
| 8×4GPU native config | Valid 8 student GPUs, 4 generation GPUs and20teacherGPUs | NativeMasterConfig/dtype/training validatorsPASS | PASS for CPU config only |
| 8×4GPU admission and supported placement | Distinct host addresses and platform-supported distributedGPU geometry | Requestedhard affinity becomespreferred weight10; platformmanualrevision3668requires8GPUperreplica for multi-node NVIDIA/RDMA | REJECTED; no GPU submission |
| Admission dry-run retention | GET404, no retained GPU job | `exp-1113-160428-882591` GET404 | PASS |
| 4×8GPU native allocation negative control | Reject before Ray/GPU initialization when five teacher nodes exceed four total nodes | ExactValueError: `cluster.num_nodes must leave at least one node for the student after reserving 5 non-colocated teacher node(s); got cluster.num_nodes=4.` | PASS; candidateREJECTED |
| Original formal run | Actual first backward, healthy metrics, native checkpoint/resume,200successes | StillPending08:10:51Z/rank2/remaining28;0formalupdates | UNVERIFIED |

## Exact commands and evidence

Current platform and terminal read, after the runbook proxy setup:

```bash
/data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/inspect_formal_queue_and_input_terminal.py
```

The first local read used an incorrect top-level `brainpp` import and exited before any API request. The corrected read uses inspected `brainpp.rjob.RJobClient` and the documented `get_rjob_infos()` dictionary return. Selected evidence: `evidence/formal-queue-and-input-terminal-20261113.json`.

Original32GPU CPU validator: `/data/ycfeng/tmp/nemo-mopd-20261112/preflight_formal_fp16_pp4_32g.py`; log `formal-32g-config-preflight.log`; exact arguments and resolved config are `formal-200updates-fp16-pp4-32g-submit-args.json` and `formal-200updates-fp16-pp4-32g-resolved-config.json` in the same temporary directory. No trainingGPUjob used that layout.

Native4×8GPU negative control:

```bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp PYTHONPATH=/data/ycfeng/sota-engine/RL
export CUDA_HOME=/data/ycfeng/toolchain/cuda-13.0.2
export LD_LIBRARY_PATH=/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -v 31457280
timeout 60s /data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/preflight_formal_4n8g_rejection.py
```

`systemd-run --scope --quiet -p MemoryMax=30G` was first attempted and rejected with `Interactive authentication required`; no workload ran under it. The bounded30GiB address-space  invocation above passed. Raw errors and success are in `formal-4n8g-rejection.log` and `formal-4n8g-rejection-2.log`; selected receipt: `evidence/formal-4n8g-rejection.json`.

Read-only Lark commands, using the lark-cli skill:

```bash
/data/ycfeng/stepcode-codex-home/bin/lark-cli docs +fetch --as user --doc 'https://wvixbzgc0u7.feishu.cn/wiki/Bvl5wyWr2itF15kGmJIcyr0Uneb' --scope keyword --keyword 'anti-affinity|反亲和|同一节点|多机|整机' --context-before 1 --context-after 1
/data/ycfeng/stepcode-codex-home/bin/lark-cli docs +fetch --as user --doc 'https://wvixbzgc0u7.feishu.cn/wiki/T0SMwwCxNiqxToke2qacYU9Sn3g' --scope keyword --keyword 'hard|soft|反亲和|anti-affinity|GPU'
```

Responses `ok=true`. Trainingmanual documentMCC8dAui9o6M9rxoZQGcnp3ynCf/revision3668 explicitly requires8GPUperreplica for multi-node NVIDIA/RDMA. Affinitymanual documentZ5Y0daKLgovgxZxMqKYcFY9rnsT/revision261 documents strict versus best-effort affinity but does not explain the observed servermutation. Retained selected request/admission/source evidence: `evidence/formal-32g-admission-rejection.json`. No Lark message or write was performed.

## Decision and limits

Keep the existing56GPU nonpreemptible formal job queued with its launcher alive. The per-replicaGPU geometry of the smaller proposal is outside the current platform instructions; changing only affinity does not establish supported startup. The native4×8GPU alternative also fails its necessary allocation condition. The two unverified affinity edits were removed from `submit_on_rjob.py`; `git diff --quiet -- tests/manual/mopd/submit_on_rjob.py` passes.

This does not claim that partial-node jobs can never execute or explain every platform admission mutation. It records why neither proposed replacement is ready for this task. D diagnostic and completed cleanup remain accepted. FormalCUDAbackward, loss/grad norm,200successfulupdates, final finite/state/refit and training/scorer/vLLM parity remain pending.
