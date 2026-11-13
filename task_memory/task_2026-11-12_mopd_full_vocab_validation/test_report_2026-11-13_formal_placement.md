## Modification History

| Date | Change |
| --- | --- |
| 2026-11-13 | Verify corrected32GPU bundle allocation, reservation order, CPU regressions and current input terminal state. |
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

## Decision at16:11 HKT and limits

Keep the existing56GPU nonpreemptible formal job queued with its launcher alive. The per-replicaGPU geometry of the smaller proposal is outside the current platform instructions; changing only affinity does not establish supported startup. The native4×8GPU alternative also fails its necessary allocation condition. The two unverified affinity edits were removed from `submit_on_rjob.py`; `git diff --quiet -- tests/manual/mopd/submit_on_rjob.py` passes.

This does not claim that partial-node jobs can never execute or explain every platform admission mutation. It records why neither proposed replacement is ready for this task. D diagnostic and completed cleanup remain accepted. FormalCUDAbackward, loss/grad norm,200successfulupdates, final finite/state/refit and training/scorer/vLLM parity remain pending.


## Corrected32GPU layout, verified before replacement

The native allocator counted every teacher logical node as an exclusive physical node even without topology constraints. RayVirtualCluster actually reserves GPU bundles. `_non_colocated_teacher_node_count()` now rounds the summed GPU capacity up to cluster-node capacity when `segment_size` is absent; topology-constrained teachers retain dedicated-node counting. The replacement recipe requests4×8 H800:8 training,8 generation, five2GPU teachers and6 spare GPUs. Student TP2/PP4/CP1, all model identities,151936 vocabulary entries,256 trajectories,65536 ceiling and200-success target remain unchanged.

The first actual Ray reservation exposed GPU fragmentation: teachers reserved first, and a student8-bundle group could span nodes under the native softPACK strategy. Reserving training and generation placement groups before partial-node teachers removed that observed fragmentation. Native setup now performs that reservation order before any model/Gym startup. No public config schema or model-worker interface changed.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Native config/dtype/objective/model identity |200/256/65536; FP16 with FP32 gradient reduction; scheduler horizon200; exact six roles; valid4×8GPU allocation |All native validator markers present in `formal-packed32-native-config-preflight.log` |PASS CPU |
| Actual teacher-first Ray allocation |Each logical student node stays on one simulated physical node |Five teachers reserve but a student/generation8-bundle group spans nodes |FAIL; cause identified |
| Student-first real Ray placement |4 simulated nodes;26 CREATED GPU bundles; single-node student/generation; teachers occupy the other2nodes |Exact native `_build_clusters`, `reserve_teacher_clusters` and Ray placement-group APIs satisfy all assertions;6GPU capacity unused |PASS CPU simulation |
| Reservation-order regression before fix |Catch native teacher-first behavior |Exact expected reserve_train/reserve_inference events absent |FAIL as intended |
| Affected allocation/order/topology regressions after fix |No missed or reordered resource claims |6passed,323deselected in11.80s |PASS CPU |
| Ruff0.14.3 lint/format and Git whitespace |Scoped source/tests clean |All checks passed;2files formatted; diff --check exit0 |PASS |
| Current immutable input identities |41files/six roles/changed-size negative control; no repeated tensor scan |`exp-1113-173227-790743`, businessPASS09:35:00Z, platformSucceeded09:35:48Z, exit0/restart0, exact d297imageID/personalcreator/localNFS |PASS |
| Formal GPU result |Healthy actual backward/loss/grad; native save/resume;200successes; finalparity |Original56GPU run stillPending at09:36:34Z/rank2/remaining42; no formal checkpoint |UNVERIFIED |

Commands below run from `/data/ycfeng/sota-engine/RL` with `UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker`, Python3.13.14. CPU probes used `ulimit -v31457280` and `timeout -k10s60s`. Raw logs and scripts remain under `/data/ycfeng/tmp/nemo-mopd-20261112`.

```bash
uv run --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/preflight_formal_fp16_pp4_packed32.py
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/preflight_packed32_ray_reservation.py --student-first
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --no-sync pytest --confcutdir=tests/unit/single_controller tests/unit/single_controller/test_setup.py -q -k 'teacher_node or dedicated_teacher or partial_node_teachers or mopd_reserves_before_models or reserves_topology_constrained'
uv run --no-sync /data/ycfeng/tmp/uv-cache/archive-v0/L1BVEpO2ROvDZA3I/ruff-0.14.3.data/scripts/ruff check nemo_rl/algorithms/single_controller_utils/setup.py tests/unit/single_controller/test_setup.py
uv run --no-sync /data/ycfeng/tmp/uv-cache/archive-v0/L1BVEpO2ROvDZA3I/ruff-0.14.3.data/scripts/ruff format --check nemo_rl/algorithms/single_controller_utils/setup.py tests/unit/single_controller/test_setup.py
git diff --check
/data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/inspect_packed32_queue.py
```

Initial CPU probe failures were environmental: inheritedOTEL_* caused GCS/raylet `GRPC_CALL_ERROR_INVALID_METADATA`; clearing those keys as the formal wrapper already does removed the abort. Explicitly disabling Ray's uv runtime hook before imports removed unwanted repository packaging. A broader config-test attempt failed during the parent Ray fixture; it is not a configPASS. Mocked setup regressions use the existing `--confcutdir` boundary; real-placement evidence comes separately from the four-node RayCluster probe. Shutdown emitted pre-existing `RayVirtualCluster.__del__` interpreter-finalization errors after businessPASS; all explicit cluster shutdowns completed. These CPU results do not establish CUDA memory sufficiency or training parity.

Decision: the corrected4×8GPU request is now CPU/storage ready. Re-read the exact original job before stopping it; replace only while stillPending and verify no duplicate training allocation before local submission. Retain selected receipts in `evidence/formal-packed32-ray-reservation.json`, `evidence/formal-queue-and-packed32-input-terminal-20261113.json` and the exact packed32 argument/configJSONs. Stop extending preparation once GPU submission is possible.


## Replacement submission at2026-11-13T17:47:07+08:00

Signedcommit `d1111060` contains the GPU-capacity/reservation-order repair, scopedregressions and4×8recipe. Requiredstopscript re-read original56GPUjob09:42:53Z asPending, stopped09:42:57Z, and personalAPI verifiedStopped/noReplicas09:43:25Z. Replacement `exp-1113-174329-659007` was created09:43:29Z, UID `da3fd441-bc51-4d5b-aa27-ad216d7fe445`. Personalcreator, exactcurrentNFSsource,4×8H800/32CPU/768GBpernode/companyimage/cloudmount and both nativeRDMArequests are recorded in `evidence/formal-packed32-final-launch-request.json` and `evidence/formal-packed32-submitted-status.json`. Lateststatus at20261113T094702Z isPending/rank2/remaining30. SubmissionPASS is not GPUtrainingPASS.

Exactsubmission command: after existing `personal_env_20261113.sh` initialization, `/data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_formal_200updates_fp16_pp4_packed32.py`. Persistentlauncher session82605. Newindependent0GPUreader `exp-1113-174403-962663`, launcher53011; monitor5931. Their commands/scripts are `submit_formal_packed32_mounted_evidence_reader.py` and `watch_formal_fp16_pp4_packed32.py --interval45 --duration172800 --output /data/ycfeng/tmp/nemo-mopd-20261112/formal-packed32-monitor`. Oldreader stop was explicitly requested with the same requiredscript after replacement; terminal confirmation pending. No cleanup or completedD rerun.

Cloudrunprefix: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-five-teacher-fp16-resume/exp-1113-174329-659007`. CPUreceipt originals/prestop/stopped proof are retained in the currentevidence directory. FormalGPUmemory/backward/loss/grad/save-resume/200successfulupdates/finalparity remainUNVERIFIED whilequeued.
