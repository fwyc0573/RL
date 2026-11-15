## Modification History

| Date | Change |
| --- | --- |
| 2026-11-15 | Verify renewed current-task OSS cleanup, stop the unsupported128GPU request and submit the former72GPU layout. |

# OSS cleanup and72-H800 FIFO continuation

The renewed current-task cleanup completed. The128-H800 request was stopped before GPU allocation. One replacement72-H800 formal job is queued; durable optimizer progress remains22/200. GPU restore, graph capture, FIFO throughput and new training metrics are unverified.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Stop128GPU request | Exact personal job Stopped; no replicas | exp-1115-143203-969279 Stopped at19:27:44HKT; stopped-by i-fengyicheng; zero replicas | PASS |
| Obsolete file identities and ownership | Unchanged manifest; exact associated jobs terminal/personal | 389 file sizes/mtimes matched;48 associated jobs terminal and owned by i-fengyicheng | PASS |
| OSS namespace cleanup | All selected files absent | 389 files deleted;15,329,381,876 logical bytes (14.276599GiB) | PASS |
| Preserve compact trajectory evidence | Keep sample identity, version, masks, lengths and numerical summaries before deletion |119 summaries;12,923,523 bytes in evidence/cleanup-trajectory-summaries-20261115/ | PASS |
| Protected assets | Prepared model/data/current conversion/recovery identities unchanged |41 model files unchanged; prepared dataset, six current conversion caches and sole step40 checkpoint unchanged | PASS |
| Checkpoint constraint | Only latest; no copies/exports | step40/Adam22 remains the sole complete checkpoint; new checkpoints0; no HF export | PASS |
| Cleanup terminal | Correct source/creator; native completion | exp-1115-193120-091174 / replica Succeeded; container Completed/exit0/restart0; creator i-fengyicheng; local100.96.128.195:/data/ycfeng:/data/ycfeng;0GPU | PASS |
|72GPU native preflight | Original model/workload, FIFO/recovery/scheduler/resource checks pass | TP2/PP4 student8, rollout48, teacher10, spare6; batch256;65536; vocab151936; age5; inflight256; buffer1536; native step40 restore metadata and10240→10496 scheduler counts pass | PASS (CPU) |
|72GPU submission | Single personal local RJob; nine eight-GPU replicas; RDMA mounts | exp-1115-193456-282702 created; exact payload requests9×8 H800/codesign, explicit company image/local NFS/cloud mount and both existing RDMA resources | PASS (submission) |
| Actual GPU execution | Verify workers/restore/graph/FIFO/parity/loss/grad/update count | Pending at19:35HKT, quota41 and zero replicas | UNVERIFIED |
| Physical OSS reclamation | Measure backend physical usage or garbage collection | Namespace deletion measured; shared trash/backend reclamation not inspected or modified | UNVERIFIED |

The remaining current-task cloud namespace contains50,501,557,995 logical bytes (approximately47.03GiB), mostly required native recovery and conversion inputs. Raw historical trajectories and completed diagnostic tensors were intentionally deleted; their archived numerical observations remain, but those tensors cannot be replayed directly. No other task prefix or shared trash was touched.

Commands and environment:

```bash
bash /data/ycfeng/steptron/playground/users/ycfeng/tools/rjob_stop.sh --creator i-fengyicheng --reason 'Correct unmeasured 128-GPU rollout expansion; restore prior 72-GPU MOPD layout as requested by user' exp-1115-143203-969279
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh /data/ycfeng/tmp/nemo-mopd-20261112/check_cleanup_jobs_20261115.py
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh /data/ycfeng/tmp/nemo-mopd-20261112/submit_cleanup_cloud_20261115.py
bash /data/ycfeng/tmp/nemo-mopd-20261112/run_step40_72gpu_native_preflight_20261115.sh
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh /data/ycfeng/tmp/nemo-mopd-20261112/submit_step40_fifo_72gpu_20261115.py --dry-run
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh /data/ycfeng/tmp/nemo-mopd-20261112/submit_step40_fifo_72gpu_20261115.py
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 PYTHONPATH=/data/ycfeng/sota-engine/RL UV_CACHE_DIR=/data/ycfeng/tmp/nemo-mopd-20261112/uv-cache /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/check_fifo_observer_20261115.py
```

Personal submission uses the established StepMind Python3.10.6/brainpp2.8.6 environment. Native preflight uses offlineUV0.13.7, installed Megatron actor environment, Python3.13.14 and Torch2.13.0+cu130; the script sets a30GiB process address-space limit and120-second timeout. CPU native preflight covers config, model/teacher identity, FP16 dtype/FP32 reduction, native replay metadata, scheduler restore and latest-only settings; it does not load GPU tensors. Cleanup uses the prepared basePython3.13 environment withNumPy in a0GPU/4CPU/16GiB codesign worker.

Exact72GPU arguments, resolved config and dry-run payload are in evidence/formal-step40-fifo-72gpu-*.json. Cleanup ownership, all deleted paths, per-file receipts, retained summaries and native terminal status are in evidence/cleanup_2026-11-15_*.json. Temporary launcher/raw-log paths are under /data/ycfeng/tmp/nemo-mopd-20261112/.

A separate0GPU mounted observer, exp-1115-193548-376103, follows the new formal job. It reuses the prior collector, records same-weight comparisons only when weight_age=0 and separately reports admitted older samples. The CPU check verifies age0 andage2 classification with expectedMAE0.15, and records an age6 violation. This is a preparation check, not production parity. After publishing and reading back each compact summary, the observer deletes its completed raw trajectory dump; failures remain soft evidence failures and do not control training. It writes into evidence/formal-step40-fifo-72gpu-live/. Source: /data/ycfeng/tmp/nemo-mopd-20261112/collect_step40_fifo_72gpu_20261115.py.

The formal local launcher PID1579154 and observer launcher PID1591600 are detached on this host and must stay alive through queueing/workload. They are not remote launchers. The prior teacher utilization clean-policy event remains a measured operational risk. Native200-success accounting, final tensor finiteness, final three-way parity, curves and English completion archive remain open.

Latest personal API observation at19:38:42HKT: formal72GPU job Pending, quota42, rank annotation2, no replicas. Observer0GPU job Running at19:38:43HKT. The annotation is the queue service's reported rank, not a promised start time. Selected status and local launcher receipts are in evidence/formal-step40-fifo-72gpu-current-status-20261115.json.

Git delivery PASS: signed commit `663b9143f868364ab4d0104a1a5cc68718555803` was pushed to origin/mopd-sys and independently matched by `git ls-remote --heads origin refs/heads/mopd-sys`. The delivery contains135 scoped current-task records and compact evidence files; the prior FIFO implementation is already present in remote ancestry.
