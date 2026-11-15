## Modification History

| Date | Change |
| --- | --- |
| 2026-11-15 | Verify actual 72-H800 restore, graph capture, 91 consecutive successful updates, latest-only step130/Adam112, and the measured timing improvement. |

# Live FIFO training validation

The formal continuation is running on the original 72-H800 scale. At the inspected native publication, step130 contains eight optimizer shards whose two parameter groups all have Adam step112. There is one complete checkpoint, step130; older checkpoints are absent. Native scalar events through step131 show91 successful updates after the restored step40, with zero overflow skips. This establishes112 durable successes and113 observed successes;200 durable successes and final acceptance remain unfinished.

## Actual environment and restore

Formal job `exp-1115-193456-282702`, UID `102dc861-ec0d-4fdd-85bc-dc216ff00572`, was approved at21:39:39HKT on November15,2026. Personal Python API at23:49:58HKT confirms Running with nine replicas, creator i-fengyicheng, and local NFS source100.96.128.195:/data/ycfeng:/data/ycfeng. Both local launcher processes remain alive. The separate0GPU observer is Running and its receipt reports no collection failures.

All nine workers report eight NVIDIA H800 GPUs, driver570.124.06, Python3.13.14, Torch2.13.0+cu130, and CUDA13.0. The mounted actor environments remain in use; the selected generation engines report vLLM0.29.0. The image tag's historical vLLM version does not describe the mounted Python runtime. Every replica's prepared-input check reports41 unchanged model files.

Actual Ray actor placement reconciles48 generation actors and18 Megatron actors: student8 on gpu-h800-0582; math/stem/if teachers6 on gpu-h800-0474; code/logic teachers4 on gpu-h800-0108; generation48 across six other nodes. The configured parallelism remains studentTP2/PP4/CP1/DP1,24 TP2 generation engines, and five TP2 teachers. Six GPUs are unassigned. This is separate student/generation/teacher placement; no co-located change was made.

Native restore telemetry records step40, zero complete replay groups,512 unfinished groups found,512 siblings rerun/restarted, and27.643s redispatch scheduling. The total snapshot/data-plane/replay/recovery load is6.609s. This verifies actual recovery execution rather than the earlier CPU preflight alone.

Two inspected generation engines log `enforce_eager=False`, compilation backend `eager`, `FULL_AND_PIECEWISE` CUDA graphs and FlashAttention4. Both TP ranks finish graph capture; actual graph pool usage is0.07GiB. Inductor/AOT compilation is disabled by this selected backend. A first-use `_topk_log_softmax_kernel` JIT warning occurs at21:53:48HKT; the logs and training progress do not establish a persistent JIT bottleneck. The initial incorrect `get_log(address='http://10.191.7.40:9285')` read failed because this installed Ray API requires the GCS bootstrap address; using10.191.7.40:9457 succeeded. The read failure had no effect on training.

## Numerical and sample checks

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Native optimizer progress | Match saved parameter-group counters | Eight optimizer shards, all groups Adam112, step130 | PASS |
| Continued updates | Successful native update flags and no unexplained skips | step41–131:91/91 successful; skipped_step0; scale1 | PASS for inspected interval |
| Native metric finiteness | All exact-step metrics finite | Every inspected metric finite in all91 steps | PASS |
| Full-vocabulary objective | Native loss equals full-RKL; no objective substitution | loss equals opd_full_reverse_kl in every inspected step; vocabulary151936 unchanged | PASS |
| Loss/grad health | Finite values with interpretable trend | loss0.11774 atstep41→0.03545 atstep131; gradnorm0.70466→0.13431; LR1e-6 | PASS for inspected interval |
| Batch and masks |256 distinct valid samples/update; positive loss tokens; no masked sequences |23,296 unique samples,135,210,664 response tokens; num_masked_seqs_total0 and sample_loss_mask1 | PASS |
| Original teacher routing | All five source groups represented | math/code/stem/logic/if present in all91 batches | PASS |
| Staleness | Every admitted sample age0–5 | age0/1/2/3:256 each; age4:2835; age5:19437; no age violation | PASS |
| Same-weight scorer/vLLM | MAE≤0.02nats/token, exact tokens/masks |256 age0 samples /856,388 response tokens; pooledMAE0.00118336; every groupMAE≤0.00223202 | PASS |
| Native sequence ceiling | Original65536 configuration; accepted samples within ceiling | Actual longest sequence35,351; all within65536 | PASS for inspected samples; no new actual65536 sample claimed |
| Checkpoint retention | Only newest complete checkpoint; no copies/exports | step130 only;31files;56,842,080,650bytes; step120 removal recorded | PASS |
| Formal total and final validation |200 durable successes; final tensor scan and three-way parity |112durable /113observed; final checks await completion | PENDING |

The same-weight check comes from the restored weight-version40 batch. Most later samples have age4 or5, so scorer-minus-generation differences include actual model updates. They must not be interpreted as same-weight parity or used to establish final model.train() parity. Actual final native training/scorer/vLLM comparison remains required. Completion FIFO's ordinal behavior is established by the committed sampler/recovery tests and runtime config; the current trace does not contain the teacher-enriched completion sequence of each consumed group, so this report does not claim an independent production ordinal audit.

## Measured performance

| Quantity | Old step32–38 | FIFO step51–130 |
| --- | ---: | ---: |
| GPU allocation |72H800 |72H800 |
| Mean step time |650.300s |70.424s |
| Mean exposed ready wait |561.854s /86.40% |0.320s /0.45% |
| Mean policy training |66.926s /10.29% |53.910s /76.55% |
| Mean student scoring |6.863s |5.526s |
| Mean response tokens/update |1,890,494 |1,451,283 |
| Effective response tokens/s, including step overhead |2907 |20,608 |

Observed mean-step speedup is9.234×. The first ten continuation steps average86.441s; steps100–130 average69.176s. Current step131 totals72.021s: ready wait0.121s, scoring5.575s and policy training54.487s. The consumer's readiness wait is no longer the principal observed bottleneck; student training accounts for76.55% of the steady interval.

This is a combined completion-FIFO/staleness/graph change. It is not an isolated FIFO or graph A/B. Mean response tokens are23.23% lower, and these are different fresh batches and optimizer versions. Token-normalized effective throughput is7.09× higher, but it still measures consumer throughput, not isolated GPU decode speed. Native stale-buffer eviction averages47.69 groups/update; trace `group_discarded` also includes normal consumed-ledger cleanup and cannot be counted directly as rollout failure.

The current GPU dashboard samples have timestamps17–21seconds before collection. Generation GPUs are mostly89–100% utilized. One teacher node is idle in this snapshot and the other has two busy GPUs; this is not a complete3-hour utilization window. The historical teacher clean-policy risk remains unresolved and is monitored without synthetic busy work or changing native failure handling.

The larger current checkpoint is not an additional saved copy: its eight TQ storage-unit files total approximately31.95GB and hold the windowed replay/recovery contents, while policy/optimizer files remain approximately24.88GB. These active recovery files were retained. No physical OSS reclamation measurement or shared-trash deletion was performed.

## Commands and evidence

```bash
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh /data/ycfeng/tmp/nemo-mopd-20261112/current_job_state_20261115.py exp-1115-193456-282702 exp-1115-193548-376103
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh /data/ycfeng/tmp/nemo-mopd-20261112/read_step40_fifo_logs_20261115.py
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 PYTHONPATH=/data/ycfeng/sota-engine/RL UV_CACHE_DIR=/data/ycfeng/tmp/nemo-mopd-20261112/uv-cache PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/read_fifo_cpu_snapshot_20261115.py
python3 /data/ycfeng/tmp/nemo-mopd-20261112/analyze_fifo_live_20261115.py
```

State/metric inspection uses the mounted basePython3.13.14/Torch2.13.0 environment and the existing native metadata reader. It reads5,609,356metadata/scalar/replay-metadata bytes and no model/Adam tensor payloads. Personal API reads use the existing Python3.10.6/brainpp2.8.6 environment. Exact-job cloud inventory reads use bounded SSH Python as i-fengyicheng, restricted to /mnt/ws-jfs/ycfeng; they do not submit, stop or delete jobs remotely.

Selected durable evidence is `evidence/formal-step40-fifo-72gpu-{running-status,worker-logs,placement,graph-logs,native-state,selected-metrics,live-health}-20261115.json` and `evidence/formal-step40-fifo-72gpu-training-20261115.csv`. Per-sample numerical evidence is under `evidence/formal-step40-fifo-72gpu-live/parity/`; compact summaries are written and read back before completed raw cloud dumps are deleted. The raw native event stream and trace remain on the formal run's cloud path.

Next actions: preserve this single72GPU writer and its launcher; verify subsequent native publications and latest-only deletion; continue to Adam200; run the existing final full-tensor CPU scan and unsaved16-H800 three-way replay; refresh the final curves and English archive. Auxiliary evidence failures remain soft and never control healthy training.
