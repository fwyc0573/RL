## Modification History

| Date | Change |
| --- | --- |
| 2026-11-16 | Archive the paused-run analysis and identify validation items that remain unverified. |

# Paused MOPD validation and recorded metric analysis

On November 16, 2026, the user stopped the training campaign and replaced the 200-update target with analysis of the last completed step. This archive closes that analysis scope. It does not declare the original exhaustive GPU validation contract passed.

The writer `exp-1115-193456-282702` and observer `exp-1115-193548-376103` were stopped through the required personal Python stop path. All nine H800 replicas and the zero-GPU observer replica are Stopped. Final replay waiter PID2019267 was terminated before stopping the writer; no final replay was launched.

The authoritative restored chain ends at step176:159 attempts,158 successful optimizer updates, the previously diagnosed step25 overflow only, and136 consecutive successful FIFO updates. TensorBoard, compact sample summaries and the `step_trained` trace agree on this frontier. Each attempt has256 valid samples; all40704 sample IDs are distinct. The older writer's discarded post-step40 attempts are excluded.

The sole durable checkpoint remains step170/Adam152 at `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-five-teacher-fp16-resume/exp-1113-174329-659007/checkpoints/step_170`. Eight optimizer shards and all16 param-group counters match152. It contains31 files and55,182,356,048 bytes. No new checkpoint, weight copy or HF export was saved on pause. The six observed updates171–176 were not checkpointed.

## Deliverables

All paths below are relative to `task_memory/task_2026-11-12_mopd_full_vocab_validation/`:

- `test_report_2026-11-16_paused_training_metrics.md`: self-contained Chinese analysis, commands, observations, issue classifications and verification limits.
- `evidence/formal-training-health.png` and `.csv`: six-panel figure and exact authoritative steps18–176 values.
- `evidence/formal-training-health-verification-20261116.json`: independent row, sample, mask, group, age and scalar verification.
- `evidence/paused-training-metric-analysis-20261116.json`: window and group metrics, input hashes, admission/training trace reconciliation and native counters.
- `evidence/paused-training-scalar-statistics-20261116.json`: every observed scalar tag's finite range and historical nonfinite-event list.
- `evidence/user-pause-terminal-20261116.json`: exact stopped job identities and replica states.
- `evidence/formal-step40-fifo-72gpu-live/parity/`:136 compact step summaries, including newly archived171–176.
- `requirements.md`, `plan.md`, `progress.md`: latest user decision, paused-analysis acceptance and evidence ledger.

## Findings

Latest loss/full-RKL is0.0243255217 and pre-clipping gradient norm is0.0819471106, both finite. Recent20 steps have mean loss0.02506212 and gradient norm0.09537221. No new skip, masked sample or zero-valid-token issue exists. Constant LR1e-6, WD0.01 and loss scale1 match configuration. Historical stddev NaNs at18–23 are the previously repaired singleton aggregation issue; all recorded24–176 scalar events are finite.

The main concern is training-mixture drift: admission is approximately20% per teacher, but actual FIFO logic samples are12.86%, falling to11.15% in the last20 steps. All five teachers remain covered. Completion order and age selection are consistent with slow groups being underconsumed, but the trace omits per-group discard reasons, so an exclusive causal attribution is not claimed.

Malformed-thinking flags rose from1.05% in early steady steps51–70 to7.17% in the last20 steps, with8.01% in the last10. The flag detects thinking-tag placement or repetition, not answer correctness. Compact summaries omit original texts and individual flag provenance, so actual format degradation versus parser compatibility remains unverified.

Most FIFO samples have age5:85.36% overall and89.71% in the last20 steps. There were6229 stale-group evictions. This is an observed cost and distribution issue, with no age-bound violation. The global token-normalized objective also means IF contributes only4.28% of recent response tokens despite23.63% of samples.

Recent20-step mean wall time is70.167s, with53.168s policy training (75.77%) and0.581s consumer wait. Teacher request and lock timers sum overlapping calls and cannot be added to step wall time. Dump serialization is mostly nested within the advantage timer; checkpoint time is already included in the outer total timer.

Historical age0 scorer/vLLM pooled MAE0.00123753 passes the existing0.02 mean criterion. The last step contains age4–5 samples only; its0.00532999 MAE includes weight drift and does not establish latest same-weight parity.

## Remaining limits and deferred work

Final model/Adam tensor scanning, actual model.train()/scorer/vLLM final three-way parity, per-group full-RKL and gradient curves, fixed-set accuracy, and the RCA of individual malformed outputs were not executed or not recorded. The user superseded200 updates and final GPU replay for this phase; neither is claimed passed. Training must remain paused unless the user requests resumption. Future equal-mixture selection should preserve group-local FIFO and age semantics and measure the cost of slow-group waiting.
