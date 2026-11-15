## Modification History

| Date | Change |
| --- | --- |
| 2026-11-16 | Refresh the paused authoritative chain throughstep176 and link the final metric analysis. |
| 2026-11-16 | Refresh the authoritative restored chain through step147; distinguish same-weight error from stale-weight drift and verify129 successes. |
| 2026-11-14 | Emit LF CSV records after the whitespace check identified the default CRLF terminators. |
| 2026-11-14 | Add reproducible curves and exact CSV values for the first17 formal successes through controller step35. |

# Formal MOPD training health curves

The current snapshot covers controller steps18–176:159 actual attempts,158 successful optimizer updates, and the previously diagnosed step25 overflow. The writer and observer are Stopped by user request. The 200-update target and final GPU replay were superseded on November16; actual final training/scorer/vLLM parity remains unverified. The dated paused analysis is test_report_2026-11-16_paused_training_metrics.md.

The updated figure and CSV remain `evidence/formal-training-health.png` and `evidence/formal-training-health.csv`. The previous step35 snapshot remains available at the earlier Git commit. No model, optimizer tensor, diagnostic checkpoint or HF export was written.

## Inputs, command and environment

Script: `tests/manual/mopd/plot_training_health.py`. Native scalar exports are deduplicated by latest wall time. Legacy summaries are selected only through the actual restored step40; the earlier writer's later attempts were discarded by recovery and are excluded. Steps41–176 use the current FIFO writer's complete sample summaries. A temporary directory contains23 symlinks to these existing legacy summaries, with no tensor copies.

Each sample's weight age is calculated from its recorded generation version and the native scorer version `step-1`. The CSV now records minimum/maximum age, same-weight sample/token counts, and pooled same-weight MAE. Missing same-weight observations remain empty CSV cells and gaps in the figure. Group curves include stale-weight differences and are explicitly labelled as including model weight drift. The0.02 line applies to same-weight MAE only. The existing selector and plot interface are unchanged.

Runtime: offline UV0.13.7, Python3.13.14, Matplotlib3.11.2. Local commands use /data/ycfeng/tmp, a30GiB address-space limit and a60-second timeout. The prior systemd MemoryMax scope was unavailable because it required interactive authentication.

```bash
ulimit -v 31457280
TMPDIR=/data/ycfeng/tmp \
MPLCONFIGDIR=/data/ycfeng/tmp/nemo-mopd-20261112/matplotlib \
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
timeout 60s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python \
 tests/manual/mopd/plot_training_health.py \
 --metrics /data/ycfeng/tmp/nemo-mopd-20261112/formal-native-resume-3-training-metrics-20261114.json \
 /data/ycfeng/tmp/nemo-mopd-20261112/formal-native-resume-5-training-metrics-20261114.json \
 /data/ycfeng/tmp/nemo-mopd-20261112/formal-native-resume-6-training-metrics-20261114.json \
 /data/ycfeng/tmp/nemo-mopd-20261112/formal-step40-fifo-72gpu-training-metrics-20261115.json \
 --parity-directory /data/ycfeng/tmp/nemo-mopd-20261112/formal-authoritative-legacy-parity-20261116 \
 task_memory/task_2026-11-12_mopd_full_vocab_validation/evidence/formal-step40-fifo-72gpu-live/parity \
 --through-step 176 --mae-limit 0.02 --grad-clip 1.0 \
 --output task_memory/task_2026-11-12_mopd_full_vocab_validation/evidence/formal-training-health.png

UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 \
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python \
 /data/ycfeng/tmp/nemo-mopd-20261112/verify_paused_health_curves_20261116.py

UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-gym-servers/responses_api_models/vllm_model/.venv \
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync \
 ruff check tests/manual/mopd/plot_training_health.py
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-gym-servers/responses_api_models/vllm_model/.venv \
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync \
 ruff format --check tests/manual/mopd/plot_training_health.py
```

## Verification

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Authoritative attempt chain | One row per actual resumed attempt; discarded attempts excluded |159rows,steps18–176; four exact source RJobs | PASS |
| Optimizer accounting | Native flags and cumulative count agree; preserve skip |158successes;skip25;136consecutive FIFO successes | PASS |
| Native values | CSV scalars and per-group values equal exact sources | All rows checked; missing gradient onskip25 preserved | PASS |
| Samples/masks/length/teacher routing |256distinct samples/update,fivegroups,finite logprobs,valid mask,ceiling65536 |40,704distinct sample IDs;243,350,200 response tokens; no issues | PASS inspected chain |
| Staleness | Actual generation/scoring versions resolve age0–5 |6144age0;256each age1/2/3;4073age4;29719age5 | PASS |
| Same-weight error | Positive observed coverage; pooled MAE≤0.02 |6144samples,46,227,111tokens;pooledMAE0.00123752925 | PASS measured scorer/vLLM mean |
| Figure | Correct labelled plots; skip and missing observations visible | Actual PNG inspection confirms six readable panels and explicit drift label | PASS |
| Source checks | Scoped Ruff, format and whitespace | All exit0 after actual execution | PASS |
| Paused analysis | Exact terminal/frontier report; no new training or save | Complete through176; final tensor/train-forward checks remain unverified | PASS analysis scope |

Verification evidence: `evidence/formal-training-health-verification-20261116.json`. Every plotted scalar, group mean, sample count, response token count and age is reopened and checked. The chart script is tested by its actual workload; no unit test duplicating the script's settings was added.

The first live plot attempt saw the optimizer flag atstep147 before that step's timing events had reached the observer snapshot, raising KeyError147. Refreshing the actual observer snapshot supplied the missing native events; the unchanged strict completeness check then passed. This was not a training failure.

The first independent verification used the host Python3.10 rather than the plotting Python3.13 and found pooled summation differences of at most1.301e-18. Repeating in the exact plotting runtime gives equality without a tolerance change. Production metrics, thresholds and code are unchanged by this inspection correction.

## Observations and limits

The final paused chain shows loss0.190382123→0.0243255217 (-87.22%), grad1.14753270→0.0819471106 and136consecutive successful FIFO updates. Onlysteps18–21 exceed pre-clipping norm1.0; skip25 remains visible. Recent20steps157–176 average70.167365s total,0.581097s exposed wait,53.167988s policy training.

The figure was reopened and inspected after refresh. EveryCSVscalar/groupmean/samplecount/token count/age was independently checked in the plottingPython3.13.14 runtime; output is formal-training-health-verification-20261116.json. The original scoped Ruff checks remain valid because the plotting code was unchanged in this phase. The CSV contains actual values, blanks for absent same-weight observations, and the existing red skip marker.

The paused analysis additionally scans all282tags observed in this step interval:210historical stddevNaNevents are confined to18–23 and explained by the prior pooled-statistics fix. Current24–176 scalar events are finite. Group panels show backend differences and response lengths, not per-group full-RKL, which native metrics do not emit.

The detailed paused report identifies the actual logic sample-mix shift (12.86%FIFO versus20%admission), rising malformed-thinking flags, staleness/evictions, overlapping teacher timers and remaining parity limits. Loss decline alone does not establish fixed-set accuracy or stable per-group progress. Final tensor scanning and actual training/scorer/vLLM threeway checks were not performed; no200-update or finalGPU acceptance is claimed.
