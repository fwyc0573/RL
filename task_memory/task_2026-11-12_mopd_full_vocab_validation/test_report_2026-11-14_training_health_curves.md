## Modification History

| Date | Change |
| --- | --- |
| 2026-11-16 | Refresh the authoritative restored chain through step147; distinguish same-weight error from stale-weight drift and verify129 successes. |
| 2026-11-14 | Emit LF CSV records after the whitespace check identified the default CRLF terminators. |
| 2026-11-14 | Add reproducible curves and exact CSV values for the first17 formal successes through controller step35. |

# Formal MOPD training health curves

The current snapshot covers controller steps18–147:130 actual attempts,129 successful optimizer updates, and the previously diagnosed step25 overflow. The original formal writer continues. This snapshot does not establish Adam200 or final actual-training/scorer/vLLM parity.

The updated figure and CSV remain `evidence/formal-training-health.png` and `evidence/formal-training-health.csv`. The previous step35 snapshot remains available at the earlier Git commit. No model, optimizer tensor, diagnostic checkpoint or HF export was written.

## Inputs, command and environment

Script: `tests/manual/mopd/plot_training_health.py`. Native scalar exports are deduplicated by latest wall time. Legacy summaries are selected only through the actual restored step40; the earlier writer's later attempts were discarded by recovery and are excluded. Steps41–147 use the current FIFO writer's complete sample summaries. A temporary directory contains23 symlinks to these existing legacy summaries, with no tensor copies.

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
 --through-step 147 --mae-limit 0.02 --grad-clip 1.0 \
 --output task_memory/task_2026-11-12_mopd_full_vocab_validation/evidence/formal-training-health.png

UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 \
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python \
 /data/ycfeng/tmp/nemo-mopd-20261112/verify_fifo_health_curves_20261116.py

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
| Authoritative attempt chain | One row per actual resumed attempt; discarded attempts excluded |130rows,steps18–147; four exact source RJobs | PASS |
| Optimizer accounting | Native flags and cumulative count agree; preserve skip |129successes;skip25;107consecutive FIFO successes | PASS |
| Native values | CSV scalars and per-group values equal exact sources | All rows checked; missing gradient onskip25 preserved | PASS |
| Samples/masks/length/teacher routing |256distinct samples/update,fivegroups,finite logprobs,valid mask,ceiling65536 |33,280distinct sample IDs;203,270,737 response tokens; no issues | PASS inspected chain |
| Staleness | Actual generation/scoring versions resolve age0–5 |6144age0;256each age1/2/3;3313age4;23055age5 | PASS |
| Same-weight error | Positive observed coverage; pooled MAE≤0.02 |6144samples,46,227,111tokens;pooledMAE0.00123752925 | PASS measured scorer/vLLM mean |
| Figure | Correct labelled plots; skip and missing observations visible | Actual PNG inspection confirms six readable panels and explicit drift label | PASS |
| Source checks | Scoped Ruff, format and whitespace | All exit0 after actual execution | PASS |
| Main completion |Adam200, full tensor scan, actual training/scorer/vLLM parity | Remaining formal/final work | INCOMPLETE |

Verification evidence: `evidence/formal-training-health-verification-20261116.json`. Every plotted scalar, group mean, sample count, response token count and age is reopened and checked. The chart script is tested by its actual workload; no unit test duplicating the script's settings was added.

The first live plot attempt saw the optimizer flag atstep147 before that step's timing events had reached the observer snapshot, raising KeyError147. Refreshing the actual observer snapshot supplied the missing native events; the unchanged strict completeness check then passed. This was not a training failure.

The first independent verification used the host Python3.10 rather than the plotting Python3.13 and found pooled summation differences of at most1.301e-18. Repeating in the exact plotting runtime gives equality without a tolerance change. Production metrics, thresholds and code are unchanged by this inspection correction.

## Observations and limits

Loss changes from0.190382123 to0.0292639453, an endpoint decline of84.6288%. Gradient norm changes from1.14753270 to0.0833321586; maximum1.17616403; onlysteps18–21 exceed clip1.0. All inspected losses are finite, full-RKL equals loss, loss scale1 and256valid samples are retained. These are trends across fresh batches, not fixed-data accuracy or monotonic-convergence guarantees.

Current FIFO steps41–147 average72.1219seconds: exposed ready wait0.58728s and policy training54.8998s. The latest20steps128–147 average71.30795s, ready wait0.36098s and policy54.29877s. The prior measured steady-state comparison51–130 remains9.234× faster than oldsteps32–38 at the same72GPUs. It combines FIFO/staleness/graph and changed response lengths; these curves establish no isolated causal speedup.

The code group's larger early FIFO difference is measured on weight ages above zero. It cannot be compared directly against the same-weight0.02 criterion. Final fresh-response parity will load the final checkpoint and measure the actual model.train(), scorer and generation forwards on identical tokens.

Native metrics do not emit per-group full-vocabulary RKL. The group panels show measured backend differences and response length only. Per-group full-loss trends, final tensor finiteness, actual-training/scorer/vLLM parity,200successful updates and final terminal states are not established by this snapshot.
