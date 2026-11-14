## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Add reproducible curves and exact CSV values for the first17 formal successes through controller step35. |

# Formal MOPD training health curves

Status: the observed steps18–35 are plotted and checked, covering17 successful optimizer updates and the previously recorded step25 overflow skip. This is an intermediate snapshot; the200-success and final actual-training parity requirements remain incomplete. The original writer continues during this CPU analysis.

## Inputs, commands and environment

Script: `tests/manual/mopd/plot_training_health.py`. Inputs are the existing native TensorBoard scalar exports and complete per-step response-token summaries from formal segments3/5/6. The script selects the latest event for each tag/step, retains the missing gradient norm on an unsuccessful update, and writes the plotted values to CSV. Source job identity is retained on each CSV row. No weights, native checkpoint payloads or HF exports are read or written.

Runtime: local offline UV0.13.7, Python3.13.14, Matplotlib3.11.2. The original `systemd-run --scope -p MemoryMax=30G` invocation failed before Python started with `Interactive authentication required`. The successful command instead sets a30GiB address-space limit and a60-second timeout; no additional privileges or dependencies are required:

```bash
ulimit -v 31457280
mopd_tmp=/data/ycfeng/tmp/nemo-mopd-20261112
env TMPDIR=/data/ycfeng/tmp \
  MPLCONFIGDIR="$mopd_tmp/matplotlib" \
  UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 60s "$mopd_tmp/uv-cli/bin/uv" run --offline --no-sync python \
  tests/manual/mopd/plot_training_health.py \
  --metrics "$mopd_tmp/formal-native-resume-3-training-metrics-20261114.json" \
    "$mopd_tmp/formal-native-resume-5-training-metrics-20261114.json" \
    "$mopd_tmp/formal-native-resume-6-training-metrics-20261114.json" \
  --parity-directory "$mopd_tmp/formal-native-resume-3-current-inputs-20261114/parity" \
    "$mopd_tmp/formal-native-resume-5-current-inputs-20261114/parity" \
    "$mopd_tmp/formal-native-resume-6-current-inputs-20261114/parity" \
  --through-step 35 --mae-limit 0.02 --grad-clip 1.0 \
  --output task_memory/task_2026-11-12_mopd_full_vocab_validation/evidence/formal-training-health.png
```

Output: `evidence/formal-training-health.png` and `evidence/formal-training-health.csv`. These two small files are updated in place as the run advances. The PNG contains six plots: full-vocabulary loss, pre-clipping gradient norm, cumulative successful updates, per-group scorer/vLLM MAE, per-group response length and native timing. This comparison uses independent scorer outputs; it does not measure the actual training forward.

## Verification

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Completed attempt coverage | One row per observed attempt, no gaps or duplicates |18rows,steps18–35 | PASS |
| Successful-update accounting | Count native flags, preserve the skip |17successes;skip25;CSV agrees with all native flags | PASS |
| Gradient norm | Exact emitted norm on successful steps; no borrowed value on skip |17exact matches;step25CSV cell empty and plot has a gap | PASS |
| Per-group parity | Five groups on each step; values match input summaries |90group values match exactly; all means below0.02 | PASS for the measured mean criterion |
| Sample/token coverage | CSV/event counts match the complete sample summaries |256samples/step;positive token counts agree for all18rows | PASS |
| Figure | Readable axes, distinct groups and visible skip/thresholds | Local image inspection confirms six readable plots | PASS |
| Source checks | Scoped Ruff check and format | Both exit0; no unrelated files checked or changed | PASS |
| Main acceptance |200successful updates and final native three-way replay |17successes in this snapshot; final replay pending | INCOMPLETE |

The independent local CSV check reopens all three scalar inputs, matches every successful flag and norm by step, and checks all overall/per-group MAEs against the summaries. All checked numeric values agree exactly; step25 retains no norm. The script is verified by running the real inputs; no unit test duplicating script settings was added.

Scoped lint commands:

```bash
env TMPDIR=/data/ycfeng/tmp \
  UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-gym-servers/responses_api_models/vllm_model/.venv \
  /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync \
  ruff check tests/manual/mopd/plot_training_health.py
env TMPDIR=/data/ycfeng/tmp \
  UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-gym-servers/responses_api_models/vllm_model/.venv \
  /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync \
  ruff format --check tests/manual/mopd/plot_training_health.py
```

## Observations and limits

Loss declines from0.1903821 atstep18 to0.1349312 atstep35, with intervening increases; the endpoint change is−29.13%. The gradient norm declines from1.1475327 to0.5589148, with a maximum1.1761640. Steps18–21 have pre-clipping norms above the configured1.0 threshold. These are observations across different fresh batches, not a fixed-data convergence or accuracy test. The native loss scale remains1, and the history still contains the known step25 skip.

Overall scorer/vLLM MAE ranges0.0011886–0.0013359. Group mean ranges are math0.0010523–0.0011720, code0.0013135–0.0014703, stem0.0010741–0.0012411, logic0.0010988–0.0013152 and if0.0018375–0.0036963. The if group has the largest means but remains below0.02. Its historical step25 maximum token error2.6184369 remains visible in the CSV; the existing scaler report records its repetitive-output concentration. This curve adds no causal claim about that tail or the overflow operator.

For the latest five steps, mean native step time is611.0577seconds and exposed generation accounts for84.52% of summed step time. Per-step generation takes466.3–557.1seconds and policy training58.2–85.5seconds. These timings identify where the current run spends time; they are not a promise for the remaining183successful updates and do not authorize changing256fresh trajectories or the65536sequence ceiling.

Native metrics do not emit per-group full-vocabulary RKL here. The per-group plots therefore show the measured scorer/vLLM error and response length, without inventing per-group full-loss values. Final tensor finiteness, actual-training/scorer/vLLM parity, all200successful updates and final terminal states remain pending.
