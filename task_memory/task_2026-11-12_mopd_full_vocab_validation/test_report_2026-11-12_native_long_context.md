## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record BF16 65,536-token native execution and failing same-weight parity. |

# Native 65,536-token BF16 control

Status: length and native optimizer execution PASS; same-weight scorer/vLLM parity FAIL. This frozen initial-student self-control does not establish distinct-five-teacher training or nontrivial teacher-driven parameter changes.

Job exp-1112-181524-408383 used 24 H800/codesign, TP2/CP4, personal i-fengyicheng, local NFS100.96.128.195:/data/ycfeng:/data/ycfeng. Platform Succeeded with all three replicas at2026-11-12T10:31:01Z; native finished.json identifies the same job. All five input rows had65,504 prompt and32 response tokens, for65,536 total. One successful native update, skipped_step=0, valid sequences=5, valid tokens=160; loss=0 and grad norm=0.

| Comparison | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Actual total length |65,536 for all5rows|65,536 for all5rows|PASS|
| Native successful updates |1|1|PASS|
| Loss/grad finite |Both finite|0 /0|PASS|
| Scorer/vLLM response LP MAE |≤0.02 nats/token|0.1361867711|FAIL|
| Parity tails |Report separately|p950.320202446; p992.77063072; max6.68006516|Observed|

Five per-sample means are0.04744163/0.06472498/0.05306252/0.42868447/0.08702026. No mask filtering or threshold change. Independent scorer is a separate native forward; actual training equality was established in short observed controls, so long actual training/vLLM is not directly observed yet. Selfteacher/student native loss0 suggests identical native outputs in this case but does not settle vLLM consistency.

The raw all-metrics finite verdict includes NaN per-group singleton stddev. nemo_rl/experience/metric_utils.py:47 explicitly returns math.nan when there is only one value. Key training metrics contain no NaN; do not count this defined statistic as a NaN loss or gradient. Evidence JSON separates the two observations.

Commands: PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_native_long_context.py, using native-long-context-submit-args.json, tests/manual/mopd/submit_on_rjob.py and run_on_rjob.py. Inputs: tasktmp/long-context-inputs.jsonl, SHA256bc162cf4696b1aefc828124671b4e2e1174cb02ef7e8f7dfe1f705845e6a525e. Exact native config validation output is native-long-context-config-validation.log. Analysis inputs and raw native-long-context-results.json remain under tasktmp; selected metrics, lengths, errors and terminal receipt are evidence/native-long-context-parity.json.

Runtime: mounted Python3.13 native actor environments and exact package versions in evidence/native-runtime-versions.json; personal launcher Python3.10.6/brainpp2.8.6. Large artifacts remain under/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/native-long-context/exp-1112-181524-408383/.

Next affected verification: paired FP32 student/vLLM/teacher heads with the same five inputs, length and parallelism, job exp-1112-183922-688559. If it remains above0.02, isolate CP/packing and long-position attention/RoPE through existing model paths. Main200updates and checkpoint/resume remain outstanding.

## Paired FP32 control and fixed-token replay

Paired FP32 job exp-1112-183922-688559 Succeeded at2026-11-12T10:52:10Z, all3replicas succeeded. Native finished.json and SC1/1update agree. Engine cleanup and worker stacks occur after business completion. They do not indicate a failed training run.

| Comparison | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Total length |65,536 for all5rows|65,504prompt+32response each|PASS|
| Successful / skipped updates |1 /0|1 /0|PASS|
| Key loss / gradient norm |Finite|1.50832058e-8 /0.00406362070|PASS|
| Valid sequences / tokens |5 /160|5 /160|PASS|
| Scorer/vLLM MAE |≤0.02|0.1310551181|FAIL|
| p95 /p99 /max absolute error |Report|0.315386868 /2.742420683 /6.692833900|Observed|

All5per-sample MAEs fail0.02:0.059144416,0.039555755,0.047824948,0.424820966,0.083929505. Signed mean=-0.090743085. These are fresh outputs of the same fixed synthetic prompts, not a paired comparison on identical BF16/FP32 generation trajectories. The small change from BF16 MAE0.13619 supports investigating beyond head rounding; it does not isolate a causal component by itself.

Command: `PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_native_long_fp32.py`, exact args native-long-fp32-submit-args.json. Analysis: mounted base Python3.13 through `uv run --no-sync python` on read_long_fp32_scalars.py and analyze_long_fp32_parity.py, both under tasktmp. Input receipts: native-long-fp32-analysis-inputs; selected evidence/native-long-fp32-{parity,terminal}.json. Cloud:/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/native-long-fp32/exp-1112-183922-688559/.

The8GPU fixed-token replay exp-1112-184750-688596 failed before any scorer forward. It omitted the controller's `_maybe_inject_megatron_train_iters` configuration preparation; native `_validate_training_config` asserted at10:53:06Z, worker exit1 at10:53:10Z, RJob Failed10:53:14Z, launcher cleanup10:53:16Z. Exact-object RCA returned live/historical Events, incident logs and zero existing DeepScan flows. Both all-operation audit queries timed out. Stop actor/reason/request time are UNKNOWN. Earlier ConfigMap waiting resolved before container start and did not cause this terminal failure. No positive preemption/OOM evidence was returned; coverage is limited.

After reusing the original preparation function, CPU execution of the actual native guard reproduces RED before injection and GREEN for CP4/CP1 afterward. CLI help and Ruff PASS. Exact command uses UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker, PYTHONPATH=/data/ycfeng/sota-engine/RL, CUDA_HOME=/data/ycfeng/toolchain/cuda-13.0.2 and existing cuda-compat LD_LIBRARY_PATH, then uv run --no-sync python tasktmp/check_replay_native_config.py. No persistent unit tests for the manual script were added.

Affected replay rerun exp-1112-185915-382589 uses the identical recorded BF16 long token IDs/mask/checkpoint, sequential CP4/CP1 atDP1, same packing. Launcher53211/monitor73684, new native-long-cp-replay-2 prefix. GPU scoring remains UNVERIFIED until collected. Formal200updates remain unsubmitted.

## Fixed-token CP replay and RoPE initialization repair

Native fixed-token replay `exp-1112-185915-382589` Succeeded. It uses the same initial checkpoint, 5 recorded 65536-token sequences, identical response mask, TP2 and DP1. CP4 uses 8 GPUs; CP1 uses 2 of the same 8-GPU allocation. Optimizer and reference model are disabled; no new generation is involved.

| Comparison | Response tokens | MAE | p95 | p99 | Max | Verdict at 0.02 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CP4 minus original scorer | 160 | 0 | 0 | 0 | 0 | PASS |
| CP4 minus original vLLM | 160 | 0.13618677 | 0.32020245 | 2.77063072 | 6.68006516 | FAIL |
| CP1 minus original vLLM | 160 | 0.12153486 | 0.12904083 | 2.81014998 | 6.73461437 | FAIL |
| CP1 minus CP4 | 160 | 0.04371640 | 0.08855095 | 0.44826845 | 0.72641373 | Numerical-control observation |

All observed logprobs are finite. CP affects the numerical result; switching to CP1 alone does not meet acceptance. This excludes the proposed CP-only remedy and keeps long-position attention/RoPE as the next native path to isolate. No tolerance or token filtering changed.

Submit command: `PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_native_long_cp_replay_2.py`. Exact args: task temporary `native-long-cp-replay-2-submit-args.json`. Reusable script: `tests/manual/mopd/replay_logprobs.py`. Selected evidence: `evidence/native-long-cp-replay.json` and `evidence/native-long-cp-replay-terminal.json`. Durable output: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/native-long-cp-replay-2/exp-1112-185915-382589/logs/replay-logprobs.json`.

Unfused RoPE control `exp-1112-190831-616219` failed before model initialization at 11:10:13Z. The actual parser error is `argument --context-parallel-sizes: invalid int value: 'policy.megatron_cfg.apply_rope_fusion=false'`. This is a command-order error, not a RoPE measurement. Reordering the override before `--dump` passes actual native CLI RED/GREEN checks in `check_unfused_replay_cli.py` (`UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 uv run --no-sync python ...`). Personal dry-run passes. One affected 2-H800 rerun `exp-1112-191624-359257` was submitted at 19:16:24 HKT with `submit_native_long_unfused_rope_replay_2.py`; its numerical result remains pending.

Scoped RCA collector retained exact objects, live/historical Events, incident logs and existing DeepScan query. Both all-operation audit requests timed out at 45s; collector exit1 means incomplete query coverage. No returned positive OOM or preemption record establishes such a cause. Manual-stop actor/reason/request-time remain UNKNOWN. Parser/native exit precedes launcher cleanup. Raw RCA: `/data/ycfeng/tmp/nemo-mopd-20261112/rca-unfused-rope-init-1/`.


## CP1 unfused RoPE result and pending attention controls

`exp-1112-191624-359257` Succeeded. Identical initial BF16 checkpoint, recorded65536-token sequences and all160 response positions; only existing `apply_rope_fusion=false` differs from CP1/fused.

| Metric | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Overall response-token MAE | ≤0.02 | 0.1005940875 | FAIL |
| Every sample MAE | ≤0.02 | 0.06673004 /0.08895532 /0.07155938 /0.20877250 /0.06695320 | FAIL |
| p95 /p99 /max | Report | 0.24005744 /0.51850528 /5.26047039 | Observed |
| All response logprobs finite | True | True | PASS |

Command: `PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_native_long_unfused_rope_replay_2.py`. GPU output: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/native-long-unfused-rope-replay-2/exp-1112-191624-359257/logs/replay-logprobs.json`. Selected evidence: `evidence/native-long-unfused-rope-replay.json`.

Switching CP and disabling RoPE fusion do not individually meet the unchanged criterion. FA4 CP1/fused `exp-1112-193156-178680` and vLLM native prompt scoring `exp-1112-193212-322978` are submitted, not measured. Their output prefixes are native-long-fa4-replay and native-long-vllm-prefill-replay. vLLM scores all original65536 tokens and generates one unused new token with engine capacity65537; this capacity is diagnostic only. Exact submission args and wrappers reside under task temporary directory with corresponding names.

Final executable-script verification: `UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --no-sync /data/ycfeng/tmp/nemo-mopd-20261112/uv-cache/archive-v0/ZpRSfVMFE7WBmM2b/ruff-0.9.9.data/scripts/ruff check tests/manual/mopd/{replay_logprobs,replay_generation_logprobs,resume_training}.py`, plus `format --check` on the same files. Actual output: All checks passed;3files already formatted. GPU successful CP4 reproduction and CP1 scoring are the functional checks of replay_logprobs.py; no unit test of script settings is added.


## FA4 fixed-token result

FA4 CP1/fused `exp-1112-193156-178680` completed the native replay and published matching `finished.json`; launcher exit0. Scorer/vLLMMAE0.1298855153 fails0.02; p950.19516401, p992.84165693, max6.74194527. Per-sample MAEs0.03664838/0.06462016/0.04545669/0.42011610/0.08258625 all fail. All160response logprobs are finite. No filtered positions or threshold changes. Exact command is tasktmp/submit_native_long_fa4_replay.py through personalStepMindPython; exact args native-long-fa4-replay-submit-args.json. Selected evidence/native-long-fa4-replay.json; cloud native-long-fa4-replay/exp-1112-193156-178680/logs/replay-logprobs.json under the task's personal prefix.

The pending vLLM native prompt replay uses synchronous defaults (observed batched-token cap16384) versus original asynchronous default2048. Any gap will require a matched chunk-cap control before exclusively attributing it to prefill/decode history.


## FA4 and native vLLM prompt replay results

FA4CP1/fused job `exp-1112-193156-178680` completed business scoring and launcher exit0. Native same-tokenMAE0.1298855153, p950.19516401, p992.84165693, max6.74194527:FAIL0.02. The attention-version setting alone is insufficient. Selected raw output:`evidence/native-long-fa4-replay.json`.

Default vLLM prompt replay `exp-1112-193212-322978` completed5samples/160response positions; prompt-minus-recorded async generationMAE0.0368034811, p990.28300462, max0.99954462:FAIL0.02. Native synchronous scheduling selectedchunk16384 versus originalasync2048, so this is not an isolated prompt/decode comparison. Selected output:`evidence/native-long-vllm-prefill-replay.json`.

Matchedchunk2048 replay `exp-1112-194130-498485` Succeeded2026-11-12T11:45:09Z, launcher58813exit0. Same weights/tokens/mask,TP2/FA3/BF16/prefixcache; MAE0.03742300957, p990.35544111, max0.91518831:FAIL0.02. Chunk matching is not a sufficient remedy. SourceNFS100.96.128.195:/data/ycfeng:/data/ycfeng, creator i-fengyicheng/codesign. Terminal receipt:`evidence/native-long-vllm-prefill-2048-terminal.json`. Native/internal differences still need causal diagnosis.

Commands use `submit_native_long_fa4_replay.py`, `submit_native_long_vllm_prefill_replay.py`, and `submit_native_long_vllm_prefill_2048_replay.py` under `/data/ycfeng/tmp/nemo-mopd-20261112/`, via `PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u`. Exact matching `*-submit-args.json` files preserve overrides. Large outputs remain under corresponding personal cloud prefixes.

Layer diagnosis preparation: actual CLI/config checks in tasktmp/check_native_layers_cli.py and check_generation_layers_cli.py pass, including real sequence-packing sharding of32capture positions. PinnedRuff0.9.9check/format and personal dry-runs pass. Two2H800jobs submitted at19:49:49HKT: native`exp-1112-194949-717142` and eager vLLM`exp-1112-194949-673462`. GPU hooks and their numerical results remainUNVERIFIED. The eager engine's logprobs must be compared separately before attributing its states to originalasync behavior.
