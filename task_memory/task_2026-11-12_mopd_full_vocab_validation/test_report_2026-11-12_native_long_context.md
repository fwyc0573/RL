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
