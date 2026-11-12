## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record the completed native self-control, successful optimizer updates and numerical limitations. |

# Native initial-student self-distillation control

Status: native workflow and five successful optimizer updates PASS. Independent scorer/vLLM overall mean criterion PASS; IF-group mean and error tails remain under investigation. Actual training-forward parity, hidden/direct-logit comparison and five distinct teachers remain UNVERIFIED.

## Execution identity

Job `exp-1112-173036-554443`, UID `e7d21741-1753-4bec-8dd8-70d133f6f998`, personal creator `i-fengyicheng`, 3×8 H800/codesign, nonpreemptible. Local NFS source `100.96.128.195:/data/ycfeng:/data/ycfeng`. Company image and complete argument vector are in `evidence/self-diagnostic-submit-args.json`; both launch scripts are committed in `f92b7685`. Mounted native Python3.13.14/Torch2.13.0+cu130 and actor revisions are in `evidence/native-runtime-versions.json`.

The native student and frozen selfteacher use the specified initial R1-Distill-Qwen-1.5B checkpoint. All five group aliases point to that checkpoint in this control. GPU logs confirm vocabulary151936 and zero dummy tokens. This is full-vocabulary reverse KL; it is not production five-teacher coverage.

Cloud run directory: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/self-diagnostic/exp-1112-173036-554443/`.

## Commands and artifact analysis

Submit with personal `/data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/submit_native_control_5.py`, after runbook initialization. It replays the recorded exact argument vector using `tests/manual/mopd/submit_on_rjob.py`. Native worker entrypoint is `tests/manual/mopd/run_on_rjob.py`; it invokes `uv run --no-sync python -u examples/run_grpo_single_controller.py --config ...` and records the exact command in the cloud `command.json`.

Downloads use read-only SSH as `i-fengyicheng` from `/mnt/ws-jfs/ycfeng/...`, retained under `/data/ycfeng/tmp/nemo-mopd-20261112/native-5-analysis-inputs/`. Each dump has32 rows, with exact IDs, response masks, generation logprobs, independent trainer `prev_logprobs`, frozen-teacher logprobs and weight versions.

Analysis invocation: `/data/ycfeng/envs/nemo-mopd-py313/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/analyze_native_control_5.py`. Input identities were read from only160 selected dataset indices. TensorBoard numeric values were recovered with its native `EventFileLoader`, because copied filenames without `tfevents` cause `EventAccumulator` to return empty tags; original event records are intact. Raw scalar results: `native-5-scalars.json`. No redownload, trainer alteration or threshold change was required.

## Observed results

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Native business/platform completion | Controller5 steps; successful native receipt; job and3replicas succeeded; launcher0 | All observed; platform finish09:44:05Z; launcher71916 exit0 and monitor32270 exit0 | PASS |
| Successful optimizer updates | Five true native optimizer success flags; no skipped update | `[1,1,1,1,1]`; skipped_step all0 | PASS |
| Valid token normalization |32 valid samples and positive response-token count per update |32samples/4096tokens each;160samples/20480response tokens total | PASS |
| Token IDs, mask, alignment and versions | Prompt prefix equal; no prompt tokens scored; same generation/trainer version | All160 samples pass; versions0–4; native lag0 | PASS for dump scorer path |
| Full-vocabulary loss/RKL | Finite; existing smoke median<0.02, min>-1e-3, max<0.5 | Values below; max reported RKL0.0214047; min reported>=-3.092e-7 | PASS for existing smoke guards; residual explanation pending |
| Gradient, LR and clipping | Finite preclip norm; LR1e-6, clip1.0 | Grad0.03493–0.06668, LR~1e-6, WD~0.01; no clipping | PASS for finite/configured values; no parameter-delta proof yet |
| Independent scorer/vLLM mean |<=0.02nats/token on same-version response tokens | Overall0.014616026; each step0.013099–0.016504 | PASS for the declared overall mean |
| Tails and group differences | Investigate systematic numerical differences | IF0.0219348; p990.1227255; max0.3456368 | UNRESOLVED |
| Sample failures/accounting | Native success counts reconcile; no unexplained drops | Final committed160; skipped/data failures/infra drops0 | PASS for this control |
| Five distinct teacher models | Every requested teacher executes | teacher_alias_unique5, teacher_model_unique1 | UNVERIFIED: selfteacher control only |
| Actual training-forward parity | Direct numeric comparison of training output and generation | Dump is written before train; sampled logprobs not computed inside this full loss | UNVERIFIED |
| Native payload, long context, save/resume and200 updates | Plan D/E acceptance | Not exercised here | UNVERIFIED |

| Step | Full RKL / loss | Grad norm | Scorer/vLLM MAE | Signed mean | p95 | p99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
|1|0.000237406479|0.0349325202|0.0145550513|-0.000518805286|0.0835879445|0.123305273|0.246927977|
|2|0.000450705964|0.0496489592|0.0133613089|-0.00110238499|0.0803160965|0.118872714|0.275139809|
|3|0.000456191803|0.0642045513|0.0130991837|-0.000241173889|0.0794182718|0.122459054|0.345636845|
|4|0.000513846811|0.0666771755|0.0155604|-0.000918779429|0.0866666138|0.122335234|0.326391339|
|5|0.00055432244|0.056791205|0.0165041861|-0.000671428615|0.087926954|0.124894643|0.251918077|

Full-loss entropy+cross-entropy decomposition max reported error9.20e-7 supports arithmetic consistency, not correctness of the transported teacher distribution. The frozen selfteacher and student start identical but BF16 forward/projection shapes may differ; nonzero self-loss/gradient must be explained with native controls. No claim that fresh-batch loss should fall monotonically is made.

Selected evidence: `evidence/self-native-5-terminal.json`, `evidence/self-native-5-training-metrics.json`, `evidence/self-native-5-parity.json`. Percentiles are computed over absolute response-token logprob differences. Signed errors are trainer minus vLLM. Per-group and length breakdowns are retained in parity evidence. These are observations; their cause is not yet established.
