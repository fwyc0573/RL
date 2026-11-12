## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record matched FA4 execution, all-token parity and remaining normalization boundary. |

# Matched FA4 long-context comparison

Overall mean absolute logprob error is **0.01905496458 nats/token**, below the unchanged0.02 ceiling. Three individual sample means remain above0.02. This is a partial D result. Compiled FA4 fails the same criterion; final fresh-generation/refit, CP1 training and formal 200 updates remain unverified.

Native exp-1112-210903-042205 completed at2026-11-12T13:13:51Z, launcher5080/monitor59626 exit0. Generation exp-1112-210311-662995 was already completed and its capture is reused. Both use2H800/codesign, personal i-fengyicheng auth, original initial student, CP1/TP2, originaltheta10000, FP32head, correctedFP32RoPE, and explicitly selectedFA4. Native uses unfusedSwiGLU. Native runtime: Python3.13.14, Torch2.13.0+cu130, CUDA13.0; source100.96.128.195:/data/ycfeng:/data/ycfeng.

The five original65536-token cases each contribute every32responsepositions:160tokens total, no filtering. Selected evidence: `evidence/native-long-matched-fa4-{eager-parity,layer-comparison,mlp-comparison,native-terminal}.json`.

| Metric | Observed | Condition/result |
| --- | --- | --- |
| Mean absolute error |0.01905496458|PASS overall mean≤0.02|
| Signed mean |0.00493660311|Observed|
| p50 / p95 |0.01291587949 /0.05413026810|Observed tails|
| p99 / max |0.09266241074 /0.11156654358|Requires remaining boundary diagnosis|
| Sample means in original input order |0.01836089417/0.02007043734/0.01614211872/0.02000783756/0.02069353685|3/5 exceed0.02|
| First QKV/Q/K/attention/projection |MAE/max0 on all samples/ranks|PASS measured first attention path|
| First MLP projection |2/5 exact;3/5sparse errors MAE0.8e-6–1.7e-6,max0.0078125|First retained remainder lies between attentionprojection andMLPprojection|
| Full captures finite |All compared tensors finite|PASS observed captures only|

Exact submission wrapper/arguments: `/data/ycfeng/tmp/nemo-mopd-20261112/submit_long_native_fp32_head_fa4_2.py` and `long-native-fp32-head-fa4-2-submit-args.json`. Launch uses `PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u <wrapper>`. Fetch uses baseUVenv/uv--no-sync python `fetch_native_fa4_2.py layers`; analysis uses `analyze_matched_fa4.py` underOMP/MKL/OPENBLAS4, timeout60s, original CUDAcompat loader andRLIMIT_AS30GiB. Host has no systemd bus; scope attempts fail before analysis, then resource-limited analysis exits0 (`matched-fa4-comparison-3.log`).

Native cloud root: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/long-native-fp32-head-layers-fa4-2/exp-1112-210903-042205/`. Generation root uses `long-generation-core-fp32-layers-theta-fa4/exp-1112-210311-662995/`. Local raw captures stay undertasktmp, notGit.

## Observed normalization boundary and compiled control

The paired observation jobs exp-1112-212103-283781 (native) and exp-1112-212103-266417 (generation) both Succeeded; launchers15358/9966 and monitor99203 exited0. The compiled control exp-1112-211826-090194 Succeeded; launcher34455/monitor98749 exited0. Platform completion here proves capture execution, not numerical acceptance.

Both normalization observers exactly reproduce all160 response logprobs of their original replay (MAE/max0). First-layer native normalization input equals vLLM rounded residual exactly. Two samples have identical first post-attention normalization outputs. Three have1–2 differing BF16 elements per captured49152-element output, MAE1.99e-8–3.97e-8, max0.0009765625–0.001953125. Both closely match residual addition rounded toBF16 followed byFP32 normalization/weight product. This is the first measured remaining boundary; propagation to the final logprob tail is an inference, not a completed full-model intervention.

| Comparison | MAE | p99 | Max | Result |
| --- | --- | --- | --- | --- |
| Native vs eager FA4 |0.01905496458|0.09266241074|0.11156654358|PASS overall;3samplemeans exceed0.02|
| Native vs compiled FA4 |0.03365743577|0.43999475181|0.46564197540|FAIL unchanged0.02 ceiling|
| Eager vs compiled FA4 |0.03563335682|0.44177067637|0.46224468946|Observed|

Compiled logs select PyTorch normalization and allreduce_rms fusion; eager logs select vLLM C++ normalization. This identifies different executed arithmetic paths, without assigning the entire compiled discrepancy to one fusion. Optional transformer_engine import in the vLLM environment fails; no dependency installation or new normalization backend was undertaken.

Analysis scripts: `/data/ycfeng/tmp/nemo-mopd-20261112/analyze_matched_fa4.py` and `analyze_matched_fa4_norm.py`, executed through the mounted Python3.13.14 UV runtime, original CUDAcompat loader, OMP/MKL/OPENBLAS4 and30GiB RLIMIT_AS. CPU observer import/boundary checks and pinnedRuff0.9.9 check/format passed; selected JSON evidence retains numerical comparisons and zero-error observer replay identity.

## Runtime decision and next verification

Use the existing eager FA4 route, FP32 RoPE at originaltheta10000, FP32 heads andTP2/CP1. The new recipe is `examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-7n8g-megatron-tp2cp1-pack-single-controller-fullvocab.yaml`. This is within delegated parallelism and does not relax numerical acceptance. Compiled mode remains a documented failing path.

Three final workflow checks passed actual MasterConfig/single-controller validation and personal submission dry-runs:24GPU selfteacher actual training/scorer/refit,6GPU one-update65536-token training, and56GPU five-distinct-teacher native save/resume. Resolved configs and logs are under `/data/ycfeng/tmp/nemo-mopd-20261112/final-{self-observed,long-training,five-teacher-resume}-*`. Exact validation script: `validate_final_runtime_configs.py`. GPU execution is next; formal200-update run has not been submitted. NativeForwardWorker is limited to the24GPU diagnostic and must not enter formal training.
