## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record two completed native forward controls and the FP32 teacher projection defect. |

# Actual native forward precision controls

Status: actual trainer/vLLM overall mean PASS for both controls; FP32 passes every group mean. BF16 same-forward teacher reconstruction is exact. FP32 reconstruction FAILS its full-distribution contract despite passing the sampled-token mean. Repair and GPU revalidation remain pending.

Both jobs ran the specified initial student with a frozen copy as teacher, TP2/CP4, 3×8 H800/codesign, personal creator i-fengyicheng, local NFS100.96.128.195:/data/ycfeng:/data/ycfeng. These are numerical controls rather than distinct-teacher production runs. Native finished.json reports success for each job; platform Succeeded was observed at10:22:58Z. Launcher83836/44927 and monitor77515 exited0. Native TB records5/5 successful optimizer updates, skipped_step0 throughout,32 valid sequences/4096 response tokens per update.

| Control | Job | Training/vLLM MAE | p95 | p99 | Max | IF-group MAE | Loss range | Grad norm range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | exp-1112-181156-140223 |0.01458137552|0.08354603052|0.12300251007|0.62267309427|0.02203990|0.000233609–0.000566535|0.0348144–0.0883716|
| Paired FP32 | exp-1112-181232-890845 |0.00989524160|0.04650037289|0.09610750556|0.44635438919|0.01687489|0.000264274–0.000407942|0.0365815–0.1093646|

All norms are finite and below configured clipping1.0. Actual training minus independent native scorer MAE is2.82e-10 (BF16) /3.19e-10 (FP32), maximum2.38e-7 /3.57e-7. Thus an independent scorer covers this actual native training forward numerically for the observed workload. No objective, tolerance, token filter or parameter scale was changed. Full vocabulary151936,0dummy tokens is established by native initialization.

Same-forward teacher observation covers160calls and all8TP/CP ranks (1280shard records) per control. BF16 logit error and sampled LP error are0. FP32 direct output is float32, weight storage is bfloat16, but native reconstruction torch.matmul uses the weight storage dtype and rounds its output back to bfloat16. Across every vocabulary element the observed mean absolute logit error is0.00253643327, max0.125; sampled response LP MAE0.00945954095/max0.11890888214. This deterministic precision mismatch needs repair even though its response-token mean is below0.02.

The FP32 head reduces training/vLLM MAE and every group mean; residual tails remain. Initial same-weight teacher/scorer MAE is0.00663062(BF16)/0.00430544(FP32), consistent with a forward-shape or attention difference beyond output rounding. A separate shape-controlled comparison is needed to attribute that residual; the post-update teacher is frozen and is no longer same-weight with the student. Nonzero self-loss/gradient cannot alone prove an optimizer defect.

Exact commands: recorded submit args native-{bf16,fp32}-observed-submit-args.json in /data/ycfeng/tmp/nemo-mopd-20261112, through tests/manual/mopd/submit_on_rjob.py and run_on_rjob.py; observer entrypoint tests/manual/mopd/native_forward_entrypoint.py. Native actual-logit observations are detached inside prepare_loss_input and teacher postprocessing. Run directories are /mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/native-{bf16,fp32}-observed/<job>/.

Analysis: uv run --no-project --python /data/ycfeng/envs/nemo-mopd-py313/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/analyze_observed_native_control.py --run-dir <downloaded-run> --identities /data/ycfeng/tmp/nemo-mopd-20261112/native-5-input-identities.jsonl --output <report> --mean-ceiling0.02 --teacher-ranks8. Reports contain exact token IDs/mask/value/version coverage checks, signed mean, absolute percentiles, per-step/group breakdowns and8-rank coverage. A first analysis assertion incorrectly equated valid length with padded array length; corrected to validate the complete array and slice only native input_lengths. Both analyses then exit0 with160/160samples and20480response tokens.

Evidence collection retained3.65MB/4.23MB archives via bounded read-only SSH. Collection returned2 only because a requested runtime-rank-0.json does not exist; logs, forward observations, all5dumps, TB and finished.json were recovered. Native runtime identity remains in the earlier native-runtime-versions.json and worker receipts. No missing runtime filename is counted as a training failure.

Limits: controls use one unique selfteacher; no distinct-five-teacher or parameter-delta acceptance. Native save/resume and200successful updates remain outstanding. FP32 teacher full-distribution discrepancy remains FAIL until fixed and measured again. Raw files remain under tasktmp/native-{bf16,fp32}-observed-analysis-inputs; selected reports, metrics and terminal receipts are evidence/native-{bf16,fp32}-observed-{parity,training-metrics,terminal}.json.


## Root-cause repair and CPU verification

Worker stores teacher output dtype from its actual policy config and retains it across eviction/reload. Checkpoint LM heads materialize in FP32 when that teacher enables fp32_lm_head; other teachers retain their configured precision. Mixed teacher rows compute independently in each head's dtype, then cast to a promoted shared output buffer.

Loader regression: FAIL before repair (expected FP32, reconstructed BF16), PASS after repair for both output precisions and none/evict lifecycles. All nine affected worker tests PASS (23.00s); all 59 tests/unit/algorithms/test_opd_full.py checks PASS (11.78s). Ruff0.9.9 check/format PASS; actual mounted MCore observer import and paired-head native config validation PASS. GPU direct/reconstructed distribution/loss/gradient validation is pending in exp-1112-183916-289473.

Pure loader command uses mounted actor interpreter through uv run --no-sync python -m pytest --noconftest -m mcore tests/unit/models/policy/test_megatron_worker.py -k 'teacher_head_reconstruction_preserves_logit_precision or evict_reloads_every_teacher_once_per_release or off_last_stage_rank_reenters_the_reload_in_lockstep or offload_parks_every_teacher_head_not_just_the_first or second_teacher_with_another_hidden_size_is_rejected_at_load or off_last_stage_joins_the_load_without_requesting_a_shard'. Environment: UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker, LD_LIBRARY_PATH includes cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat, CUDA_HOME=/data/ycfeng/toolchain/cuda-13.0.2, PYTHONPATH=/data/ycfeng/sota-engine/RL, MLFLOW_DISABLE_AGENT_HINT=1. Full-loss suite uses mounted base Python3.13 through uv run with --noconftest. Raw command errors and final logs remain in tasktmp; no numerical PASS is assigned to collection/import failures.


## Corrected native GPU control

exp-1112-183916-289473 Succeeded; native controller completed5steps/trainer_version5. Exact corrected submission is PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_native_fp32_corrected.py; args are native-fp32-corrected-submit-args.json. Same24H800, TP2/CP4, initial student/selfteacher and pairedFP32 heads. CPU imports/config/dry-run passed before submission.

All160actual training forwards and20,480response tokens match native masks, scorer/generation values and versions. Trainer/vLLM MAE0.00991450586, p950.0470875740, p990.0919413096, max0.326207399; every group mean≤0.02 (code0.01164027, if0.01648130, stem0.00741723, math0.00622298, logic0.00825258). Actual training/scorer max2.682209e-7. Teacher direct/reconstructed sampled LP MAE3.149561e-6, max5.102158e-5. Coverage and sampled mean checks PASS without filtering.

All1280TP/CP records cover160teacher calls and10,309,769,216vocabulary elements. Full-logit MAE4.125251e-6, max1.144409e-4; full-logprob MAE3.225163e-5, max1.449585e-4. Native reverse-KL per-position loss error max5.573191e-7≤1e-6. Remaining errors are consistent with the observed nativeBF16-input/FP32-output GEMM versus FP32 reconstruction accumulation, after removing BF16-output rounding. Bitwise equality is not established.

The detached observer sums divergence over all positions, including prompts, for its raw probe. Its unnormalized logit-gradient error max1.668435e-5 exceeds the existing1e-5 absolute assertion applied directly to that sum; raw verdict remains in the evidence. This is not the native optimizer objective: ClippedPGLossFn._opd_full_call masks response tokens and finish_train_step scales accumulated gradients by1/global_valid_toks (worker:2381–2391). The recipe inherits token_level_loss=true; TB confirms4096valid tokens on every update. Because fixed-teacher KL is independent across positions, applying the response mask cannot increase a coordinate's gradient error; native normalization gives an absolute logit-gradient error upper bound1.668435e-5/4096=4.073327e-9≤1e-5. This PASS is a mathematical bound derived from actual observations and inspected native normalization, not a separately observed parameter-gradient equality. Relative error is undefined for all1280reference-zero gradients; it is recorded as null. No training scaling, mask or tolerance changed.

Native successful updates5/5, skipped0, loss8.524528e-5–2.106073e-4, grad norm0.02513888–0.06578441; all finite and below clipping1.0. Distinct-teacher effects and long-context parity remain separate outstanding criteria.

Collection: read-only SSH Python tar stream of finished.json/logs/exp_001/logs/forward_observations,13,854,720bytes, exit0; extracted into tasktmp/native-fp32-corrected-analysis-inputs. Same analyzer command as prior controls with --run-dir native-fp32-corrected-analysis-inputs --output native-fp32-corrected-parity.json --mean-ceiling0.02 --teacher-ranks8, exit0. TB extraction uses tasktmp/read_corrected_scalars.py via uv run --no-sync with the mounted base environment. Selected evidence is native-fp32-corrected-{parity,distribution,training-metrics,terminal}.json. Large artifacts:/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/native-fp32-corrected/exp-1112-183916-289473/.
