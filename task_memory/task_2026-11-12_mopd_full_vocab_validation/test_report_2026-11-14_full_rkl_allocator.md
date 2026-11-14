## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Verify the first successful full-model update after the allocator correction, with13/200 formal successes. |
| 2026-11-14 | Record resumed startup, latest-only metadata recheck and the non-blocking Ray inspection timeout. |
| 2026-11-14 | Diagnose the formal step31 CUDA allocation failure and verify an allocator configuration with native full-RKL backward. |

# Formal MOPD allocation failure and recovery

Status: the native allocation reproduction and the first real-model resumed update both pass. Formal progress is13/200 afterstep31;12successes remain saved in the sole step30 checkpoint. The72-H800 writer continues. Sustained training, final200-success acceptance and actual-training three-way parity remain pending.

## Incident and recovery state

Writer `exp-1114-051900-202756`, UID `ae42005f-d314-4c86-bb10-c6f7bc385b92`, failed at November 14, 2026 06:51:03 HKT. At06:50:14, native `ChunkedDistributedReverseKLToFixedLogits.backward` reached `model_utils.py:2677`, where `torch.empty_like(student_vocab_parallel_logits)` requested4.59GiB. Rank6/pid1746 on replica`exp-1114-051900-202756-e4da3b7f`, UID`59066c44-ee24-4e3d-acd7-1636969c792a`, node`gpu-h800-0042.host.platform.shaipower.com`, failed with45.71GiBallocated/23.40GiBreserved-unused and2.29GiBdriverfree. Rank7 reports the same failure. This is a CUDA tensor-allocation exception; the stack does not identify an FP16 arithmetic overflow.

The exact replica was subsequently deleted at06:51:03. The recorded controller stop actor is`system:serviceaccount:kubebrain:rjob-controller`; the local launcher Stop appears at06:51:14.667, after failure. Incident-bound historical events contain no positive preemption/eviction/OOMKilled evidence. Both all-operation audit queries timed out after45s; initiating manual-stop fields remain UNKNOWN. No existing DeepScan flow was returned. These query limits do not exclude an unobserved platform event; the positive initiating business failure is the CUDA exception. Selected evidence is in `evidence/formal-resume5-oom-*` and `evidence/formal-native-resume-5-oom-traceback-20261114.txt`.

Fresh read-only inspection of the native root confirms only`step_30`,31files,8shards,24,884,408,499bytes. All8Adam states record12 successes,scalers1,controller/sampler30,consumed_samples7680 and8TQunits. Step23 is absent. The inspection read551,606bytes of metadata and scalar optimizer state,0model/Adam tensor payloads. Native scheduler restores7680 and advances7936 while preserving51200decay horizons,LR1e-6 andWD0.01/0.0. Full final-tensor finiteness and physical OSS quota reclamation are unverified.

## Hypotheses and discriminating check

1. Fragmented allocator segments can prevent a4.59GiB request despite23.40GiBunused reserved memory. Prediction: a native backward with sufficient aggregate free space but stranded blocks fails with ordinary segments and succeeds with expandable segments.
2. The live model/activation set exceeds capacity. Prediction: eliminating stranded blocks is insufficient for the failing allocation. This remains a possible separate future failure; the synthetic experiment cannot establish all real-model peaks.
3. Additional CUDA contexts consume the margin. Prediction: the same allocation failure needs those contexts. The synthetic failure reproduces in a single process, showing they are unnecessary for this allocation pattern. It does not establish they had no contribution in the original worker.

The feedback command is `tests/manual/mopd/check_full_rkl_allocator.py`. It invokes the actual native full-RKL autograd function, constructs eight retained/unused3GiB pairs, then leaves approximately2GiBdriverfree. Native backward must allocate a4.64GiBgradient. This is a controlled synthetic allocation history, not replay of the original model's allocator trace.

## Verification

Runtime:Python3.13.14,Torch2.13.0+cu130,NVIDIAH800. Job`exp-1114-115442-387792`,UID`e2fac3f9-4c76-469a-ae58-b5a1afd3d064`,creator`i-fengyicheng`,codesign,1GPU,current-hostNFS`100.96.128.195:/data/ycfeng:/data/ycfeng`,companyimage`megatron-step:test-optimus-3.24.0-stepccl-0.0.5.post5-vllm-0.11.0.post37-20260817-2733176`. Personal launcher reports Succeeded at11:56:04HKT. Commands and resource request are in`evidence/rkl-allocator-request-20261114.json`; no model/checkpoint files were created.

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| CPU native RKL,8tokens/32vocab | Analytic gradient agreement | Maximum absolute error0 | PASS |
| CUDA ordinary segments,16384tokens/75968localvocab | Fragmented4.64GiB allocation fails |24GiBinactive splits;52.52GiBallocated;OOM at2GiBfree | Expected RED |
| CUDA expandable segments,same inputs | Full backward completes with correct finite gradient | Inactive splits0;maximum absolute gradient error1.8189894035458565e-12 | PASS |
| Native empty_cache after expandable backward | Completes without exception or corruption | Completed;normal process exit | PASS |
| Step30 native counters/retention |12successes and only latest checkpoint |8Adamstates agree;step23absent | PASS |
| Native scheduler horizon218 | Preserve checkpoint schedule |7680→7936,original51200 horizons | PASS |
| Real-model next update | Continue from step30 with finite loss/norm | Step31 success1, loss0.1526285261, norm0.6875529289 | PASS for this update |
| Full acceptance |200successful updates and final three-way parity |13observed successes,12saved | INCOMPLETE |

Exact CPU command:

```bash
export TMPDIR=/data/ycfeng/tmp
export UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313
export PYTHONPATH=/data/ycfeng/sota-engine/RL
export LD_LIBRARY_PATH=/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python tests/manual/mopd/check_full_rkl_allocator.py --device cpu --sequence-length 8 --vocab-size 32 --chunk-size 2 --fragment-pairs 0 --half-block-bytes 0 --free-bytes 0 --output /data/ycfeng/tmp/nemo-mopd-20261112/allocator-cpu-check-20261114.json
```

Each GPU subprocess used the same command with`--device cuda --sequence-length 16384 --vocab-size 75968 --chunk-size 128 --fragment-pairs 8 --half-block-bytes 3221225472 --free-bytes 2147483648`, differing only in`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False/True`. Full shell tokenization is retained in the request. Cloud receipts:`/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/allocator-check/exp-1114-115442-387792/expandable-{False,True}.json`.

## Correction and limits

The existing FP16 TP2/PP4 recipe now sets`policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The native RKL,FP32logits,chunk size,student/five teachers,151936vocabulary,256fresh trajectories and65536ceiling are unchanged. No exception-suppressing empty-cache patch is enabled. The resumed real workload must still verify loading,refit,continued updates,memory behavior and final actual-training parity. The controlled allocator result supports the cause-specific configuration; it does not prove the original workload is repaired before continuation runs.

At 2026-11-14T04:01:09.075032+00:00, the replacement writer is Running with all nine replicas Running and restart counts zero. This confirms platform startup, not a successful resumed optimizer update. Exact runtime evidence: `evidence/allocator-terminal-resume6-runtime-20261114.json`.

## Previously completed update health

The existing reader confirms the two last completed updates before the OOM:

| Controller step | Loss / full-RKL mean | Pre-clipping gradient norm | Valid response tokens | Scorer/vLLM MAE |
| --- | --- | --- | --- | --- |
| 29 | 0.1514309049 | 0.7063075304 | 1,877,609 | 0.0012270625 |
| 30 | 0.1622027457 | 0.7467541099 | 1,907,184 | 0.0012516937 |

Each has 256 valid samples, all five source groups, native scale1, and zero mask/version issues. Every sample uses the expected preceding weight version. Loss equals the emitted full-vocabulary RKL mean. All compared logprob differences are finite. Maximum absolute logprob differences are 0.6848463 and 0.3375127 respectively; only the agreed mean criterion is a PASS. These scorer/vLLM comparisons do not replace the final actual-training comparison. Evidence: `evidence/formal-native-resume-5-step29-step30-health-20261114.json`.

## Resumed startup and recovery, November 14, 2026 12:13 HKT

The current writer and reader remain Running; all four local launcher/monitor sessions are alive. The native trace reports weight_version30 synchronization and completed restore. All eight ALIVE student actors carry `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in Ray's runtime environment. Completed rollout and teacher-inference log entries cover all five source groups. These checks pass startup/configuration propagation and restoration, while the next successful update remains UNVERIFIED.

The existing native-state reader was rerun on the CPU host:

```bash
env TMPDIR=/data/ycfeng/tmp UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker PYTHONPATH=/data/ycfeng/sota-engine/RL OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat timeout 60s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/inspect_latest_formal_native_state_20261114.py
```

Exit0: only step30 remains, 31files/24,884,408,499bytes, all eight Adam counters12. Bytes read551,606; model/Adam tensor payload reads0. No checkpoint, tensor copy or HF export was produced.

Optional local `inspect_live_resume6_allocator_20261114.py` timed out in `ray.init` after60s/exit124. Actual allocator counters/scalers were not returned. This auxiliary failure did not affect training; no invasive retry or training change was made. The exact-job Dashboard request `/api/v0/actors?limit=1000&detail=true` at12:11:57HKT supplies the eight runtime-environment observations instead. Selected evidence: `evidence/formal-native-resume-6-startup-verified-20261114.json`.

## First successful allocator-corrected update

Step31 completed on November14,2026 at12:18:49HKT. Native `optimizer_update_successful=1`, loss_scale1, and no overflow skip were recorded. The existing reader and monitor supplied the actual TensorBoard events and a complete response-token comparison. All fields below are from step31, rather than each tag's most recent event from an earlier step.

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Actual successful updates | Native flag1; add one to12saved successes |13/200 total; current segment1/1; historical step25skip retained | PASS for progress |
| Loss / full-vocabulary RKL | Finite normalized values, equal reported means | Both0.15262852609157562; per-token min−3.09290499e−7,max26.3633976 | PASS; small negative value is within roundoff tolerance |
| Gradient norm | Finite after token normalization, before clipping |0.6875529289245605; clip_grad1.0; scale1 | PASS |
| Metrics | No unexplained nonfinite values |281step metrics all finite; filtered/masked sample counts0 | PASS |
| Samples and teacher routes |256valid samples, allfivegroups, matching weights |code60/math51/stem51/if49/logic45; allweight_version30; teacher_alias/model_unique5 | PASS |
| Response-token coverage | Positive valid objective tokens |2,031,837; maximum total sample length21,831 within65,536ceiling | PASS for this batch |
| Independent scorer/vLLM | Finite same-token MAE≤0.02; report tails | MAE0.0012237102789000166; signed mean−3.59357234e−6;p500.0002995618;p950.0053094149;p990.0099352932;max0.2297739983 | PASS for mean criterion |
| Per-group scorer/vLLM | Allfivegroup means≤0.02; no mask/version issues | math0.0010652592,logic0.0011374374,stem0.0011577894,code0.0013667057,if0.0023398088; issues0 | PASS |
| Current storage | Keep only latest complete native checkpoint |step30/Adam12 remains current; step31 is resident; no extra checkpoint or export | PASS under latest-only policy |

This first resumed batch took638.1654seconds in the native step timer: exposed generation517.7575seconds and policy training85.4679seconds. These are measurements of one resumed batch, not a throughput guarantee. No new GPU run or diagnostic weights were needed for verification.

Exact evidence: `evidence/formal-native-resume-6-step31-health-20261114.json`, including all281exact-step metric events and all256sample summaries. The existing `read_formal_native_resume_6_metrics_20261114.py` and `collect_formal_native_resume_6_mounted_20261114.py` were used; both are under `/data/ycfeng/tmp/nemo-mopd-20261112/`. Inline CPU checks select `step==31`, reconcile native flags fromsegments3+5+6, and assert256samples, full-RKL/loss equality, finite metrics, five routes,weight_version30,zeroissues and the unchanged0.02mean limit. All assertions passed.

Limits: a successful real-model update supports the allocator repair, but one iteration cannot establish long-run allocator stability. This batch does not exercise the65,536maximum; the earlier dedicated native long-context check supplies that evidence. The comparison is independent scorer versus generation; the required final actual-training/scorer/vLLM comparison is still pending. The native checkpoint count remains12until the next ordinary save; no additional checkpoint is created for the thirteenth success.
