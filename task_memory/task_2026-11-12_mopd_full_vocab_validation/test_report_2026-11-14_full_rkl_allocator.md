## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Diagnose the formal step31 CUDA allocation failure and verify an allocator configuration with native full-RKL backward. |

# Formal MOPD allocation failure and recovery

Status: the allocation-pattern reproduction is PASS after the recipe correction. The original formal writer failed at 12/200 successful updates. Real-model continuation and final acceptance remain pending.

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
| Real-model next update | Continue from step30 with finite loss/norm | Pending | UNVERIFIED |
| Full acceptance |200successful updates and final three-way parity |12saved successes | INCOMPLETE |

Exact CPU command:

```bash
export TMPDIR=/data/ycfeng/tmp
export UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313
export PYTHONPATH=/data/ycfeng/sota-engine/RL
export LD_LIBRARY_PATH=/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python tests/manual/mopd/check_full_rkl_allocator.py --device cpu --sequence-length 8 --vocab-size 32 --chunk-size 2 --fragment-pairs 0 --half-block-bytes 0 --free-bytes 0 --output /data/ycfeng/tmp/nemo-mopd-20261112/allocator-cpu-check-20261114.json
```

Each GPU subprocess used the same command with`--device cuda --sequence-length16384 --vocab-size75968 --chunk-size128 --fragment-pairs8 --half-block-bytes3221225472 --free-bytes2147483648`, differing only in`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False/True`. Full shell tokenization is retained in the request. Cloud receipts:`/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/allocator-check/exp-1114-115442-387792/expandable-{False,True}.json`.

## Correction and limits

The existing FP16 TP2/PP4 recipe now sets`policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The native RKL,FP32logits,chunk size,student/five teachers,151936vocabulary,256fresh trajectories and65536ceiling are unchanged. No exception-suppressing empty-cache patch is enabled. The resumed real workload must still verify loading,refit,continued updates,memory behavior and final actual-training parity. The controlled allocator result supports the cause-specific configuration; it does not prove the original workload is repaired before continuation runs.

At 2026-11-14T04:01:09.075032+00:00, the replacement writer is Running with all nine replicas Running and restart counts zero. This confirms platform startup, not a successful resumed optimizer update. Exact runtime evidence: `evidence/allocator-terminal-resume6-runtime-20261114.json`.
