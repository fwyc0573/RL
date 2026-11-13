## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Record the native FP16 scaler reproduction, live floor configuration, successful-update accounting, and latest-only inventory. |

# FP16 scaler and successful-update validation

The formal MOPD run is incomplete: ten successful optimizer updates have been observed through controller step 28. The isolated native loss/scaler regression passes after the recipe correction. The original-model overflow operator and recovery after the next real backoff remain unverified.

## Scenario and environment

The sole active formal writer is `exp-1114-051900-202756`, UID `ae42005f-d314-4c86-bb10-c6f7bc385b92`, with 72 H800 GPUs on codesign. It uses the initial R1-Distill-Qwen-1.5B student, the original math/code/stem/logic/if teachers, the full 151936 vocabulary, 256 fresh trajectories per attempted update, and the 65536-token sequence ceiling. Student TP2/PP4/CP1 and eight TQ units remain unchanged. The matching zero-GPU reader is `exp-1114-051922-692747`, UID `a7f3b178-aaa9-41b4-84c6-3e33576307c2`.

The CPU native checks use the existing Megatron actor environment: Python 3.13.14, Torch 2.13.0+cu130, UV 0.13.7. Platform operations use Python 3.10.6 and brainpp 2.8.6. Lint uses an existing Gym server environment; no packages were installed. Existing worker image identity is `hub.i.basemind.com/stepmind/megatron-step@sha256:d29740f438d49f543989925ccee2e1e0c6a739f2254c9acc7430363a570a6177`, with source `100.96.128.195:/data/ycfeng:/data/ycfeng`. Selected identity/runtime evidence predates these focused CPU checks and remains valid for the same live allocation.

## Native overflow regression

The split training path backpropagates token sums with a placeholder normalizer of one, then normalizes accumulated gradients before clipping and the optimizer step. This permits an intermediate FP16 gradient overflow even when the reported normalized loss and expected normalized gradient are finite. The old native `min_loss_scale=1` prevented native backoff below one. The recipe now supplies the existing optimizer field through existing resolvers:

```yaml
min_loss_scale: ${div:1,${mul:${policy.train_global_batch_size},${policy.max_total_sequence_length}}}
```

For the approved 256/65536 configuration this resolves to `5.960464477539063e-8`. Native backoff still chooses the actual scale. The change adds no fixed multiplier to the objective or gradient.

The manual regression invokes the native full-RKL loss with two vocabulary entries, a shared FP16 boundary, and 65536 tokens on CPU. Distributed collectives and CUDA scalar placement are redirected only in this isolated CPU fixture. This is not an original-model or full-151936-vocabulary replay.

| Check | Expected / reference | Observed | Result |
| --- | --- | --- | --- |
| Normalize before backward | Finite loss and normalized gradient | Loss 2.309328317642212; gradient 1.5 | PASS reference |
| Sum before backward, floor 1 | Reproduce inability to recover | Sum loss 151344.140625 is finite; gradient Inf for all three attempts; scale stays 1 | Effective RED |
| Sum before backward, corrected floor | Native backoff recovers the reference gradient | First attempt Inf at 1; subsequent attempts finite at 0.5, normalized gradient 1.5 | PASS isolated regression |
| Corrected versus reference gradient | Absolute error <0.001 | Absolute error 0; relative error 0 | PASS |
| Eight real step23 scaler states restored natively | Preserve saved scale and configured floor; backoff can cross 1 | Every saved scale [1.0] restores; every injected found_inf produces 0.5 | PASS native metadata restore |
| Live configuration on eight student ranks | Change only the floor between train steps | All eight native floors changed 1 → 5.960464477539063e-8; current scale remained 1; no open train step or checkpoint writes | PASS configuration |
| Original-model step25 overflow cause | Identify the actual failing backward operator/sample | No operator capture for weight_version24 is available | UNVERIFIED |
| Original-model recovery below scale1 | Observe subsequent native backoff and successful update | Steps26 and27 succeed at scale1; no real backoff below1 observed yet | UNVERIFIED |

The RED command was executed before the recipe change and exited with an assertion failure after writing its measured JSON. The GREEN command uses the corrected inherited recipe and native metadata:

```bash
export UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
export CUDA_HOME=/data/ycfeng/toolchain/cuda-13.0.2
export LD_LIBRARY_PATH=/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export TMPDIR=/data/ycfeng/tmp
export PYTHONPATH=/data/ycfeng/sota-engine/RL
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python tests/manual/mopd/check_fp16_sum_scaling.py \
  --config examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-4n8g-megatron-tp2pp4cp1-pack-single-controller-fullvocab-fp16.yaml \
  --native-state /data/ycfeng/tmp/nemo-mopd-20261112/formal-latest-step23-native-state-20261114.json \
  --output /data/ycfeng/tmp/nemo-mopd-20261112/fp16-native-sum-scaling-green-20261114.json
```

The RED invocation used the same regression/config paths, omitted `--native-state`, and wrote `fp16-native-sum-scaling-red-20261114.json`. GREEN emitted `MOPD_FP16_RAW_SUM_SCALER_PASS`, exit zero. Evidence: `evidence/fp16-native-sum-scaling-{red,green}-20261114.json`.

The live application reused the existing writer Ray allocation, submitting `/data/ycfeng/envs/nemo-mopd-py313/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/configure_live_native_scaler_floor_20261114.py` under submission ID `mopd-native-scaler-floor-20261114-0607`. Native `__ray_call__` inspected/configured all eight actors. The first call encountered stale local Ray method-signature metadata; deriving the signature from the installed native definition repaired the client invocation. No additional RJob, GPU allocation, checkpoint, or weight copy was created. Evidence: `evidence/native-scaler-floor-ray-submission-20261114.json` and `evidence/formal-native-resume-5-live-{scalers,scaler-floor}-20261114.json`.

## Actual optimizer updates and training health

The completed steps 18–23 segment contributes six successful updates. The current segment contributes four through step28. Successful-update flags are deduplicated by step and wall time; only flags equal to one count toward the required 200.

| Controller step | Optimizer success | Normalized loss | Pre-clipping grad norm | Scale | Response tokens | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 24 | 1 | 0.1714246124 | 0.9186215401 | 1 | 2,051,505 | PASS observed update |
| 25 | 0 | 0.1748526096 | Native optimizer returns None; no event | 1 | 2,084,198 | Overflow skip; original-model RCA open |
| 26 | 1 | 0.1756095737 | 0.9495472312 | 1 | 2,015,849 | PASS observed update |
| 27 | 1 | 0.1686267257 | 0.8094019294 | 1 | 1,967,529 | PASS observed update |
| 28 | 1 | 0.1593835205 | 0.7628392577 | 1 | 1,963,104 | PASS observed update |

Every attempted batch above contains 256 valid samples and LR1e-6. Norms are measured after valid-token normalization and before clipping. A preceding norm must not be assigned to step25. The step23 publication records Adam6 on all eight saved optimizer groups; later successes remain resident until the next native publication. If step25 remains the only new skip, step30 should record Adam12.

The revised temporary final continuation counts actual native successes and requires saved Adam counters to agree on all eight ranks. It accepts finite positive native scales below one, checks finite norms only on successful updates, and waits for the exact writer/reader completion and an error-free copy receipt. For a measured deficit it computes `latest_step + (200 - successful_updates)`, then reuses the existing native launcher, exact latest checkpoint, and reader. Final replay begins only after 200 successes.

The real step23 metadata reconciles to six successes. Ten additional named CPU cases pass: 199 successes plus a skip require horizon218; 200 successes at scale0.5 are accepted; missing successful norm, missing attempt, nonbinary flag, nonfinite loss, and native counter mismatch are rejected; an active writer cannot launch a duplicate; a terminal deficit creates exactly one inert writer and reader; reader synchronization waits for the exact receipt. Mocked submissions allocate zero GPUs and write zero checkpoints. This validates reachable control-flow boundaries, not a completed future GPU continuation.

```bash
# Existing actor environment and UV invocation from above.
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/check_successful_update_continuation_20261114.py
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python tests/manual/mopd/check_native_scheduler_resume.py \
  --config examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-4n8g-megatron-tp2pp4cp1-pack-single-controller-fullvocab-fp16.yaml \
  --native-state /data/ycfeng/tmp/nemo-mopd-20261112/formal-latest-step23-native-state-20261114.json \
  --train-iters 218
```

Native scheduler GREEN restores 5888 consumed samples and advances to 6144, preserving original 51200 LR/WD horizons and constant LR1e-6/WD0.01. Evidence: `evidence/native-successful-update-continuation-check-20261114.json`. Final waiter is session 5548 / local PID256195; it replaces session 40518. Training44154, reader1762, and monitor48778 remain alive. These are tool session IDs, not process IDs.

## Step28 live full-loss and parity observation

At November14,2026,06:21:41 HKT, step28 completes successfully. Formal count is ten successes over eleven measured attempts fromsteps18–28, with the single step25 skip preserved. The full-RKL mean equals the normalized loss exactly:0.15938352048397064. Its minimum−2.2394442567019723e−7 is within the existing negative-roundoff tolerance; its maximum28.00061798095703 is finite and is an observed divergence tail. These bounds are not a monotonic-loss acceptance requirement.

All256 samples have valid masks and weight_version27, with group coverage math53/code46/stem54/logic47/if56. Scorer/vLLM response-token MAE is0.0012310227258951327, signed mean−8.395944826752386e−6, p990.01002645492553711 and max0.5445899069309235. Every group mean is below0.02; range0.0011000982 (math) to0.0023026204 (if). All differences are finite and mask/version issues arezero. This is independent native-scorer/generation evidence after refit; actual training-forward parity remains pending. JSON evidence:`evidence/formal-native-resume-5-step28-{metrics,coverage,parity}-20261114.json`. The existing reader, inline local JSON/count checks, and current platform monitor were reused; no new GPU job, checkpoint, or weight copy was created.

## Scorer/vLLM discrepancy tail

Step26 sampled scorer/vLLM MAE is 0.0012728443 with max 0.2521480; all differences are finite and mask/version issue count is zero. This passes the sampled mean criterion but does not establish final actual-training parity.

Step25's largest discrepancy is in if sample `a4b734e9-b73c-4893-b8b6-112a42ea9b07_g0`, prompt_idx24, weight_version24. Total length is 20059, prompt length72, and response length19987. The prompt IDs and mask agree. Its MAE is 0.0063376682 and maximum 2.6184369; 11 response tokens exceed 1 nat and 195 exceed 0.1 nat, concentrated in a repetitive region after response position 14182. This is an observed tail, not proof of staleness or the overflow's cause. There is no saved version24 checkpoint, and the storage constraint forbids creating a diagnostic copy. Evidence: `evidence/step25-if-largest-parity-position-20261114.json`. The final native replay and continuing live summaries remain the authorized route to settle current-weight parity.

## Latest-only retention and verification limits

At November 14, 2026, 06:16:58 HKT, a bounded personal read-only inventory of every current-task `*/exp-*/checkpoints` directory finds exactly one complete checkpoint, step23, with 31 files, 8 tensor shards and 24,884,408,489 bytes. No historical or staged checkpoint remains. Exact retained path:

```text
/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-five-teacher-fp16-resume/exp-1113-174329-659007/checkpoints/step_23
```

The inline inventory uses `ssh -o BatchMode=yes -o ConnectTimeout=10 shai-ycfeng 'timeout 50s python3 -'`, asserts personal identity, enumerates only the exact personal task prefix, and reads `latest_checkpoint_status.json` plus file sizes. Exit zero; no cloud writes, extra checkpoints, or deletions were needed. Receipt: `evidence/current-task-native-checkpoint-inventory-0618-20261114.json`. Physical OSS quota reclaim remains unverified; logical directory deletion is not evidence of backend reclamation.

Lint repeated before the scoped commit with the existing Gym tool environment:

```bash
export UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-gym-servers/responses_api_models/vllm_model/.venv
export TMPDIR=/data/ycfeng/tmp
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync ruff check tests/manual/mopd/check_fp16_sum_scaling.py
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync ruff format --check tests/manual/mopd/check_fp16_sum_scaling.py
```

Both exit zero: all checks passed and one file already formatted. The task remains incomplete until 200 actual updates, final full-parameter/optimizer finiteness, native three-way parity, anomaly reconciliation, and terminal platform/business evidence have been verified. No English completion archive is written at this stage.
