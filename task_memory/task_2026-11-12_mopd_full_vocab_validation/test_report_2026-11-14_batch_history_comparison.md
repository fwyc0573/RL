## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Verify dispatch-batch versus completion-FIFO semantics, reconstruct two source-task runs, and quantify the matched performance gap; record the platform-initiated failure and durable recovery boundary. |

# MOPD batch scheduling and historical performance

The current workflow is markedly slower than the source task. Aligning successful update ordinals 14–28 yields **627.8369s versus 54.7420s per update (11.469 times)**, while the current batch generates **7.31% fewer response tokens**. Approximately **93.34% of the wall-time gap appears in consumer-visible ready-data waiting**. These observations identify where the difference manifests; they do not isolate graph, TP, teacher payload, or sampler contributions.

The formal writer failed on November 14, 2026 at **15:03:59 HKT / 07:03:59 UTC**. The initiating event closes a teacher replica under the platform GPU-utilization clean policy at 15:03:17 HKT. Formal recoverable progress is **22/200**, in the sole native step40/Adam22 checkpoint. Twenty-eight successes were observed through controller step46; six after step40 were unsaved and must be recomputed. This investigation is complete; the original 200-successful-update task is incomplete.

## 1. What one rollout batch means in the current configuration

The actual resolved configuration is `num_prompts_per_step=256`, `num_generations_per_prompt=1`, `train_global_batch_size=256`, `min_groups_for_streaming_train=256`, `max_inflight_prompts=256`, `max_buffered_rollouts=512`, and `sampler=in_order` with `max_lookahead_versions=0`. Each student update consumes **one logical dispatch batch containing 256 prompt groups / 256 responses**, and calls one optimizer step. Packing and microbatch/gradient accumulation do not add rollout batches. Twenty-four TP2 engines share that logical batch; there are not twenty-four independent 256-response batches.

The controller submits individual asynchronous requests. Round-robin placement initially gives each of 24 engines about ten or eleven requests, which vLLM schedules dynamically. A group becomes ready only after generation, data-plane submission, and teacher enrichment. Thus `exposed_generation` measures waiting for complete training inputs, not exclusively GPU decode.

`InOrderSampler.select` chooses ready groups whose `target_step == current_train_weight`; `_finalize_selection` returns no batch until its minimum group count is met. With the threshold at256, the controller cannot start student scoring/training until all256 eligible groups are ready. Completion order does not choose this update's membership. In an ordinary fully valid batch, a late response holds the entire update. The already-verified sample-failure/drop policy remains separate from this normal-case barrier.

Zero version lookahead prevents admission of the next dispatch batch until trainer version advances. Dedicated rollout GPUs therefore do not imply overlap between next-update generation and current-update training in this configuration. The512 rollout-buffer capacity does not activate such overlap; it is capacity, not a prefetch instruction.

Lowering the streaming threshold can overlap scoring/backward of ready chunks with remaining same-version generation. The same update still ultimately requires256 groups. Enabling a cross-version ready-first sampler is a different learning policy: the current validator requires importance-sampling correction for `ready_first`, and the current stamped-drop failure budgets are incompatible with that sampler. Do not claim that changing the sampler name alone preserves on-policy semantics or the prepared failure behavior.

Source references:

- `nemo_rl/algorithms/async_utils/staleness_sampler.py:323` (minimum selection), `:476` (admission gate), `:638` / `:681` (in-order target-step selection).
- `nemo_rl/algorithms/single_controller.py:2643` / `:2677` (wait and streaming minimum), `:3068` / `:3275` (accumulation and train-step finish), `:3404` (trainer version advance).
- `nemo_rl/algorithms/opd.py:577` (teacher enrichment before ready).
- `nemo_rl/algorithms/single_controller_utils/config.py:1248` / `:1830` / `:1859` (failure-budget compatibility, optimizer-step contract and importance correction).
- `nemo_rl/models/generation/vllm/vllm_generation.py:1380` (individual asynchronous request route).

## 2. The StepTron FIFO workflow differs

The source checkout `/data/ycfeng/steptron/.worktrees/yc-mopd-learning-dynamics-s4-branch` retains the study records under `task_memory/task_2026-09-26_mopd_learning_dynamics/`. Valid generation callbacks append completed trajectories to `pre-train`. `FIFOTrainBatchPolicy` selects256 units from the queue head. Membership follows the completed queue, without an original dispatch-batch barrier; a still-running long response can be consumed by a later update.

Generation is continuously replenished while teacher/student consumer work proceeds, with dispatch paused during weight synchronization. Historical settings are `max_concurrent_rollouts=256`, `staleness_max_steps=5`, `pending_limit=1536`, and32 process workers. The1536 pending limit is an admission window; it does **not** mean six simultaneous256-request batches, because concurrency is still256.

This permits older-weight trajectories. The historical `policy_rollout_version_gap/mean=0` scalar does not prove zero staleness: `steptron/exp/rlhf.py:1727` fills0 when version metadata is missing. Actual historical per-sample weight provenance has not been fully reconstructed.

Source references: `steptron/core/trainers/fullyasync_rollout_runtime.py:1256`; `steptron/core/trainers/fullyasync_policies/train_batch/fifo.py:18`; `steptron/core/trainers/fullyasync_trainer.py:550`; `steptron/exp/rlhf.py:1690`.

## 3. Retained runs and measured costs

Both retained fresh-data runs use the original R1-Distill-Qwen-1.5B student and math/code/stem/logic/if teachers,256 responses/update, a65536-token ceiling, and200 successful updates. The source records' `Full` / `M-prime` labels describe fresh-data mode: their training feedback is sampled-token OPD/PPO, **not full-vocabulary KL**. They are useful workflow references with a different loss cost.

| Retained run | Updates | Total s/update | Ready wait s | Consumer s | Weight sync s | Response tokens/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| September `h800-mprime-v3-0928a` | 200 | 65.515 | 25.319 | 22.039 | 18.490 | 7352.53 |
| November `h800-rkl-full-1108b` | 200 | 53.440 | 4.675 | 23.160 | 26.518 | 7326.09 |

Historical timers have small scope/rounding residuals; these columns are not a perfectly disjoint decomposition. In particular a small ready wait means rollout work was already completed or overlapped, not that all256 trajectories generated in4.7s.

A fairer early-window comparison aligns **successful optimizer update ordinals**, rather than the current controller counter (which includes earlier failed attempts). Current controller steps32–46 correspond to successes14–28; November native iterations13–27 correspond to the same ordinals.

| Matched measurement | Current NeMo-RL | November StepTron |
| --- | ---: | ---: |
| Total step | 627.8369s | 54.7420s |
| Consumer-visible ready wait | 540.1195s | 5.1666s |
| Response tokens/batch | 1,851,415.7 | 1,997,369.9 |
| Current student training / historical whole consumer | 66.3933s | 23.1151s |
| Weight synchronization | 0.6010s | 26.9451s |

Current scoring adds6.7243s and other disjoint residual time14.5999s. The wait gap is534.9529s out of573.0950s total difference, **93.34%**. Full-vocabulary training adds real cost, but the observed extra student/consumer time cannot explain roughly535s of additional waiting.

Across15 current steps, Pearson correlation between ready wait and longest response is **0.90694**; between ready wait and batch response-token sum it is **0.09529**. This supports the long-response/barrier explanation. It is correlation, not a controlled isolation of sampler effects. Current steps32–38 are11.982 times slower than matched history;39–46 are11.027 times slower, so the gap persists across both windows.

November retained51,410 successful generation-call records: mean48.776s, p5046.007s, p95118.011s, p99164.112s, max521.318s. Current mean `rollout/total` is163.299s. Those timers have different boundaries, different inputs, weights, precision and inference engines; they are supporting evidence of a slower inference path, **not a3.35-times GPU decode benchmark**. Extra completed records beyond51,200 consumed responses are consistent with continuous prefetch; the count alone does not prove exactly which records were unused.

| Deployment / execution choice | Historical | Current failed writer |
| --- | --- | --- |
| Physical GPUs | 16: trainer/teachers8 + rollout8 | 72: trainer8 + rollout48 + teachers10 + unused6 |
| Student | TP2 / DP4 / PP1 / CP1 | TP2 / DP1 / PP4 / CP1 |
| Rollout engines | 8 engines, TP1 | 24 engines, TP2 |
| Precision | BF16 | FP16 with FP32 LM head / RoPE |
| Graphs | `enforce_eager=false` | `enforce_eager=true` |
| Token scheduling budget | 32768 | 2048 |
| Consumer selection | Completion FIFO, continuous prefetch | In-order target batch, zero lookahead,256-group barrier |
| Teachers | Trainer-GPU grouped packed forward,20 logical calls/update | Five dedicated TP2 teachers,256 per-response calls, hidden-state payload |

The2048/32768 token budget difference is mainly a prefill scheduling question; it cannot be interpreted as a16-times decode factor. Historical fast `Full` mode neither satisfies today's full-vocabulary criterion nor justifies copying its off-policy scheduling without correction/parity validation.

Exact historical cloud evidence:

- `/mnt/codesign-exp/ycfeng/steptron/task_2026-09-26_mopd_learning_dynamics/outputs/h800-rkl-full-1108b/tensorboard/r1distill1p5b_mopd_5teacher_training_rkl/log_0.txt`
- Same directory: `events.out.tfevents.1794153649.gpu-h800-0260.host.platform.shaipower.com.820.0`
- `/mnt/codesign-exp/ycfeng/steptron/task_2026-09-26_mopd_learning_dynamics/outputs/h800-rkl-full-1108b/metrics/sample_events/*.jsonl`
- Local source reports: `s2_fewshot_replay/test_report_2026-11-09_s2-06_completion.md` and `analysis/tensorboard/h800-mprime-v3-0928a.json` in the source study directory.

## 4. Failure and required recovery decision

The exact teacher replica `exp-1114-115901-508160-eccbc87e` on `gpu-h800-0535.host.platform.shaipower.com` has these historical events (UTC; add8h for HKT):

| UTC time on November14 | Event |
| --- | --- |
| 06:01:00 | GPU-utilization policy,1-hour reminder |
| 06:21:44 | GPU-utilization policy,2-hour mark-preemptible |
| 07:03:17 | GPU-utilization policy,3-hour close; pod delete |
| 07:03:31 | Ray reports missing heartbeats / unexpected teacher-node termination |
| 07:03:57 | Head and another wrapper exit1 after native redispatch exhaustion |
| 07:03:59 | Whole RJob Failed |

The replica's initiating `stopped-by` is **`system:serviceaccount:kubebrain:cleanpolicy`**, with `stopTimestamp=07:03:17Z`; the subsequent whole-job reason is task failure. Thus this is a directly evidenced platform clean-policy event. Subsequent vLLM SIGTERM is cleanup. No positive CUDA OOM evidence establishes OOM as the initiating cause. Historical Events are pagination-complete; the all-operation RJob audit endpoint failed its resource-name validation. The raw utilization window is unavailable, so the policy's exact measured average is not reconstructed.

Only native `step_40` remains:31 files,24,884,408,532bytes, eight optimizer shards withAdam22. Read-only inspection at15:22:31HKT read551,608metadata/scalar bytes and zero model/Adam tensor payloads. Six successes at41–46 were unsaved. Latest metrics at46 remain valid observations (loss/full-RKL0.1016600, normalized pre-clipping gradnorm0.4149944, scale1,256samples,1,911,766response tokens); they cannot be counted as durable resumed progress.

Do not submit the identical low-utilization deployment for another indefinite continuation. The shortest recovery path needs measured inference improvements and a role placement that avoids six unused teacher-node GPUs. Existing co-location guards still prohibit SingleController vLLM co-location and require dedicated teachers; a large backend migration is not the first fix.

Ranked falsifiable candidates for the next bounded comparison:

1. The fixed256-group barrier and absence of continuous prefetch expose long decode tails. Prediction: same-version streaming reduces exposed wait by overlapping ready chunks; it still must consume256groups and cannot eliminate the final tail.
2. Eager decode increases per-token host/launch cost. Prediction: FP16 CUDA graphs with `compilation_config.backend=eager`, with other settings fixed, reduce equal-token elapsed time and retain the0.02 parity ceiling. Earlier BF16 compiled failure does not settle this FP16 graph-only case.
3. TP2's Ray executor adds synchronization/host cost at small per-engine concurrency. Prediction: equal-workload TP1 versusTP2 improves request latency or GPU-normalized throughput, subject to65,536-token capacity and identical-token parity. TP1 also changes internal async-scheduling support, so the result is a combined execution-path comparison.
4. Per-response teacher enrichment and hidden-state I/O add overhead. Prediction: boundary timestamps show teacher completion or data submission delaying the last ready groups. Summed teacher inference time alone is not critical-path attribution.

Compare in this order using existing launch and replay entrypoints where possible. Any adopted change must preserve original assets, full151936vocabulary,256fresh responses/update,65536ceiling,200successful native Adam steps, and latest-only native checkpoint retention. Do not use fake GPU activity, cleanup exemptions, smaller workloads, or silently changed sampling semantics to make the original workload appear healthy.

## 5. Commands, verdicts and limits

Executed read-only reconstruction on local Python3.10.6:

```bash
python3 /data/ycfeng/tmp/nemo-mopd-20261112/analyze_history_buffers_20261114.py
```

Expected:200 historical update records each; positive tokens; matched successful ordinals; exact current events/samples; finite health metrics; correct five-group/version/mask coverage; overall/group scorer-vLLMMAE≤0.02. Actual **PASS**: both200-step records, matched15-step windows,281metrics/step (step40 has one additional native checkpoint timer), and256valid samples at each39–46. The observed performance gap remains **FAIL against historical workflow expectations**, not a newly imposed numerical acceptance threshold.

Latest-only checkpoint inspection uses `/data/ycfeng/tmp/nemo-mopd-20261112/inspect_latest_formal_native_state_20261114.py` in offline UV0.13.7 / Python3.13.14 / Torch2.13.0+cu130. Platform object reads and RCA use personal Python `RJobClient`, Python3.10.6 / brainpp2.8.6, after the established personal environment setup. Personal API confirms reader `exp-1114-115909-784580` Succeeded at15:04:24HKT; this proves evidence-reader completion, not training success. Existing launcher sessions75774(writer),89756(reader),40884(monitor),54975(final waiter) are terminal (exit1/0/0/1 respectively); the original final waiter has not been rebound to a replacement. Collector86819 is terminal exit1 because RJob audit failed; its direct initiating event reads succeeded.

Selected persistent evidence: `evidence/history-current-performance-20261114.json`, `evidence/formal-native-resume-6-step39-step46-health-20261114.json`, and `evidence/formal-resume6-clean-policy-failure-20261114.json`. Raw scoped source reads remain under `/data/ycfeng/tmp/nemo-mopd-20261112/`. No model payload was downloaded, no checkpoint/HFexport was created, no production optimization was applied, and no new GPU job was submitted for this comparison. Public web-tool reads/searches returned no results; local inspected source and the already retained primary-source report supply the technical evidence.

P's two follow-up questions are answered. E requires a revised, verified recovery configuration fromstep40/Adam22; F still requires final finite tensors, unsaved native-training/scorer/vLLM parity, final curves and English completion archive.

CPU recovery preparation also passed `preflight_formal_step40_graph_candidate_20261114.py` with the exact command and environment in `evidence/formal-step40-graph-candidate-cpu-preflight-20261114.json`. It retains all formal dimensions and native latest-only settings, changes only vLLM graph/eager-backend settings, and restores the existing step40 constant scheduler (10240→10496 scheduler samples on the next tick; Adam remains22 in checkpoint metadata). It reads zero checkpoint tensor payloads. This is configuration/scheduler preparation, not a measured graph speedup, GPU parity result, or corrected teacher placement. No candidate submitter was executed.
