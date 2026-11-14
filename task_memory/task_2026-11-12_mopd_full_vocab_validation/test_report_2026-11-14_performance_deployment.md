## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Reconcile live roles and steps32–38, verify current switches and co-location limits, and record unmeasured performance candidates. |

# MOPD deployment and performance investigation

The current job allocates72H800 across three GPU role categories. Waiting for complete rollout samples accounts for86.40% of the seven measured steps32–38; student training accounts for10.29%. The current vLLM SingleController workflow cannot become colocated through one flag. The strongest configuration candidate is CUDA graphs with Inductor disabled, but this FP16 combination has not yet passed numerical or performance validation.

The investigation leaves the existing writer and configuration running. It creates no checkpoint, HF export or GPU allocation. Original acceptance remains200 successful updates with the specified models, full151936 vocabulary,256 fresh trajectories and65536 sequence ceiling.

## 1. Actual placement

Writer: `exp-1114-115901-508160`, UID `dd6f01b6-50b5-4ee1-8598-17ff32131e93`; nine nodes, eight H800 each, codesign, creator i-fengyicheng. The roles occupy virtual clusters/worker groups inside one Ray deployment. Reader `exp-1114-115909-784580` is a separate0GPU evidence job.

| Role | GPUs | Share of72 | Parallelism |
| --- | ---: | ---: | --- |
| Student trainer and native student scorer | 8 | 11.11% | TP2 × PP4 × CP1 × DP1 |
| Student rollout | 48 | 66.67% | 24 independent vLLM engines; each TP2 × PP1, internal DP1 |
| Five teachers | 10 | 13.89% | Each teacher2GPUs; TP2 × PP1 × CP1 × DP1 |
| Unassigned model capacity in reserved teacher nodes | 6 | 8.33% | Two whole teacher nodes reserve16GPUs, of which10 are used |

There are seven logical model worker groups: student trainer, rollout fleet, and five teachers. There is no additional reference-policy, critic or GPU reward group in this run. Native scorer reuses the training workers; Gym verification is skipped by the accepted MOPD recipe. CPU control/Gym and eight TransferQueue storage units operate in the same Ray deployment.

Actor placement at13:30:18HKT reconciles66 model workers. The additional48 `RayWorkerProc` actors belong to vLLM's internal TP execution and must not be counted as another48GPUs. Runtime identities confirm H800 devices. Static placement:

| Nodes | Role |
| --- | --- |
| gpu-h800-0620 /10.191.12.56 | Student8 |
| gpu-h800-0535 /10.191.15.8 | math,logic,if:6GPUs;2 unassigned |
| gpu-h800-0474 /10.191.19.5 | code,stem:4GPUs;4 unassigned |
| gpu-h800-0032,0256,0071,0457,0223,0074 | Six rollout nodes,8GPUs each |

Sources: selected placement and engine evidence in `evidence/formal-performance-20261114.json`; resolved configuration in `evidence/formal-native-resume-6-generation48-resolved-config-20261114.json`; actual allocator `nemo_rl/algorithms/single_controller_utils/setup.py:401` and`:440`. Dashboard node-summary `now` timestamps were stale, so its dynamic GPU-utilization numbers are excluded. No current SM-utilization claim is made.

## 2. Measured wall time and its limits

The primary window is steps32–38, excluding resume/startup step31. All seven batches complete256 generations with0 generation failures. Earlier steps32–37 average614.32s; inclusion of the new long-tail step38 raises the mean to650.30s (10.84min). These are complete measured steps, not a GPU kernel profile.

| Disjoint controller wall-time category | Mean seconds,32–38 | Share |
| --- | ---: | ---: |
| Waiting for ready rollout batches | 561.85 | 86.40% |
| Student policy training | 66.93 | 10.29% |
| Student logprob scoring | 6.86 | 1.06% |
| Other preparation, advantage/dump/control and synchronization | 14.66 | 2.25% |
| Total | 650.30 | 100% |

| Step | Total seconds | Ready-batch wait seconds | Training seconds | Longest response tokens |
| --- | ---: | ---: | ---: | ---: |
| 32 | 617.21 | 528.22 | 66.84 | 20,917 |
| 33 | 653.03 | 557.13 | 73.52 | 26,109 |
| 34 | 602.93 | 512.95 | 67.69 | 20,106 |
| 35 | 543.95 | 466.28 | 58.23 | 20,993 |
| 36 | 605.82 | 521.44 | 62.48 | 23,589 |
| 37 | 662.95 | 576.04 | 66.89 | 26,026 |
| 38 | 866.20 | 770.92 | 72.84 | 37,996 |

`single_controller.py:2643` wraps sampler selection and waiting for ready groups in `exposed_generation`. A group becomes ready after its rollout and teacher enrichment (`opd.py:573`). Thus this timer includes the generation/readiness chain and cannot be labeled exclusively vLLM GPU computation. Teacher calls overlap different requests/engines.

The teacher counters in `opd.py:610` **sum call durations**: the seven-step mean is55.32s inference,2.76s lock wait and58.53s total over256 calls. They are not an additional wall-time stage to add to561.85s, and do not support claiming teachers occupy55s of the critical path. Weight sync is only0.622s/step (about0.096%); it is not the main latency source.

`train_data_dump` records serialization inside the advantage stage (`single_controller_utils/advantage_stage.py:694`) and finalization afterwards (`single_controller.py:3395`,`:5165`). It overlaps `advantage_calculation`; the table avoids double counting. The roughly8s dump timer is a low-priority cost, not an explanation for9–14min steps. The logged `valid_tokens_per_sec_per_gpu` is a throughput quantity even though the native text printer appends seconds; it is excluded from duration sums.

The earlier six-step window averages1,895,891 response tokens/batch,7,406 response tokens/sample, only288 prompt tokens/sample, and22,957 tokens for the longest response in each batch. Step38 has1,858,111 response tokens overall but a37,996-token longest response: total tokens are slightly lower while waiting increases to770.92s. This supports long-response latency and the whole-batch barrier as major contributors. It does not separately quantify GPU compute, CPU dispatch, transport and per-engine idle time.

The controller requires256 ready groups and `max_lookahead_versions=0`. The full-batch boundary keeps fresh same-version samples, but makes every update wait for late completions. At24 engines and256 inflight requests the arithmetic average is10.67requests/engine, not a measured occupancy. Trace records show sampler admission of all256 relevant groups spans less than0.002s per step; this excludes slow sampler admission itself, not downstream HTTP or engine scheduling. Speculation, increasing version lookahead, or truncating responses would require separate semantic/numerical analysis; none was applied.

Existing engine logs show4,841,312 KV tokens per TP2 engine and73.87x concurrency at65536 tokens. This is ample nominal capacity relative to the average request count, but does not prove balanced occupancy or zero preemption. Per-request completion traces, current SM activity and vLLM waiting/preemption statistics are not exposed by the collected scalar reports. HTTP `/metrics` and `/server_info` return404 from the repository's custom server; `/health` returns200. These optional inspection limits did not affect training and were not repaired as a side task.

## 3. Co-location support

`nemo_rl/algorithms/single_controller_utils/config.py:1400` rejects colocated generation unless the backend is Megatron. The current vLLM workflow therefore fails validation if only `policy.generation.colocated.enabled` is changed to true.

The repository supports Megatron trainer/generation co-location and documents its whole-batch requirement in `docs/guides/single-controller.md:28`. This requires switching the generation backend and verifying model support, HTTP rollout, native refit/resume, parity and throughput. It is not equivalent to reusing the present48GPU vLLM fleet on the8 training GPUs.

MOPD additionally requires `on_policy_distillation.non_colocated_teachers.enabled=true` (`config.py:2077`). There is no existing flag to time-share one GPU pool among the trainer, vLLM rollout and all five teachers. Native vLLM sleep/wake support is a building block, not an implementation of that controller path. Co-location could reduce reserved GPU time; its wall-time effect is unmeasured and may be adverse if generation capacity falls.

## 4. Performance settings and priorities

| Setting | Actual evidence | Assessment |
| --- | --- | --- |
| CUDA graphs / Inductor | Actual vLLM0.29.0 log: eager disables both; graph memory0 | Main candidate for an A/B. Eager was deliberately selected after historical BF16 compiled parity failure. |
| Graph capture with eager compiler backend | Official/local model guide recommends `enforce_eager=false`, `compilation_config.backend=eager` for this model family | This intermediate option remains unverified in current FP16/FA4/FP32-head/RoPE. Test before adopting; do not assume an observed speedup. |
| Internal async scheduling | TP2 forces Ray executor; installed and tagged vLLM0.29.0 Ray executor inherits capability=False | Source-derived unavailable path. `async_engine=true` enables the API, not the internal scheduler. Forcing the latter true would raise. |
| TP2 vs TP1 rollout | Current24TP2 engines, small average inflight population and ample KV capacity | Compare equal total GPU budgets. TP1 removes TP collective/Ray executor dependence but changes batching and KV capacity; faster is not guaranteed. |
| Chunked prefill | Actual engine logs: enabled,2048 tokens/iteration | Not missing. Higher budgets mainly target prefill/mixed workloads; prompt lengths are short here. |
| Prefix cache | Actual engine logs: enabled; reset after refit | Not missing. It cannot eliminate autoregressive response work. Keep cache invalidation correct after weights change. |
| `max_num_seqs` | Not explicitly set; installed H800 EngineArgs defaults imply1024 | Source-derived value, not a directly dumped resolved field. The expected limit is far above about11requests/engine; no demonstrated cap bottleneck. |
| Training packing | Resolved sequence_packing.enabled=true,65536-token packing budgets | Already active; dynamic_batching=false does not mean unpadded training. |
| CUDA_DEVICE_MAX_CONNECTIONS | Student logs warn it is not1 with sequence parallelism | Lower-priority trainer overlap candidate; training is only10.3% of total. No global env/NCCL modification was made. |
| Speculative decoding | Engine speculative_config=None, draft counters0 | Not a free missing switch; requires a supported draft mechanism and full-vocabulary/logprob correctness validation. |
| Diagnostic dumps | Full train-data dump enabled, about8s overlapping stage time | Preserve acceptance evidence now; optimize only if a later measurement makes it relevant. |

Historical BF16 comparison on160 response positions: native/eager MAE0.01905496; native/compiled0.03365744, FAIL at unchanged0.02. The present FP16 run uses a different precision. That historical result justifies caution but neither proves nor disproves FP16 graph-only parity. The concrete first candidate, without changing original assets, precision, FA4, full vocabulary or rollout dimensions, is:

```yaml
policy:
  generation:
    vllm_cfg:
      enforce_eager: false
    vllm_kwargs:
      compilation_config:
        backend: eager
```

Recommended order: graph-only FP16 parity and equal-workload timing; then equal-budget TP1/TP2 comparison; then batching or role reallocation only when the measured results justify it. Validate actual graph capture, identical-token scorer/training/vLLM errors, failures/preemptions and complete-batch wall time. Reuse the latest permitted native checkpoint and in-memory synchronization, saving no new diagnostic checkpoints or weight exports. No such A/B was launched in this investigation. Detailed primary-source findings and eight source groups are in `performance_official_sources.md`.

## 5. Commands, verification and current mainline

Read-only analysis, local Python3.10.12:

```bash
python3 /data/ycfeng/tmp/nemo-mopd-20261112/analyze_formal_performance_20261114.py
```

Expected: finite exact-step metrics;256 unique valid same-version samples; all five groups; no mask/routing issues; overall and group scorer/vLLM MAE≤0.02; native success flags count actual successes; placement8+48+10+6=72; non-overlapping duration buckets sum to total. Actual: **PASS**. Steps37–38 each have281 finite metrics and256 passing sample records; formal count20/200 across21 attempts, with only historical step25 skipped. Step38 completed13:35:25HKT, loss/full-RKL0.1235070825, normalized pre-clipping norm0.5368999839, scale1; overall scorer/vLLM MAE0.0011346195, max0.4286568. Passing the mean does not prove tokenwise equality or final actual-training parity. Exact evidence: `evidence/formal-native-resume-6-step37-step38-health-20261114.json`.

Existing native checkpoint metadata inspection:

```bash
timeout 60s env \
  UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker \
  /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python \
  /data/ycfeng/tmp/nemo-mopd-20261112/inspect_latest_formal_native_state_20261114.py
```

Python3.13.14, UV0.13.7, Torch2.13.0+cu130. At13:37:06HKT only `.../final-five-teacher-fp16-resume/exp-1113-174329-659007/checkpoints/step_30` remains:31files,24,884,408,499bytes, Adam12 across eight ranks;551,606metadata/scalar bytes read and0model/Adam tensor payloads. **PASS latest-only**, with8 later successes still resident. No extra save was triggered. Next ordinary publication isstep40; if no new skips, expect Adam22 and native deletion ofstep30 after complete publication. Physical OSS reclamation is not measured.

Dashboard reads use a proxy-free urllib opener and exact writer actors: `/api/v0/actors?limit=1000&detail=true`, `/nodes?view=summary`, `/api/v0/logs?node_id=<exact-id>&glob=*2076*&timeout=10`, and `/api/v0/logs/file?node_id=<exact-id>&pid=2076&suffix=out&lines=2000&timeout=10&filter_ansi_code=true` on `10.191.6.23:9254`. Selected raw logs/index and actor snapshots reside under `/data/ycfeng/tmp/nemo-mopd-20261112/performance-*`; only relevant sanitized evidence is retained in the task directory. Official-source reads used bounded urllib requests after web search returned no results; the research file records source URLs and version boundaries.

P is complete as an investigation. E/F remain open:180 more successful updates, latest-only regular retention checks, final full-tensor finite validation, unsaved actual-training/scorer/vLLM replay, refreshed final curves and English completion archive. Current train configuration and all existing monitor/reader/final-waiter sessions continue unchanged.
