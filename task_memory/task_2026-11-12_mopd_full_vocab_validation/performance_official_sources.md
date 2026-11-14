## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Verify vLLM 0.29.0 and NeMo-RL primary sources; distinguish measured parity constraints, executor scheduling limits, and unmeasured performance candidates. |

# MOPD performance: official-source findings

The strongest immediate candidate is CUDA graph execution with the eager compilation backend, subject to renewed FP16 train/inference parity checks. The present eager setting was deliberately selected after a measured BF16 compiled-path failure. The current SingleController vLLM route cannot be made colocated by changing one flag. A further concrete constraint is that TP2 selects the Ray executor, whose vLLM 0.29.0 implementation does not support internal asynchronous scheduling. None of these findings establishes a measured speedup for the current run.

This report is a read-only research deliverable. It does not change the running workload, submit GPU jobs, or read/write checkpoint payloads. Live placement and current timing attribution belong to the parent investigation. The latest-only checkpoint rule remains in force.

## Evidence and version boundary

- The actual generation actor environment reports `Name: vllm`, `Version: 0.29.0` in `/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/lib/python3.13/site-packages/vllm-0.29.0.dist-info/METADATA`. The container image's older version label is not the installed engine version.
- The local repository HEAD inspected was `aeeeecd6d6a92d499e7c4617125c682d8d70816d`. Local source references below identify the actual checkout behavior.
- [Recorded resolved configuration](evidence/formal-native-resume-6-generation48-resolved-config-20261114.json): 48 generation GPUs, TP2/PP1, therefore 24 independent engine replicas; FP16, FP32 LM head and RoPE, FA4, `enforce_eager=true`, `async_engine=true`, `max_num_batched_tokens=2048`, `gpu_memory_utilization=0.85`, `max_model_len=65536`. The controller uses `in_order`, `max_lookahead_versions=0`, 256 inflight prompts and 256 groups per training chunk. These are recorded configuration facts, not a fresh live-placement observation.
- All vLLM references below use the official `v0.29.0` tag. NVIDIA NeMo-RL `main` documents were fetched on November 14, 2026 and their relevant passages agree with the inspected local documents. No recommendation depends on an unverified later vLLM default.

## 1. CUDA graphs are a candidate, but eager was intentional

vLLM documents a steady-state decode cost for `enforce_eager`: it disables compilation and graph capture. CUDA graph modes can be selected separately from compiler optimization; their availability depends on the attention backend and batch shape. Thus graph execution and enabling Inductor are separate decisions. [S1, S2]

NeMo-RL's model-validation guide specifically discusses `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. It recommends checking `enforce_eager=false` with `compilation_config={"backend":"eager"}` to retain compatible graph capture without Inductor's numerical changes. Its historical measurements do not establish behavior for our FP16/FA4/FP32-head/FP32-RoPE configuration. [S8]

The local [matched FA4 report](test_report_2026-11-12_matched_fa4.md) records why eager was selected on November 12:

| Historical BF16 comparison, 160 response positions | Mean absolute logprob error | Existing acceptance ceiling |
| --- | ---: | ---: |
| Native versus eager FA4 | 0.01905496458 | 0.02 |
| Native versus compiled FA4 | 0.03365743577 | 0.02 |
| Eager versus compiled FA4 | 0.03563335682 | Descriptive comparison |

The report explicitly adopted eager and retained compiled mode as a failing path. Current FP16 graph execution has not been verified. Calling the setting an accidentally forgotten debug switch would contradict this evidence.

The smallest candidate configuration to evaluate through existing parity and rollout checks is:

```yaml
policy:
  generation:
    vllm_cfg:
      enforce_eager: false
    vllm_kwargs:
      compilation_config:
        backend: eager
```

The wrapper passes `enforce_eager` and `vllm_kwargs` into the native engine in `nemo_rl/models/generation/vllm/vllm_worker.py:798`. Before adoption, inspect the actual selected graph mode and successful capture, then compare identical response-token logprobs under the existing 0.02 ceiling, including tails and per-group errors. Preserve precision, attention, full vocabulary and rollout lengths. Reuse permitted assets and in-memory synchronization; do not create diagnostic checkpoints or HF exports.

**Verdict:** concrete performance tradeoff with historical correctness justification; graph-only FP16 improvement is unmeasured.

## 2. Internal async scheduling is unavailable on the current Ray executor

There are three distinct mechanisms: asynchronous rollout/training orchestration, the vLLM asynchronous engine API, and internal `scheduler_config.async_scheduling`. Enabling the second does not establish that the third is active.

The vLLM 0.29.0 config initially allows `async_scheduling=None`. Resolution enables it only for compatible workloads and executors; an incompatible executor produces `False`. Explicitly forcing `True` on an unsupported executor raises an error. [S5]

The actual source chain is decisive:

1. `nemo_rl/models/generation/vllm/worker_utils.py:41` returns `"ray"` when TP times PP exceeds one.
2. `nemo_rl/models/generation/vllm/vllm_worker.py:652` installs that result as the engine's `distributed_executor_backend`.
3. vLLM 0.29.0 `RayDistributedExecutor` inherits `Executor` and defines no `supports_async_scheduling` override; the base implementation returns `False`. Both installed source and official tagged source agree. [S6]
4. vLLM `config/vllm.py:1296` therefore disables internal async scheduling for this executor unless an unobserved runtime patch changes it. The actual resolved engine log remains the final runtime confirmation. [S5]

For comparison, installed `UniProcExecutor` and `MultiprocExecutor` return `True`; this does not authorize replacing the repository's Ray placement mechanism with a different executor. A TP1 candidate uses a different executor path and must be checked in its actual environment.

**Verdict:** a known executor limitation, not a missing boolean. Do not set `async_scheduling=true` on the current TP2/Ray path.

## 3. TP1 versus TP2 requires a workload measurement

Official guidance describes TP as a model-memory/KV-capacity tool and warns about synchronization overhead; replication scales request throughput. This establishes reasons to compare TP1 with TP2, not a guarantee that TP1 is faster. [S1]

At 48 generation GPUs, TP2 gives 24 replicas. TP1 would give 48 replicas if placement succeeds. With 256 simultaneous prompts, the arithmetic average falls from about 10.67 to 5.33 requests per replica. Smaller per-engine batches can offset gains from removing TP collectives. TP also changes per-GPU KV capacity and kernel shapes; the model's 1.5B weight size alone does not settle 65,536-token capacity. These are deductions from the recorded configuration, not GPU measurements.

Compare equal total GPU budgets with the same prompts, output constraints and numerical settings. Inspect per-engine request/decode latency, useful generated tokens per second, memory/KV occupancy, preemptions and identical-token parity. A TP1 result also changes executor scheduling support, so report that combined effect rather than attributing all improvement solely to TP collectives.

**Verdict:** worthwhile second candidate after resolving graph behavior; no supported speedup estimate yet.

## 4. Chunked prefill, batching and prefix caching

vLLM 0.29.0 normally enables chunked prefill when supported. Its guide describes 2048 tokens per iteration as favoring inter-token latency, while larger budgets can improve prompt throughput; it suggests values above 8192 for throughput-oriented small-model workloads on large GPUs. This is workload guidance, not an acceptance threshold. [S1]

`max_num_batched_tokens` limits tokens scheduled in one engine iteration. It does not set the number of full trajectories collected by NeMo-RL. At approximately 11 inflight requests per TP2 replica, ordinary one-token decode does not approach a 2048-token iteration budget. Raising this value mainly changes prefill or mixed-batch scheduling unless other measured work fills the budget. A 2048 setting alone cannot explain minutes of pure decode.

`max_num_seqs` and chunked-prefill state must be obtained from the engine's resolved configuration, not the scheduler dataclass defaults: vLLM's `EngineArgs.create_engine_config` resolves hardware and usage-context defaults. The recorded recipe omits both explicit values; omission alone is not evidence that either feature is disabled.

Prefix caching reuses shared prompt prefixes. It does not avoid the autoregressive computation of newly generated answer tokens, so it cannot remove a long-decode tail. [S3] Weight synchronization also requires valid KV-cache handling; the current controller's `recompute_kv_cache_after_weight_updates=true` must not be disabled solely for performance without validating correctness.

**Verdict:** inspect actual prefill fraction, resolved limits and KV/preemption metrics before changing batching. Prefix caching is not a general fix for long responses.

## 5. Current co-location support is limited to specific roles and backends

| Scope | Source-backed status | Consequence |
| --- | --- | --- |
| vLLM engine capability | Native sleep/wake can release weights/KV allocations and supports staged wake for weight updates. [S4] | An underlying building block exists; this does not prove a framework's controller supports it. |
| Current SingleController plus vLLM | `single_controller_utils/config.py:1400` rejects colocated generation unless the backend is Megatron; `grpo.py:5368` repeats the restriction. [S7] | Flipping `policy.generation.colocated.enabled=true` fails setup. |
| SingleController plus Megatron generation | Documented colocated route with Megatron trainer; NeMo-Gym requires `expose_http_server=true`; whole-batch training requires `min_groups_for_streaming_train == num_prompts_per_step`. [S7] | A backend migration route exists, subject to this MOPD model/configuration's execution and parity checks. |
| Teachers in SingleController MOPD | `single_controller_utils/config.py:2077` explicitly requires `non_colocated_teachers.enabled=true`. | Existing support does not provide one shared GPU pool for trainer, rollout and all five teachers. |

The older assertion text at `grpo.py:1120` mentions vLLM, but its actual predicate `should_use_async_rollouts` also returns `True` for Megatron (`nemo_rl/models/generation/interfaces.py:310`). That message must not be read as proof that all Megatron MOPD paths are categorically forbidden. Conversely, passing this predicate is not an end-to-end verification of the proposed migration.

The recorded zero-lookahead, full-batch controller already limits overlap between successive weight versions. A time-sharing design could reduce idle trainer allocation, but changing to Megatron generation or implementing vLLM orchestration is larger than tuning one flag. It may reduce resource cost without reducing wall-clock time; both must be measured.

**Verdict:** no supported single-flag conversion for the current vLLM workflow; a documented Megatron trainer/rollout route exists, while teachers remain dedicated.

## 6. Long-response tails and timing interpretation

Recalculation from [steps 31–35](evidence/formal-native-resume-6-step31-step35-health-20261114.json) yields:

| Recorded measurement | Five-step mean |
| --- | ---: |
| Total step duration | 611.0577 s |
| `exposed_generation` | 516.4678 s, 84.5203% of total |
| Policy training | 70.3483 s, 11.5125% of total |
| Other/residual time | 24.2417 s, 3.9672% of total |
| Response tokens per batch | 1,954,712.4 |

These are historical samples for this resumed segment, not the newest live snapshot. `exposed_generation` is controller-visible waiting time and must not be described as a GPU-kernel profile. It can include producer work, teacher scoring, scheduling and delivery dependencies. The parent investigation must inspect its exact timing boundary and existing per-role counters before assigning all 516 seconds to vLLM decode.

The same records contain maximum sequence lengths between 20,292 and 26,401 tokens. With a full 256-group training boundary and zero version lookahead, a few late requests can hold back the next update; whether they actually dominate is an inference to test using request completion times and engine occupancy. Adding replicas can improve throughput yet leave an individual long-response latency unchanged. Do not reduce 256 fresh trajectories or the 65,536 ceiling to make this acceptance run look faster.

## Primary sources

All URLs were read successfully with bounded Python `urllib.request` requests on November 14, 2026 (HTTP 200). The web-search tool returned empty results; citations here refer to the actual retrieved primary text, not unavailable search snippets.

- **S1 — vLLM 0.29.0 optimization guide:** https://github.com/vllm-project/vllm/blob/v0.29.0/docs/configuration/optimization.md — eager steady-state cost, chunked-prefill tradeoffs, preemption and parallelism guidance.
- **S2 — vLLM 0.29.0 CUDA graph design:** https://github.com/vllm-project/vllm/blob/v0.29.0/docs/design/cuda_graphs.md — independent compilation/graph controls and attention/batch compatibility.
- **S3 — vLLM 0.29.0 automatic prefix caching:** https://github.com/vllm-project/vllm/blob/v0.29.0/docs/features/automatic_prefix_caching.md — scope of reuse and decoding limitation.
- **S4 — vLLM 0.29.0 sleep mode:** https://github.com/vllm-project/vllm/blob/v0.29.0/docs/features/sleep_mode.md — memory release, sleep levels and staged weight/KV wake.
- **S5 — vLLM 0.29.0 runtime configuration:** https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/vllm.py#L1220-L1311 — compatibility-based async scheduling resolution.
- **S6 — vLLM 0.29.0 executor capability:** https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/executor/abstract.py#L382-L387 and https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/executor/ray_executor.py#L67 — base returns false; Ray inherits it without an override.
- **S7 — NVIDIA NeMo-RL SingleController guide:** https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/single-controller.md#mandatory-settings — vLLM separation and Megatron co-location requirements; locally cross-checked against `docs/guides/single-controller.md:28` and both runtime guards.
- **S8 — NVIDIA NeMo-RL model validation guide:** https://github.com/NVIDIA-NeMo/RL/blob/main/docs/adding-new-models.md — the CUDA graph/Inductor precision comparison for this model family; locally cross-checked against `docs/adding-new-models.md:295`.

## Verification limits and next decision

Source behavior and recorded arithmetic are verified; performance improvement and current live placement are not established by this research. No new runtime test was conducted. Recommended sequence: finish existing per-role timing attribution; assess graph execution with eager compilation under the unchanged FP16 parity contract; then compare TP1/TP2 and batching only when the measured bottleneck warrants them. The main task's 200 successful updates, final parity and checkpoint-retention verification remain the parent task's unfinished acceptance work.
