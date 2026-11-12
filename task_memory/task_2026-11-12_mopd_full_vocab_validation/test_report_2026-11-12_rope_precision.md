## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record corrected layer observations and identify the first RoPE precision divergence. |

# Long-context RoPE precision diagnosis

The first observed native/vLLM difference is RoPE arithmetic. Corrected embedding and first-layer QKV are exactly equal for all five samples and both TP ranks. The native fused operation uses FP32 trigonometric values; the original vLLM CUDA operation uses a BF16 cosine/sine cache. The FP32 intervention eliminated the first Q/K difference, but the remaining end-to-end error still fails acceptance. The original long-context acceptance remains FAIL; this report does not claim the overall discrepancy is fixed.

## Actual execution

| Role | RJob | GPUs | Terminal | Launcher |
| --- | --- | --- | --- | --- |
| Native CP1/TP2, fused RoPE | exp-1112-200933-746577 | 2 H800 | Succeeded 2026-11-12T12:12:41Z | 80034, exit0 |
| Eager vLLM TP2, FA3, chunk2048 | exp-1112-200933-693626 | 2 H800 | Succeeded 2026-11-12T12:12:37Z | 40877, exit0 |
| FP32 RoPE intervention | exp-1112-201719-338060 | 2 H800 | Succeeded 2026-11-12T12:20:27Z | 57505, exit0 |

The paired runs use personal i-fengyicheng auth, codesign, non-preemptible resources, and local NFS100.96.128.195:/data/ycfeng:/data/ycfeng. Both runtime receipts confirm Python3.13.14, Torch2.13.0+cu130, CUDA13.0, H800 and architecture9.0. Exact image and arguments are in `long-{native,generation}-attention-layers-submit-args.json` under `/data/ycfeng/tmp/nemo-mopd-20261112/`; submission wrappers use the same prefix. The task's initial student and fixed original65536-token samples are unchanged.

Reusable observation scripts:

- `tests/manual/mopd/inspect_native_layers.py` selects `layer_output_worker.py` through the original Policy/Ray worker builder.
- `tests/manual/mopd/inspect_generation_layers.py` uses original prompt replay plus native model inspection; `generation_layer_outputs.py` observes its model states.

Exact paired launch commands:

```bash
PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_long_native_attention_layers.py
PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_long_generation_attention_layers.py
```

Both wrappers redirect to corresponding `*-submit.log` files. Each RJob writes durable captures under `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/long-{native,generation}-attention-layers/<RJob>/logs/layer-outputs`. Bounded read-only SSH tar brought these, replay logprobs, finished.json and runtime-0.json into local `long-{native,generation}-attention-layers-inputs`.

## Comparisons and conditions

The comparison uses the same sample IDs, global prediction positions (response mask indices minus1), all32 response positions per sample, and corresponding TP ranks. No tokens are filtered. Native sequence-parallel layer outputs are gathered by global position; QKV/attention use full CP1 positions with TP feature shards. All inspected tensors are finite.

| Comparison | Observed value | Verdict |
| --- | --- | --- |
| Embedding native/vLLM | MAE0, max0, all5samples | PASS: input and embedding identities |
| First-layer rawQKV and Q/K before RoPE | MAE0, max0, bothTP ranks | PASS: first projection identities |
| Native FP32 cos/sin cast to BF16 versus actual vLLM cache | MAE0, max0 | PASS: frequencies agree at cache precision |
| Q after RoPE | max0.5; relativeRMSE0.00223–0.00258 | FAIL: first measured divergence |
| K after RoPE | max2.0; relativeRMSE0.00239–0.00288 | FAIL: first measured divergence |
| First coreattention output | relativeRMSE0.03661–0.16842 | FAIL: difference is amplified |
| Native operation versus FP32 trigonometric formula | Almost exact; largest isolatedQ difference0.03125, mean≤1.3e-6 | PASS as arithmetic diagnostic; CPU/GPU trig rounding limit retained |
| vLLM operation versus BF16-cache formula | MAE0, max0 for Q/K | PASS as arithmetic diagnostic |

These measurements establish the precision cause for the first divergence at the observed positions. They do not prove that every later discrepancy comes from it. Eager versus the original asynchronous generation previously had MAE0.04009260, and eager versus the matched compiled prompt replay had MAE0.03340532. The eager diagnostic is a separate execution mode. A subsequent corrected real-generation run remains required.

Exact analysis command:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 timeout 60s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --no-sync python -u /data/ycfeng/tmp/nemo-mopd-20261112/analyze_attention_layers.py
```

Raw output: `attention-layer-comparison.log`; selected full comparisons: `evidence/native-long-attention-layer-comparison.json`. Terminal/runtime evidence: `evidence/native-long-attention-{native,generation}-terminal.json`.

## Precision intervention and verification limits

The prepared test-only `--fp32-rope` option uses vLLM's existing FlashInfer rotary operation, whose pinned implementation requires an FP32 cosine/sine cache while retaining BF16 query/key storage. It recomputes the cache on its original device before rounding. Upcasting the old BF16 cache would preserve the defect and is deliberately not used. No native model arithmetic, token mask, tolerance, or production worker configuration has changed.

CPU preparation used the actual installed RotaryEmbedding with an explicit CPU DeviceConfig. Cache reconstruction/device/selector and CLI-to-engine boundary passed in `fp32-rope-preflight-2.log`; initial auto-device inference failure is retained in `fp32-rope-preflight.log`. Boundary observation survived storage mutation and shared-module hooks respected current-layer identity (`layer-capture-boundaries-2.log`). Native observer import passed after using the existing worker CUDA compatibility loader path (`attention-layer-native-import-2.log`). Ruff check and format passed for all four scripts. The subsequent actual CUDA run establishes first-layer Q/K alignment; it does not establish end-to-end parity.

Next acceptance: corrected postRoPEQ/K versus native, full160-token response logprobs versus nativeCP1, then the original fresh-generation workflow using the verified precision route. The unchanged mean-error limit is0.02 nats/token; group and tail checks remain required. Formal200updates are not submitted.

## FP32 intervention observed result

The completed intervention retains all 160 response tokens. First-layer post-RoPE Q/K match native exactly (MAE/max0). Actual FP32 cos/sin differ by at most5.96e-8. First attention relative RMSE falls to approximately0.000335–0.000630, while complete first-layer output MAE remains0.000645–0.000703. NativeCP1 minus corrected eager prompt logprobs: signedmean0.00899168677, MAE0.03176860982, p950.08699339, p990.21216241, max0.50518227: **FAIL** against0.02. Per-sample means:0.01852770/0.03939524/0.03309640/0.04153882/0.02628489. Four of five samples fail individually.

Exact intervention submission: `PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_long_generation_fp32_rope.py`. Analysis: the same constrained native Python/uv environment above running `analyze_fp32_rope.py`; raw output `fp32-rope-comparison.log`. Selected terminal/parity/layer evidence is `evidence/native-long-fp32-rope-{terminal,parity,layer-comparison}.json`. Cloud artifacts: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/long-generation-fp32-rope/exp-1112-201719-338060/`. Launcher57505 and monitor59780 exited0. No active GPU launcher remains from these completed probes.

Next falsifiable hypothesis: native unfused `F.silu(gate) * up` rounds a BF16 intermediate that vLLM fused `SiluAndMul` retains at higher precision. Enable the existing native `bias_activation_fusion=true` for one CP1/TP2 fixed-token replay and capture the first MLP projection/activation rows. If causal, first-layer outputs and all-token logprobs should move toward the saved FP32-RoPE vLLM outputs. If insufficient, inspect the first remaining residual/norm/MLP projection divergence; do not repeat excluded CP/head/RoPE/backend defaults. Production FP32 RoPE integration and actual compiled asynchronous generation remain outstanding.

## Existing SwiGLU fusion comparison

One nativeCP1/TP2 fixed-token replay, `exp-1112-202910-954148`, Succeeded2026-11-12T12:31:26Z; launcher74729 exit0. Personal creator/codesign/localNFS confirmed. It enables only the existing `policy.megatron_cfg.bias_activation_fusion=true`. All160response-token values are finite. MAE against saved FP32-RoPE eagervLLM is0.03172934032, signedmean0.00024701809,p950.08786572,p990.23786309,max0.39793074: **FAIL**. Every per-sample mean fails0.02. First-layer output MAE0.0009077–0.0010146 is worse than the original unfused0.000645–0.000703.

Both first MLP activation captures on bothTP ranks match the FP32 SiLU/product formula exactly (MAE/max0); the BF16 intermediate formula differs by up to2.0, mean≤8.609e-5. This confirms native fusion changes arithmetic, but excludes it as a sufficient remedy. Preserve original unfused recipe setting. It does not establish the unobserved vLLM activation precision.

Exact submit: `PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_long_native_fused_swiglu.py`. Analyze via the constrained basePython3.13/uv command above with `analyze_fused_swiglu.py`; rawlog `fused-swiglu-comparison.log`. Selected terminal/parity/layer/activation evidence: `evidence/native-long-fused-swiglu-*.json`. Cloud: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/long-native-fused-swiglu/exp-1112-202910-954148/`. Original systemPython tar extraction rejected its unsupported filter argument before any extraction; nativePython3.13 extraction succeeded.

## Production cache repair preparation

The existing vLLM source-patch mechanism now supports explicit `hf_overrides.rope_parameters.fp32_rope=true` for default full-head BF16 rotary embeddings. The existing factory key includes the entire rope_parameters mapping, so corrected and original modules cannot share a BF16 cache. Recompute cache before model compilation and select the same FlashInfer operation verified above; reject unsupported scaling/partial rotation/precision and non-boolean requests. No new NeMo config schema or automatic global precision change. Worker and native prompt replay request the patch before engine construction. Keep the initial model's full original rope_parameters alongside the precision request; do not overwrite rope_theta with an incomplete override.

Observed CPU RED: actual factory ignores requested precision and returnsBF16 cache/use_flashinfer=false (`rope-factory-red.log`). Six real-factory source/cache/unsupported/idempotence checksPASS in11.26s; affected worker request propagation plus previous routed-expert behaviorPASS,12checks/39deselected in11.49s (`rope-factory-worker-regression.log`). All use mounted pinned vLLM actorPython3.13/Torch2.13.0+cu130, CUDA compatibilityloader, uv--no-sync, pytest--noconftest and4CPUthreads; Ruff0.9.9PASS. These are CPU evidence, not yet actual productionCUDA/compile/fresh-generation validation.

Next pre-execution head hypothesis: after aligningRoPE, BF16 head output rounding may still dominate responseLP error. Paired existing FP32 heads should reduce the remaining MAE/tails if causal. Native keeps original unfusedSwiGLU/CP1/TP2/tokens; vLLM eager captures also observe firstattentionprojection/MLP actual arithmetic. Separate compiled prompt and actualfresh-generation checks remain required after capture acceptance.

## Core repair GPU result and submitted-frequency error

Native FP32-head CP1/TP2 (`exp-1112-203938-713091`), eager core FP32-RoPE/head (`exp-1112-203938-677167`), and compiled core FP32-RoPE/head (`exp-1112-203959-100754`) all Succeeded. All three local launchers exited0. Selected terminal evidence records creator i-fengyicheng, codesign and current-machine source100.96.128.195:/data/ycfeng:/data/ycfeng. Native finished2026-11-12T12:42:28Z. Exact arguments and logs use `long-generation-core-fp32-{layers,compiled}` and `long-native-fp32-head-layers` under the local tasktmp root.

All160 response positions remain finite. Native minus eager MAE1.9457531214/p99=8.8010214710/max12.1376628876; native minus compiled MAE1.9473593901/p99=8.7890008545/max11.9961223602. Both **FAIL**. Eager minus compiled MAE0.0238099017, max0.1757888794; do not treat the modes as interchangeable. The first-layer raw QKV still matches exactly, but post-RoPE Q/K and cache now diverge strongly.

The submitted overrides incorrectly used rope_theta1000000.0. The original student `config.json` explicitly declares rope_theta10000. The actual native angle capture agrees with10000 up to CPU/GPU FP32 frequency rounding(max0.000244140625 radians), while1000000 differs by9689.498046875. The actual generation cache agrees with1000000 (MAE1.331706e-6/max0.0001116693), and differs from10000 byMAE0.81184137/max1.99904513. Thus this enlarged error is caused by the submitted model-frequency change. It cannot establish whether the core cache repair matches the earlier intervention at the original frequency.

Analysis command: constrained basePython/uv command above with `analyze_core_fp32.py`, rawlog`core-fp32-comparison-2.log`. Its first attempt retained a sequence-parallel position assertion error in`core-fp32-comparison.log`; merge attention projection/MLP output rows by global positions, retain TP feature-shard comparisons for MLP projection/activation. CPU analysis then exited0 without any GPU rerun. Frequency diagnostic command uses`verify_core_rope_theta.py`, selected evidence`native-long-core-fp32-theta-preflight-2.json`. Its original bitwise CPU/GPU angle assertion failed; the difference is explicitly reported, not suppressed as equality.

Before the affected rerun: restore the original config's10000 frequency, change no tokens/weights/masks/parallelism/head/cache arithmetic, reuse the completed native capture. If the submitted frequency caused the regression, first-layer post-RoPE Q/K should again match and end-to-end error should return to the earlier precision regime. Independently compare eager and compiled; either still requires fresh rollout/refit acceptance. Corrected2GPU wrappers/arguments use`long-generation-core-fp32-{layers,compiled}-theta`; both personal dry-runsPASS. Formal200updates remains unsubmitted.

## Core repair at the original frequency

Corrected-frequency eager `exp-1112-205326-305203` and compiled `exp-1112-205326-529652` both Succeeded. The two launchers63645/57389 and monitor53952 exited0. Actual model frequency is10000 in the exact corrected arguments. Both retained all160 response positions, unchanged native FP32-head weights and masks, and native completion receipts. Small selected evidence is in `native-long-core-fp32-theta-*.json`.

First-layer embedding, QKV and post-RoPE Q/K are now exactly equal. FP32 cache max difference5.96e-8 and first attention relativeRMSE0.000335–0.000630 reproduce the earlier test-only intervention. Thus the production factory integration fixes the measured first precision defect. It does not yet fix the entire long-context discrepancy.

| Paired comparison | MAE | p99 | max | Criterion |
| --- | --- | --- | --- | --- |
| Native FP32 head minus corrected eager | 0.0265770316 | 0.2228605831 | 0.5153255463 | FAIL, ceiling0.02 |
| Native FP32 head minus corrected compiled | 0.0385081589 | 0.4230281925 | 0.6105447412 | FAIL, ceiling0.02 |
| Eager minus corrected compiled | 0.0278024156 | 0.1574191380 | 0.3677145839 | FAIL, modes remain distinct |

Actual eager activation equals the BF16 SiLU-intermediate/product formula exactly; it differs from the FP32 formula. The native fusion control is not promoted. First-layer attention projection MAE0.000105–0.000140 and MLP output MAE0.000579–0.000629 remain nonzero. Native config pins FA2, while vLLM uses FA3; native auto selection also permits cuDNN. The next control uses the existing native `attention_backend=flash` with `flash_attention_version=3`, retaining every other native setting. If attention accumulation is causal, core attention and complete response-logprob errors should decrease. If it is insufficient, the retained first two layer states delimit residual/norm/projection investigation.

Exact comparison command: constrained basePython3.13/uv command above with`analyze_core_fp32_theta.py`; rawlog`core-fp32-theta-comparison.log`, exit0. Both cloud roots append `-theta` to their original names and retain exact RJob names. Reusable observation now uses only the verified production `hf_overrides.rope_parameters` route; the duplicate test-only `--fp32-rope` implementation is removed after its historical intervention was measured. Historical intervention artifacts remain retained.

## Final core regression checks

After removing the duplicate test-only precision route, Ruff0.9.9 check and format passed for8affected files. The mounted vLLM actor runtime ran `uv run --no-sync python -m pytest --noconftest tests/unit/models/generation/test_vllm_patches.py -k 'fp32_rope or threads_routed_experts_capture' -q`: **12PASS**,39deselected,11.50s. Exact environment: UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker, PYTHONPATH=/data/ycfeng/sota-engine/RL, existing CUDA compatibility LD_LIBRARY_PATH, OMP/MKL/OPENBLAS4, timeout60s. Rawlog`rope-final-worker-regression.log`. No widened numerical tolerance.

The first FA3 dependency preflight found no independent FA3 in the native actor environment; no GPU job was submitted. Existing nativeFA4 config/backend setup passes. vLLM's FA4 implementation is vendored at`vllm.vllm_flash_attn.cute.interface`; an initial preflight checked the independent package name and failed withModuleNotFoundError, which does not establish vendoredFA4unavailability. Original errors are retained in`long-native-fa3-preflight.log` and`fa4-generation-preflight.log`. The matched FA4 hypothesis retains unchanged original frequency/cache/head and inspects actual coreattention outputs.
