## Modification History

| Date | Change |
| --- | --- |
| 2026-11-13 | Record effective native scaled-overflow RED/GREEN, original-config replay, and regression scope. |

# Native FP16 scaled-gradient overflow

Actual five-teacher FP16 job exp-1113-010116-581403 failed before a verified optimizer update with `found Inf in local grad norm for bucket #0 in backward pass before data-parallel communication collective`. The pre-collective DDP fatal check evaluated scaled gradients before the native mixed-precision optimizer could unscale, detect nonfinite values across ranks, skip the unsafe update and lower the dynamic loss scale.

The repair creates the actual OptimizerConfig once, passes it to ConfigContainer, and derives DDP check_for_nan_in_grad as not optimizer_cfg.fp16. FP16 keeps native global nonfinite reduction, scaler updates and skipped-update reporting. BF16 and FP32 retain fatal nonfinite DDP checks. No loss-scale constant is reduced manually and no unrelated training precision is changed.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Native CPU seam before repair | Reproduce the exact early fatal bucket Inf | 1 failed, stop after first failure,26.05s; fatal Inf at param_and_grad_buffer.py:420 | RED verified |
| Inf/NaN after repair | Reach native unscale/scaler, lower scale after hysteresis, accept subsequent finite gradient | 2 FP16 cases pass; final unscaled gradient0.25, native global MAX nonfinite reduction asserted | PASS |
| BF16/FP32 | Preserve fatal detection of unscaled nonfinite gradient | 2 cases pass | PASS |
| Affected config builders | Preserve Gloo, FP8 recipes/buffers and offload config behavior |14 passed,329 deselected,20.81s | PASS |
| Original prepared formal policy config | Native Inf then NaN trigger scaler backoff; finite batch continues | Native reproduction exits0, scaler65536→32768, finite gradient0.25 | PASS |
| Scoped lint/format | Existing pinned Ruff0.9.9 and git diff --check | Both changed modules pass; format normalization only in test module | PASS |
| Actual five-teacher GPU save/resume | Successful parameter updates, native optimizer/scaler/scheduler recovery, all teacher coverage and numerical parity | Pending new GPU run; earlier job is FAIL and not retroactively repaired | UNVERIFIED |

## Environment and commands

CPU runtime is Python3.13.14 in /data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker; Torch2.13.0+cu130, CUDA_HOME=/data/ycfeng/toolchain/cuda-13.0.2, LD_LIBRARY_PATH=/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat, TMPDIR=/data/ycfeng/tmp, OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4. Native Megatron source is the repository's pinned submodule. The seam invokes native bucket/scaler/optimizer methods on CPU and patches only device allocation and the collective transport; actual GPU communication is not covered.

```bash
uv run --no-sync pytest --noconftest tests/unit/models/megatron/test_megatron_setup.py::TestCreateMegatronConfigGradOverflow -m mcore -q -s --tb=short
uv run --no-sync pytest --noconftest tests/unit/models/megatron/test_megatron_setup.py -k 'CreateMegatronConfig or OptimizerDtype' -m mcore -q --tb=short
uv run --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/probe_native_fp16_overflow_order.py
uv tool run --offline --default-index http://mirrors.i.basemind.com/pypi/simple/ --from ruff==0.9.9 ruff check nemo_rl/models/megatron/setup.py tests/unit/models/megatron/test_megatron_setup.py
uv tool run --offline --default-index http://mirrors.i.basemind.com/pypi/simple/ --from ruff==0.9.9 ruff format --check nemo_rl/models/megatron/setup.py tests/unit/models/megatron/test_megatron_setup.py
```

Each native check is wrapped in timeout60s. The UV executable is /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv and UV_PROJECT_ENVIRONMENT is the actor environment above. --noconftest excludes the unrelated autouse Ray packing fixture; earlier red-4 timeout124 is not counted as RED. Ruff was restored from the internal package mirror after deleted download/index cache made offline tool resolution unavailable; installed training environments were unchanged. Exact outputs are retained in evidence/native-fp16-overflow-*.log.
