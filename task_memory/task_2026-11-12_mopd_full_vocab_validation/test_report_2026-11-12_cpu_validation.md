## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record existing CPU numerical, teacher-routing, and recipe checks. |

# CPU validation

Environment: `/data/ycfeng/envs/nemo-mopd-py313`, Python 3.13.14, Torch runtime `2.13.0+cu130`, uv 0.13.7. Working directory: `/data/ycfeng/sota-engine/RL`. No GPU was used.

All commands use this prefix:

```bash
env UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 \
  PYTHONPATH=/data/ycfeng/sota-engine/RL PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --no-sync \
  python -u -m pytest --noconftest -q
```

| Command suffix | Expected | Actual | Result | Log under `/data/ycfeng/tmp/nemo-mopd-20261112/` |
| --- | --- | --- | --- | --- |
| `tests/unit/algorithms/test_opd_full.py -o cache_dir=/data/ycfeng/tmp/nemo-mopd-20261112/pytest-cache` | Masking, global normalization, reconstruction and loss/gradient assertions pass. | 58 passed, 17 warnings, 90.01 s. | PASS | `opd-full-unit.log` |
| `tests/unit/distributed/test_model_utils.py -k 'student_teacher or reverse_kl_vanishes' -o cache_dir=/data/ycfeng/tmp/nemo-mopd-20261112/pytest-kernel-cache` | Closed-form values and gradients, BF16-to-FP32 normalization and mocked TP reductions pass. | 17 passed, 70 deselected, 9.89 s. | PASS | `opd-kernel-unit.log` |
| `tests/unit/models/policy/test_teacher_worker_group.py -o cache_dir=/data/ycfeng/tmp/nemo-mopd-20261112/pytest-routing-cache` | Routing, payload alignment, packing and configuration assertions pass. | 49 passed, one CUDA-only skip, 25.73 s. | PASS for CPU cases; CUDA case unverified | `teacher-routing-unit.log` |
| `tests/unit/test_config_validation.py -k 'test_shipped_config and mopd-r1distill' -o cache_dir=/data/ycfeng/tmp/nemo-mopd-20261112/pytest-config-cache` | Existing validator accepts the recipe and full-loss settings. | One passed, five subtests passed, three inapplicable subchecks skipped, 432 deselected, 17 warnings, 28.53 s. | PASS | `recipe-unit.log` |

Kernel forward tolerance is `rtol=1e-5, atol=1e-6`; backward reference uses `rtol=1e-4, atol=1e-5`. These existing assertions pass. They do not emit numeric error maxima, so no measured error maxima are claimed. The BF16 normalization case uses the suite's `1e-5` absolute tolerance.

Warnings are existing Pydantic/Ray deprecations. The single skipped routing test requires CUDA. These checks do not establish real multi-rank collective correctness, actual model payload equivalence, trainer/vLLM parity, training health, or save/resume correctness. Those remain required GPU checks.

The input preparation script also passes `ruff==0.9.9 check` and `ruff==0.9.9 format --check`, run through `uv tool run --from ruff==0.9.9` with the corporate index and task cache. `git diff --check` passes. The base environment did not contain Ruff; using the repository-pinned tool version resolved that tooling prerequisite without modifying project dependency files.
