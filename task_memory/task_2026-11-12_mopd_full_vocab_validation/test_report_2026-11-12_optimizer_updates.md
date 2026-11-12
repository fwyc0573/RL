## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record successful-update telemetry, fault controls, and the native verification boundary. |

# Successful optimizer update observation

Status: metric propagation PASS under CPU fault controls; actual native GPU updates remain unverified.

The 200-update contract requires successful optimizer updates. Inspection found that the split Megatron worker computes `update_successful` but its result, TQPolicy aggregation, and controller metrics do not expose it. The controller's `step_trained` counter alone therefore cannot establish that contract. This is an observation gap; no failed native optimizer update has been observed.

The split worker now reports `optimizer_update_successful`. TQPolicy requires every reported DP replica leader to have succeeded, and the existing step metric aggregation emits 1 or 0. This does not change optimizer, scheduler, counters, tolerances, or failure handling. Consumers without this optional result continue to emit their existing metrics.

## Commands and environments

Working directory: `/data/ycfeng/sota-engine/RL`, branch `mopd-sys`. `uv` is `/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv`. Base environment: `/data/ycfeng/envs/nemo-mopd-py313`, Python3.13.14, Torch2.13.0+cu130. MCore environment: `/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker`, same Python/Torch and pinned MCore0.20.0/Bridge0.7.0.

Subprocesses exclude inherited `OTEL_*`, set `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`, and task-local TMPDIR/RAY_TMPDIR. MCore uses CUDA_HOME/CUDA_PATH `/data/ycfeng/toolchain/cuda-13.0.2` and the existing compatibility library prefix `/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat`.

```bash
uv run --no-sync python -m pytest tests/unit/models/policy/test_tq_policy_placed.py -k reports_any_failed_replica -q --tb=short
uv run --no-sync python -m pytest tests/unit/models/policy/test_tq_policy_placed.py tests/unit/single_controller/test_utils.py -q --tb=short --disable-warnings
uv run --no-sync python -m pytest tests/unit/models/policy/test_megatron_split_state.py --mcore-only -k reenables_forward_pre_hook_after_a_successful_step -p cpu_split_controls -q --tb=short --disable-warnings
```

The last command adds `/data/ycfeng/tmp/nemo-mopd-20261112` to PYTHONPATH. Its temporary `cpu_split_controls.py` mocks only CUDA tensor/zero allocations to CPU and CUDA synchronization; the existing test already mocks optimizer, model, and collectives. This checks success/failure propagation and pre-hook state, not GPU numerics. The committed assertion is in the existing MCore regression and still runs with its ordinary device semantics when selected on a GPU worker.

Pinned Ruff0.9.9 was executed with `uv run --no-sync` using the existing cached executable `/data/ycfeng/tmp/nemo-mopd-20261112/uv-cache/archive-v0/MUGJfPQq178kcgz1/ruff-0.9.9.data/scripts/ruff`, on all five changed Python files. `git diff --check` also passed.

## Results

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Original observation gap | New integration assertion fails before implementation | KeyError `optimizer_update_successful`; 1 failed, 5 deselected in8.66s | Expected FAIL reproduced |
| TQPolicy and step aggregation | Two DP leaders both successful ->1; either failed ->0; existing tests remain passing |38 passed in8.56s | PASS |
| MCore result and pre-hook state | Actual mocked optimizer success flag is preserved for True/False; pre-hook is restored only for success |2 passed,105 deselected in41.12s | PASS under CPU device/collective mocks |
| Ruff check/format and diff | Pinned checks pass on scoped files |All checks passed;5 files formatted; no diff whitespace errors | PASS |
| Actual native optimizer updates | GPU metrics and parameter evidence establish200 successful updates |Both authorized native/TP jobs remain Pending in the latest platform inspection;0 observed updates | UNVERIFIED |

Raw logs are in `/data/ycfeng/tmp/nemo-mopd-20261112/`: `optimizer-update-status-red.log`, `optimizer-update-status-green.log`, and `optimizer-update-status-mcore-cpu-control.log`.

Two earlier MCore invocations are retained and excluded from PASS: without `--mcore-only`, the project deselected the MCore tests (exit5); with that flag on the CPU host, the unmocked CUDA allocation failed before optimizer execution (`optimizer-update-status-mcore-2.log`). The targeted CPU device control resolves the observation check without changing production CUDA behavior. The 107-case MCore suite has not passed here and is not required evidence for this one-field observation change.

No native training-health, parity, refit, resume, or 200-update acceptance item is satisfied by these controls.
