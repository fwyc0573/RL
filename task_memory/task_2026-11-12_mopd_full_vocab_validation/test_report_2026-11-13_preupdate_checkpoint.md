## Modification History

| Date | Change |
| --- | --- |
| 2026-11-13 | Record native pre-update checkpoint RED/GREEN and submit the affected GPU save/resume regression. |

# Checkpoint after an FP16 overflow skip

The 56-H800 five-teacher PP4 job `exp-1113-095147-620678` failed at `2026-11-13T02:03:18Z`. Its first business exception, at `02:03:02Z`, is `KeyError: 'master_param'` while the native distributed optimizer serializes the first checkpoint. The DDP repair allowed the first scaled-gradient overflow to reach the native scaler; that update was skipped with `grad_norm=None`. FusedAdam initializes its master weights and Adam moments lazily in its first actual optimizer step. Saving before that step exposed an empty state dictionary.

The NeMo-RL checkpoint integration now calls `initialize_optimizer_state_for_checkpoint()` before native Bridge serialization. It uses the installed FusedAdam's own `initialize_state(parameter, store_param_remainders)` method for empty state only, including the native BF16 remainder rule. It does not perform an optimizer step, alter weights or step counters, overwrite existing state, change loss scaling, or replace native serialization. The existing distributed optimizer and chained optimizer interfaces determine which state is handled. No third-party source or configuration schema changed.

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Original CPU native reproduction | Same missing-master exception before the first successful optimizer step | Real FusedAdam plus native `_get_main_param_and_optimizer_states`, no offload: `KeyError('master_param')`, `distrib_optimizer.py:1233` | RED verified |
| Regression before correction | Same native checkpoint exception | 1 failed, stopped after first failure, 21.12s | RED verified |
| Checkpoint regression after correction | Exact initial master weights / zero moments; support FP16, BF16 and chained optimizer; preserve existing state and step | Four cases pass; pre-update param groups unchanged; second call preserves pointers, values and step=7 | PASS |
| Existing overflow regressions | Native Inf/NaN scaler recovery and BF16/FP32 fatal detection remain intact | Four cases pass; combined eight cases pass in 21.01s | PASS |
| Original CPU reproduction after correction | Native reader returns complete state | Exact FP32 master `[1.25, -0.5]`, zero `exp_avg` and `exp_avg_sq`, exit0 | PASS |
| Native recipe preflight | Five original FP16 teachers, full vocabulary, TP2/PP4, FP32 gradient reduction, original scaler and native save/resume | MasterConfig and native validation pass; initial scale65536 | PASS |
| Launcher and lint | Personal local-source request validates; scoped Ruff0.9.9 check and format pass | Dry-run exit0; all three changed modules pass Ruff check/format and whitespace check | PASS |
| Affected GPU workflow | Complete early save/resume, at least five successful updates, finite parameters and post-update parity | New job `exp-1113-104851-054395` submitted once; actual results pending | UNVERIFIED |

The first GREEN attempt had a test-fixture error: it expected BF16 remainder state while its FusedAdam constructor used the default `store_param_remainders=False`. The corrected fixture explicitly enables the native setting, testing FP16 dtype gating and BF16 remainder storage. No production code change was needed for that assertion failure. The final eight-case result above is the accepted run.

## Commands and environment

Python3.13.14, Torch2.13.0+cu130, actor environment `/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker`, CUDA_HOME `/data/ycfeng/toolchain/cuda-13.0.2`, LD_LIBRARY_PATH `/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat`, TMPDIR `/data/ycfeng/tmp`, OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4. UV executable: `/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv`; set UV_PROJECT_ENVIRONMENT to the actor environment above.

```bash
uv run --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/probe_native_preupdate_checkpoint.py
uv run --no-sync pytest --noconftest tests/unit/models/megatron/test_checkpoint_before_optimizer_update.py -m mcore -q --tb=short
uv run --no-sync pytest --noconftest tests/unit/models/megatron/test_checkpoint_before_optimizer_update.py tests/unit/models/megatron/test_megatron_setup.py::TestCreateMegatronConfigGradOverflow -m mcore -q --tb=short
uv run --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/probe_native_preupdate_checkpoint_green.py
uv run --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/preflight_five_teacher_fp16_resume_pp4.py
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh /data/ycfeng/tmp/nemo-mopd-20261112/submit_final_five_teacher_fp16_resume_pp4_2.py --dry-run
```

CPU commands use timeout60s. `--noconftest` excludes the unrelated autouse Ray runtime fixture. Only the FusedAdam CUDA allocation is redirected to CPU in the new checkpoint tests; real native state creation and checkpoint-state reading remain in use. These checks do not prove actual GPU I/O, distributed communication, scaler serialization or resume continuity.

GPU submission uses the same 56 H800 / codesign shape and local NFS source as the previous diagnostic, the retained conversion root, 16 attempts, 32 fresh trajectories and 8192 ceiling. The formal 200-successful-update / 256-fresh / 65536 run remains unsubmitted. Auxiliary evidence collectors are non-blocking.

## Exact-job failure RCA

The collector verified personal creator and RJob UID `d09dee5f-b749-41a9-b178-f2da02cc686a`, replica UID `a2a3b15c-4660-4cd7-964a-ef36422819c1`, and the first native business exception. Live/historical Events, incident logs and existing DeepScan reads succeeded. All-operation RJob and replica audit queries each timed out after45s; they are failed reads, not successful empty results. The task fail-policy stop at `02:03:17Z` and launcher cleanup at `02:03:19Z` follow the business exception. External initiating stop actor, explicit reason and request time are UNKNOWN; no preemption or OOM conclusion is inferred from exit status alone.

Selected evidence: `evidence/final-five-teacher-fp16-resume-pp4-business-traceback.txt`, `evidence/final-five-teacher-fp16-resume-pp4-failed-objects.json` and `evidence/final-five-teacher-fp16-resume-pp4-rca-coverage.json`. Raw native RED/GREEN and RCA logs remain under `/data/ycfeng/tmp/nemo-mopd-20261112/`; failed-run cloud evidence remains under the task's personal cloud prefix.
