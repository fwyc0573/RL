## Modification History

| Date | Change |
| --- | --- |
| 2026-11-15 | Record completion FIFO implementation, native recovery compatibility, CPU regression checks and actual step40 metadata preflight. |

# Trajectory completion FIFO validation

The requested trainer selection is implemented through the existing `windowed` sampler. The only new user selection control is `async_rl.sampler.selection_order=completion_fifo`; the default remains `dispatch_fifo`. The formal MOPD recipe now consumes the first 256 eligible, training-ready groups across dispatch batches and weight versions, with a five-version age limit. CPU behavior and recovery checks pass. H800 throughput and the remaining formal training are not verified by this report.

## Requirements and inspected mechanisms

- `in_order` requires a particular `target_step`; `weight_fifo` waits for the oldest weight cohort. Neither implements the requested selection.
- `ready_first` selects ready groups under a dispatch lookahead gate and does not expire old stragglers. It also requires importance-sampling correction, which the current full-vocabulary objective rejects.
- `windowed` already provides ungated dispatch, buffer backpressure, cross-batch ready selection and real weight-age eviction. Its previous selection order followed reservation indices. The missing control is completion FIFO.
- StepTron comparison sources: `/data/ycfeng/steptron/.worktrees/yc-mopd-learning-dynamics-s4-branch/steptron/core/trainers/fullyasync_policies/train_batch/fifo.py`, `steptron/core/trainers/fullyasync_rollout_runtime.py` and `steptron/generation/staleness_manager.py`. Its FIFO uses the completion queue; its staleness accounting limits backlog rather than strictly expiring each trajectory's age.

Configured behavior:

| Control | Value | Effect |
| --- | --- | --- |
| `async_rl.sampler.name` | `windowed` | No target-step or weight-cohort barrier. |
| `async_rl.sampler.selection_order` | `completion_fifo` | Oldest eligible training-ready publication first. |
| `async_rl.sampler.max_staleness_versions` | `5` | Eligible age is 0 through 5 inclusive; future and older versions are excluded. |
| `async_rl.sampler.sample_freshest_first` | `false` | Preserves completion FIFO; enabling it simultaneously is rejected. |
| `min_groups_for_streaming_train` | `256` | Waits for 256 eligible groups before training. |
| `max_inflight_prompts` | `256` | Caps concurrent prompts. |
| `max_buffered_rollouts` | `1536` | Caps unconsumed groups at six full updates. |
| Drop budgets | both `0` | Required by the existing validator for unstamped samplers; an exhausted prompt fails rather than silently shrinking the batch. |

Canonical TQ publication alone is insufficient: `commit` and `commit_finalized` assign `replay_ready_order` only after all required teacher enrichment succeeds. FIFO order is carried in each group's metadata and included in the existing replay manifest digest. Training claims retain it for periodic snapshots. Restore validates integer/nonnegative/unique ordinals; legacy checkpoints without ordinals retain serialized group order, log the unavailable historical order, and append subsequent completions after the restored groups.

Native recovery now allows the requested `in_order -> windowed` switch. All other sampler-name changes remain rejected. Trainer step/version, epoch, partition, schema, mode, manifest/group count and teacher numbering remain checked. Recovery refuses the switch if there are pending target-step shortfalls or allowed-short steps at/after the restored trainer version, and restores the new sampler's unstamped runtime capability. Historical target stamps remain available for ownership recovery.

## Environment and commands

Host repository: `/data/ycfeng/sota-engine/RL`, branch `mopd-sys`, implementation parent `f2061fcd`. CPU-only execution; no GPU RJob submission in this step. Python 3.13.14, Torch 2.13.0+cu130, offline UV 0.13.7, Ruff 0.9.9. The test process address-space limit is 30 GiB. Temporary files/logs use `/data/ycfeng/tmp`.

The exact runner is `/data/ycfeng/tmp/nemo-mopd-20261112/run_fifo_cpu_checks_20261115.sh`. Its environment and invocation are:

```bash
export TMPDIR=/data/ycfeng/tmp
export HF_HOME=/data/ycfeng/tmp/nemo-mopd-20261112/hf-cache
export UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
export UV_CACHE_DIR=/data/ycfeng/tmp/nemo-mopd-20261112/uv-cache
export PYTHONPATH=/data/ycfeng/sota-engine/RL
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat
timeout 180s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -c 'import resource, runpy; resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3)); runpy.run_module("pytest", run_name="__main__")' --noconftest "$@" -q --tb=short -p no:warnings -o cache_dir=/data/ycfeng/tmp/nemo-mopd-20261112/pytest-fifo-cache
```

Exact check commands:

```bash
bash /data/ycfeng/tmp/nemo-mopd-20261112/run_fifo_cpu_checks_20261115.sh tests/unit/single_controller/test_setup.py -k test_restores_in_order_checkpoint_into_windowed_sampler
bash /data/ycfeng/tmp/nemo-mopd-20261112/run_fifo_cpu_checks_20261115.sh tests/unit/single_controller/test_checkpointing.py -k test_in_order_to_windowed_restore_handles_target_step_state
bash /data/ycfeng/tmp/nemo-mopd-20261112/run_fifo_cpu_checks_20261115.sh tests/unit/single_controller/test_tq_replay_buffer.py tests/unit/single_controller/test_sampler_interface.py tests/unit/single_controller/test_setup.py
bash /data/ycfeng/tmp/nemo-mopd-20261112/run_fifo_cpu_checks_20261115.sh tests/unit/single_controller/test_checkpointing.py tests/unit/single_controller/test_checkpoint_recovery_matrix.py tests/unit/single_controller/test_checkpoint_borrow_restore.py
bash /data/ycfeng/tmp/nemo-mopd-20261112/run_fifo_cpu_checks_20261115.sh tests/unit/test_config_validation.py -k 'test_shipped_config and (mopd-r1distill or single-controller-streaming2) and not tp4cp1'
```

The two focused resume commands were run before their respective fixes and then included in the passing suites. `tp4cp1` excludes an unrelated untracked recipe; no tracked inheritance member used by this task was excluded. `--noconftest` avoids unrelated distributed fixture initialization. Required test-module fixtures remain active. The first full core suite stopped after 287 passing tests because `HF_HOME` was unset while resolving an existing Qwen3 config. Adding the required environment variable, without changing the test or recipe, produced the complete 609-test PASS.

For FIFO's original RED, the same offline runtime ran `test_tq_replay_buffer.py -k TestCompletionFifoSelection` with a 120-second timeout. Before the FIFO implementation it selected `second` before `third` despite reverse completion order. The captured result is `1 failed, 115 deselected` in 15.25 s. After the implementation, the original case passed in 16.45 s and is included in the complete suite below. Earlier 60-second import attempts timed out and provided no behavioral verdict.

## Results

| Check | Expected result | Observed result | Verdict |
| --- | --- | --- | --- |
| Completion ordering RED | Reverse completion must differ from dispatch selection. | Actual `second` then `third`; expected `third` then `second`, reproducing the omission. | Expected FAIL before fix |
| Native metadata-switch RED | Existing validation blocks `in_order -> windowed`. | `sampler_name` mismatch; 1 failed in 12.19 s. | Expected FAIL before fix |
| Runtime capability RED | Old stamped capability must not override the new sampler. | Restored capability was `true`; 1 failed in 13.39 s. | Expected FAIL before fix |
| Core sampler/replay/setup | Full count, FIFO tail, cross-target/version selection, teacher-enrichment boundary, age boundary/future exclusion, ownership, claims and legacy/integrity restore pass. | 609 passed in 73.79 s. | PASS |
| Checkpoint/recovery | Native snapshots, claims, all built-in recovery scenarios, sampler switching and pending-shortfall rejection pass. | 342 passed in 32.17 s. | PASS |
| Shipped configuration validation | Modified recipe, all tracked FP16/PP4 inheritance members and existing streaming configs validate. | 7 passed, 35 subtests passed; 21 inapplicable checks skipped, 431 cases deselected; 17.42 s. | PASS |
| Scoped lint/format | Ruff 0.9.9 and whitespace checks pass. | Import ordering fixed in one test file; three Python files formatted; final 9-file check clean. | PASS |
| Actual step40 metadata preflight | Original teacher order, replay digest, sidecar checksum/schema/group count, no pending-shortfall incompatibility, latest-only publication. | One retained step40; replay groups0, recovery-ledger groups512; 703,834 metadata bytes read, weight tensor payloads0. | PASS within CPU metadata scope |
| Native scheduler next tick | Restore existing sample horizon and preserve LR/WD. | 10240→10496 consumed samples; LR1e-6/WD0.01; native constant horizon200. | PASS |
| H800 workflow and throughput | Complete actual GPU restore and compare the same workload. | No GPU run in this step. | UNVERIFIED |

Logs:

- `/data/ycfeng/tmp/nemo-mopd-20261112/fifo-selection-red-profile-20261115.log`
- `/data/ycfeng/tmp/nemo-mopd-20261112/fifo-selection-green-20261115.log`
- `/data/ycfeng/tmp/nemo-mopd-20261112/fifo-resume-red-20261115.log`
- `/data/ycfeng/tmp/nemo-mopd-20261112/fifo-ledger-resume-red-20261115.log`
- `/data/ycfeng/tmp/nemo-mopd-20261112/fifo-core-tests-20261115.log` (missing environment variable; superseded)
- `/data/ycfeng/tmp/nemo-mopd-20261112/fifo-core-tests-with-hf-home-20261115.log`
- `/data/ycfeng/tmp/nemo-mopd-20261112/fifo-checkpoint-tests-20261115.log`
- `/data/ycfeng/tmp/nemo-mopd-20261112/fifo-shipped-config-tests-20261115.log`
- `/data/ycfeng/tmp/nemo-mopd-20261112/fifo-native-preflight-20261115.log`

Scoped verification used `/data/ycfeng/tmp/nemo-mopd-20261112/uv-cache/archive-v0/kXL4cKhaO5P88NA2/ruff-0.9.9.data/scripts/ruff`: `check --select I --fix`, `format`, `check` and final `format --check` on the nine modified Python files, followed by `git diff --check`.

## Actual checkpoint and continuation preparation

At 2026-11-15 02:37:41 HKT, read-only SSH as `i-fengyicheng` verified the published native checkpoint through `/mnt/ws-jfs/ycfeng`, the remote mount equivalent of:

`/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-five-teacher-fp16-resume/exp-1113-174329-659007/checkpoints/step_40`

Only six small metadata/sidecar files were read: training info, native TQ metadata, replay metadata, rollout recovery, spare prompts and the eight-unit storage layout. No model/Adam tensor payloads were read or copied. The actual metadata was passed through the native setup validator and replay loader using empty temporary local format markers and in-memory decoding; no recovery checkpoint was written. Sidecar checksum, schema8, group count512 and pending-step checks also passed. This does not load the cloud TQ tensor payload or start GPU actors.

`preflight_formal_step40_fifo_candidate_20261115.py` runs under the environment above through offline `uv run`, a 120-second timeout and the same 30 GiB process limit. Its exact command and retained findings are in [evidence/formal-step40-fifo-cpu-preflight-20261115.json](evidence/formal-step40-fifo-cpu-preflight-20261115.json). The CPU preparation retains original student/five-teacher identities, full151936 vocabulary, FP16 payload/training, TP2/PP4/CP1, 256 trajectories/update, 65536 sequence limit, eight TQ storage units and latest-only native recovery.

Prepared artifacts:

- [Continuation arguments](evidence/formal-step40-fifo-candidate-submit-args-20261115.json)
- [Resolved configuration](evidence/formal-step40-fifo-candidate-resolved-config-20261115.json)
- [Actual native metadata/config preflight](evidence/formal-step40-fifo-cpu-preflight-20261115.json)

The candidate remains `gpu_submission_ready=false`: it still carries the former72-H800 role placement. Graph capture/parity, improved teacher placement, actual TQ/model/optimizer resume and same-workload timing must be checked before promotion. Native durable progress remains22/200; 178 further successful updates remain, and final Adam must be200. A controller limit of218 is a continuation attempt horizon, not proof of178 future successes.

NeMo-Gym does not currently cancel old in-flight work at refit. The existing age window excludes those results and evicts their ready rows after they finish; this can still waste rollout/teacher work. Legacy nonempty replay checkpoints cannot reconstruct historical completion ordering. No runtime speedup, final parameter finite check, final parity or overall completion is claimed here.
