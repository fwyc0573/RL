## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record native launch preparation, fault handling checks, and outstanding GPU verification. |
| 2026-11-12 | Reconcile completed locked-wheel/import evidence and actual native submissions at15:58HKT. |

# Native MOPD preparation checks

Status: CPU preparation PASS; native GPU training and all 200-update acceptance criteria remain unverified.

## Environment and commands

Repository: `/data/ycfeng/sota-engine/RL`, branch `mopd-sys`. Native runtime: `/data/ycfeng/envs/nemo-mopd-py313`, Python 3.13.14, Torch 2.13.0+cu130, Ray 2.58.0. The personal StepMind submission client uses the runbook's existing Python 3.10.6 environment at `/data/ycfeng/tmp/stepmind-env`.

Commands ran from the repository. `uv` is `/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv`, with `UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313` and task-local `UV_CACHE_DIR`, `TMPDIR`, and `RAY_TMPDIR`. Test subprocesses exclude inherited `OTEL_*`; values are not archived.

```bash
uv run --no-sync python tests/manual/mopd/run_on_rjob.py --help
PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python tests/manual/mopd/submit_on_rjob.py --help
uv run --no-sync python -m pytest tests/unit/test_config_validation.py -k mopd-r1distill-qwen --disable-warnings -rA
uv run --no-sync python -m pytest tests/unit/experience/test_rollout_generation_failures.py tests/unit/experience/test_rollout_manager.py tests/unit/single_controller/test_rollout_pump.py tests/unit/single_controller/test_rollout_trace.py --disable-warnings -rA
```

Ruff 0.9.9 check and format were invoked through the existing `uv tool run --from ruff==0.9.9` tool environment for the two launch scripts. The exact dry-run argument vector is `evidence/self-diagnostic-submit-args.json`. Its submission command is the same personal interpreter and submit script above plus that vector and `--dry-run`.

## Expectations and observations

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Native RJob/Ray adapter CLI | Parse the intended native entrypoint options | exit 0, help emitted | PASS; no GPU runtime claim |
| Personal submit CLI | Parse explicit resource, mounted environment and image settings | exit 0, help emitted | PASS |
| Final recipe validation | Native schema and SingleController checks accept the complete recipe | 1 test and 5 subtests passed; 3 inapplicable checks skipped | PASS |
| Native timeout/failure/replacement/trace checks | Cancel timed-out work, release reservations, retain successful rows, continue or replace as configured, and preserve systemic failure handling | 263 tests passed in 18.41 seconds | PASS within test coverage |
| Submission dry-run | 3 nodes, 8 H800/node, 32 CPU/node, 192 GB/node, codesign, local `/data/ycfeng` source, explicit company image and both existing RDMA resources | exit 0, request fields match; no job created | PASS; placement/mount/native engine unverified |
| Auxiliary JSON write failure | Report unverified evidence and return without propagating the auxiliary filesystem failure | actual unwritable `/proc` destination printed `MOPD_EVIDENCE_UNVERIFIED` followed by `MOPD_EVIDENCE_FAILURE_ISOLATED`; exit 0 | PASS |
| Locked Triton runtime | Exact SHA256, installs in base and all actor environments, then real required imports | Exact wheel SHA256 verified; installs exit0 in four environments; required base/MCore/vLLM/Gym imports PASS (MCore77.731s) | PASS for installation/import only |

First errors were preserved. `recipe-onpolicy-trace-unit.log` aborted at Ray gRPC `Illegal header key`, matching the inherited OTEL issue diagnosed earlier. Excluding inherited OTEL removed this error; no Ray library code changed. `recipe-onpolicy-failure-unit.log` rejected a 7200-second rollout deadline under the default 600-second stall watchdog. Setting the existing watchdog to 9000 seconds passed in `recipe-onpolicy-failure-unit-2.log`.

All raw logs above reside at `/data/ycfeng/tmp/nemo-mopd-20261112/`. Additional receipts: `native-failure-accounting-unit.log`, `self-diagnostic-launch-dry-run.log`, and `launch-soft-gate-check.log`.

## Selected native behavior and limits

The recipe uses the existing zero-lookahead in-order sampler so parity is measured on the same model version. It enables native rollout trace and training dump. Raw vLLM logprobs and the existing vocabulary divisor override preserve an exact 151936-logit vocabulary under TP2; actual shapes and numerical parity still require GPU evidence.

Formal sample failure handling reuses `async_rl.rollout_failure`: at most 128 data-exhausted prompts over the run, at most one consecutive infrastructure-exhausted prompt, replacement with fresh prompts, and a 100% minimum step batch fraction. Existing bounded retry budgets are preserved. A replacement that cannot restore the configured batch cannot be counted as a successful acceptance update. The 7200-second native rollout and router deadlines permit long contexts; the 9000-second native stall watchdog remains active. The short diagnostic overrides these to 300/300/600 seconds.

Native ledger/trace events link prompt IDs, group IDs, attempts, model versions, environment instances, wall-clock times and outcomes. Existing metrics retain committed/skipped/failure/retry counts; the formal report will reconcile attempt events with the final update and group counts. The full training dump will supply exact token IDs, masks and trainer/generation logprobs. This is preparation evidence, not observed formal coverage.

Auxiliary runtime/command/replica/completion evidence and final Ray log retention use an OSError soft gate. The Ray head rendezvous remains required startup I/O. Native training owns its logger/checkpoint I/O and failure handling. Console output goes directly to platform logs, removing the adapter's cloud stdout file as a dependency of healthy training.

At15:58HKT, native self-control exp-1112-153814-140170 and corrected TP reference exp-1112-155645-933258 are submitted and Pending; both local launchers remain alive. Observed native optimizer updates remain0. Exact runtime versions are in evidence/native-runtime-versions.json; locked install/import receipts are in the task tmp directory. Same-weight initial/refit/resume parity, native teacher payload shape/reference, actual long context, nontrivial parameter updates, native checkpoint/resume, and 200 successful optimizer updates remain pending.
