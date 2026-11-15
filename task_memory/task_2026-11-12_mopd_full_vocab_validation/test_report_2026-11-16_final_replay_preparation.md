## Modification History

| Date | Change |
| --- | --- |
| 2026-11-16 | Validate exact five-domain prompt tokens, current graph/16GPU final replay config, native success accounting, and the final-validation waiter; record step140/Adam122 and health through143. |

# Final replay CPU preparation and continued native progress

Final validation inputs are ready, while formal training remains incomplete. Current native step140 contains Adam122 on all eight optimizer shards and is the only complete checkpoint. Native metrics plus sample summaries through143 show103 consecutive successful updates since the step40 restore, giving125 observed successes. The remaining formal criterion is200 durable successes.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Five source prompt-token seeds | Exact actual IDs, source sample and locator for each original domain | Five native-shaped records fromstep139;7771bytes; prompt lengths430/184/129/44/135 for code/if/logic/math/stem | PASS |
| Existing prompt selector | Reads prepared record unchanged | Native select_prompts returns allfivegroups with exact token IDs/locators/source lengths | PASS CPU |
| Final replay configuration |TP2/PP4;2×8H800; native checkpoint loader/refit; current graph; checkpoint writes off | Non-colocated generation, enforce_eager=False/backendeager, checkpointing.enabled=False | PASS CPU |
| Successful-update reconciliation | Native Adam counts agree with retained actual update history | step130:112successes,113attempts fromstep18, overflowstep25; next horizon218; eight Adam shards112 | PASS CPU with actual saved state |
| Windowed sampler metadata | Use actual sampler semantics | Saved dispatch_index40 while trainstep130; windowed/count check replaces invalid index==trainstep assumption | PASS CPU |
| Final submitter shape | Exact16GPU payload; no premature GPU allocation | Existing submitter --dry-run exits0; preview checkpointstep130; no job created | PASS preparation |
| Final waiter | Exact current writer/observer; no GPUs until native completion/Adam200 | Local PID2019267 alive; first log Running; new GPUs0 | PASS startup; final execution PENDING |
| Current native publication | Only latest complete checkpoint; eight matching Adam counters | step140/Adam122;31files/57,126,362,986bytes; sole checkpoint; no copies/exports | PASS |
| Continued health | Finite full-RKL/grad,256valid samples,fivegroups,age0–5,no skip | All12additional steps132–143 pass;103/103 resumed successes | PASS inspected interval |
| Final acceptance |Adam200, full native tensor finiteness, actual training/scorer/vLLM parity, curves/archive | Not reached or executed | PENDING |

At step143, loss/full-RKL=0.0298014227, gradnorm=0.1113078147, loss scale1, LR1e-6,256valid samples and1,449,000response tokens. Measured total75.75275s, exposed wait0.87193s, policy57.15583s. These observations extend the earlier report's step131 window; they do not replace its retained step130 evidence.

The historical raw trajectories were removed under the user's cleanup authorization, and the observer promptly deletes completed current dumps after compact evidence publication. The first42-second read missed a completed raw batch. A subsequent48-second bounded read captured five exact prompt records fromstep139. Only those small seeds are retained; final responses will use the final reloaded weights. The original completion contract requires fresh five-domain responses and same-weight forwards, not a response generated at a particular earlier controller step.

The first CPU config check incorrectly used attribute access on dictionary-valued checkpointing and failed with AttributeError. Switching the inspection to the actual dictionary interface passes. Both this inspection failure and the missed read window were separate from training. The old final waiter also assumed on-policy dispatch-index semantics; its prepared continuation now checks the actual windowed sampler and deduplicated native-success history. No core sampler code or running configuration was changed.

The final waiter follows only writerexp-1115-193456-282702/UID102dc861-ec0d-4fdd-85bc-dc216ff00572 and observerexp-1115-193548-376103/UID3a0f803d-ff74-48e9-93a4-db4f1aff6cdd. It waits for their successful completion and confirms nativeAdam200 before using the existing full-tensor CPU scan and unsaved16GPU native training/scorer/vLLM replay. A failed/stopped writer or remaining successful-update deficit ends this waiter and requires exact-job RCA or a prepared latest-only continuation; it does not stop healthy training or submit a duplicate writer. Final replay performs no checkpoint/HF export.

Commands:

```bash
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker PYTHONPATH=/data/ycfeng/sota-engine/RL UV_CACHE_DIR=/data/ycfeng/tmp/nemo-mopd-20261112/uv-cache PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat MLFLOW_DISABLE_AGENT_HINT=1 /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/check_fifo_final_replay_20261116.py
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh /data/ycfeng/tmp/nemo-mopd-20261112/check_fifo_final_waiter_20261116.py
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh /data/ycfeng/tmp/nemo-mopd-20261112/dry_fifo_final_replay_20261116.py
# Local detached Popen of the following established personal wrapper; PID2019267, not a remote launcher.
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh /data/ycfeng/tmp/nemo-mopd-20261112/continue_fifo_to_final_native_parity_20261116.py
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 PYTHONPATH=/data/ycfeng/sota-engine/RL UV_CACHE_DIR=/data/ycfeng/tmp/nemo-mopd-20261112/uv-cache /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/read_fifo_observer_metrics_20261116.py
```

CPU replay checks use the installed Megatron actor Python3.13.14/Torch2.13.0+cu130 environment; its group-index fixture is metadata-only. Native counter/waiter checks use the established personal Python3.10.6/brainpp2.8.6 environment. Cheap native publication reads use the existing metadata reader and read5,630,220bytes with zero tensor payloads forstep140. The full final scan and actual GPU replay remain unverified until they execute.

Prepared contract/input/preflight/launch evidence is under `evidence/final-{parity-selected-prompts,fifo-three-way-cpu-preflight,fifo-waiter-preflight,fifo-waiter-launch,checkpoint-three-way-launch-contract}-20261116.*`. Continued health and native state are in `evidence/formal-fifo-{continued-health,latest-native-state}-20261116.json`. Temporary commands/logs remain under /data/ycfeng/tmp/nemo-mopd-20261112/. Keep current writer/observer/waiter processes alive; finish nativeAdam200, final numerical checks and the English archive before declaring task completion.
