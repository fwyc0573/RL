## Modification History

| Date | Change |
| 2026-11-14 | Verify latest-only deletion, native scaler/scheduler state, loss-scale regression, and corrected resume preflight. |
| --- | --- |
| 2026-11-14 | Recover formal failure, unsuccessful-update metrics and exact rank3 preemption; native state verification remains in progress. |

# Formal MOPD failure recovery

Expected: the original student/five teachers, full151936vocabulary,256fresh trajectories/update,65536ceiling,200successfuloptimizerupdates, native save/resume and final response-logprob MAE≤0.02nats/token.

| Check | Observation | Verdict |
| --- | --- | --- |
| Physical runtime | Four8H800nodes, Python3.13.14/Torch2.13.0+cu130/CUDA13.0, expectedd297image/currentlocalNFS | PASS |
| Formal updates | Eleven optimizer attempts; every successflag0; zero actual successful updates | FAIL |
| Last loss/sample coverage | loss0.1954945773 finite;256samples/2,242,562validresponse tokens | PASS for observed attempts |
| Formal gradient/scaler health | No returnedgradnorm/scaler metrics; native state inspection pending | UNVERIFIED |
| Scorer/vLLM same-weight comparison | All11steps/fivegroups MAE0.00128222–0.00135436<0.02; finite logprobs/zeroissues; maxerror~0.63457 | PASS for pre-update sampled means; tails retained |
| Native save/resume | Initial/resume logs andstep1/step10 small state exist; scaler/Adam/scheduler state inspection pending | PARTIAL |
| Formal post-update training/scorer/vLLM | No successful formalupdate or finalcheckpoint | UNVERIFIED |
| Initiating termination | Exactrank3Events at13:37:14Z: CleanPolicyTriggered andPreempted, precedingRaynode death13:37:30Z | PASS: preemption established |
| Requestingpod/manualstopactor | Not returned; all-operationaudit reads timeout15s | UNKNOWN |

PersonalPython command, run fromthe localmachine under existing personal_env_20261113.sh:

```text
/data/ycfeng/tmp/stepmind-env/bin/python -u -B /data/ycfeng/stepcode-codex-home/.stepcode/codex/skills/stepmind-gpu-worker/scripts/collect_failure_evidence.py exp-1113-174329-659007 --replica exp-1113-174329-659007-eccbc87e --expected-creator i-fengyicheng --credential-dir /data/ycfeng/tmp/brainctl-auth-i-fengyicheng --start 2026-11-13T13:35:00Z --end 2026-11-13T13:39:00Z --timeout 15 --page-size 20 --output /data/ycfeng/tmp/nemo-mopd-20261112/formal-packed32-rank3-rca-20261114
```

Python3.10.6/brainpp2.8.6; collector exit1 because both audit reads timed out. Six other source reads completed. Historical replicaEvents returned7/7rows, no additionalpage; incidentlog1row; existingDeepScan0flows. PositiveEvents are stored in evidence/formal-packed32-rank3-preemption-20261114.json, bound to exactjob/replicaUID. Originalincident/Raylogs remain private in tasktmp. PlatformGPUutilization classification is a returnedevent, not independently reconstructed continuous utilization.

Formaljob`exp-1113-174329-659007`, UID`da3fd441-bc51-4d5b-aa27-ad216d7fe445`, failed2026-11-13T13:38:07Z/21:38:07HKT. Source`100.96.128.195:/data/ycfeng:/data/ycfeng`; cloudroot`/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-five-teacher-fp16-resume/exp-1113-174329-659007`. The terminalstatus and all recoveredtrainingmetrics are retained beside this report's selectedRCAreceipt. ContainerGPUallocatorwarnings before step11 do not establish the initiatingtermination. LauncherJobStop13:38:17Z followed thefailure.

Native-state command with the actorprojectenvironment timed out60s/nooutput(exit124). This setup failure is separatefromtraining/RCA; noGPUwasallocated. Furthernative state/scaler/optimizerchecks andresumeverification remainpending. The task remains incomplete.

## Latest-only retention and native recovery prerequisites

The user requires only the latest checkpoint. Exact manifest:`evidence/checkpoint_latest_only_2026-11-14_manifest.json`; direct native retention checks:`evidence/checkpoint_latest_only_2026-11-14_native_retention.json`; deletion receipt:`evidence/checkpoint_latest_only_2026-11-14_cleanup_receipt.json`; exact platform receipt:`evidence/checkpoint_latest_only_2026-11-14_cleanup_terminal.json`.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Native retention | Only latest after both periodic and nonperiodic publication |10synchronous/asynchronous checks atsteps1,10,11,20,21;only latest remains | PASS |
| Delete obsolete current-task checkpoints | Exact terminal personal job inputs; preserve formalstep10 |10directories removed;149,303,776,903logicalbytes; one24,884,408,375byte checkpoint remains | PASS |
| Retained checkpoint integrity | All inventory file sizes/mtimes unchanged;8nonemptytensor shards |31files unchanged;8shards;training_infocurrent_step10 | PASS (structure/change check) |
| Cleanup terminal state | RJobandreplicaSucceeded,exit0 |exp-1114-020444-806306/UIDd647985a-bc4a-4cbf-824a-bfcd26f1a18b;Succeeded;exit0/restart0 | PASS |
| Missing metric reproduction | Actual worker→TQ→stepaggregation lacks loss_scale |KeyError:loss_scale after CPU fixture isolation | EXPECTED FAIL |
| Pre-backoff metric regression | During mockoverflow128→64, record usedscale128/success0/no gradnorm |108state-machine tests passed, including cross-layer regression | PASS |
| Formal native resume preflight | Original6models/5teachers/full151936/256/65536/TP2PP4CP1/200scheduler; latest-only retention |7PASSmarkers;32H800mesh;75minboundary;save_period200/keep_top_k0/ft_keep_latest_k1 | PASS (CPU preparation) |
| Personal submission dry-run | Correctimage/localmount/offlineenv/retentionarguments |exit0;noGPUsubmittedbydry-run | PASS |
| Formal optimizer updates |200actualsuccessflagsandAdamsteps |0successfulupdates before newresume | UNVERIFIED / INCOMPLETE |

Exact cleanup commands from the local host:

```bash
source /data/ycfeng/tmp/nemo-mopd-20261112/personal_env_20261113.sh
/data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/check_latest_only_parent_jobs_20261114.py
/data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_cleanup_latest_only_20261114.py
/data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/inspect_cleanup_latest_only_20261114.py
```

Readonly inventory executedtheexactscript`/data/ycfeng/tmp/nemo-mopd-20261112/inventory_latest_only_checkpoints_20261114.py` through `ssh shai-ycfeng python3 -` with60stimeout; allmutationsranintheabove locallysubmitted0GPUPythonRJobBackendworker. Source100.96.128.195:/data/ycfeng:/data/ycfeng; companyimagedigestd29740f438d49f543989925ccee2e1e0c6a739f2254c9acc7430363a570a6177. Sharedfilesystemfreeblocks do not establish personalquotaavailability or backendtrashreclamation; report logical deletion only.

Exact CPU test environment and commands:

```bash
export UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
export CUDA_HOME=/data/ycfeng/toolchain/cuda-13.0.2
export LD_LIBRARY_PATH=/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export TMPDIR=/data/ycfeng/tmp
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
timeout 600s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -u /data/ycfeng/tmp/nemo-mopd-20261112/check_native_retention_and_loss_scale_red_20261114.py
timeout 600s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -u /data/ycfeng/tmp/nemo-mopd-20261112/run_loss_scale_red_20261114.py
timeout 600s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -u /data/ycfeng/tmp/nemo-mopd-20261112/run_loss_scale_green_20261114.py
timeout 600s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -u /data/ycfeng/tmp/nemo-mopd-20261112/preflight_formal_native_resume_1_20261114.py
```

Versions:Python3.13.14,Torch2.13.0+cu130,pytest9.1.1,UV0.13.7; nativeCUDAlibrariesremaininstalledbutthisCPUstate-machine testdoesnotinitializeCUDA. Greenrunnercalls`pytest.main(['tests/unit/models/policy/test_megatron_split_state.py','-m','mcore','-q'])`;108PASS/33.21s/17existingPydanticwarnings. Ruff0.9.9binary:`/data/ycfeng/tmp/nemo-mopd-20261112/uv-cache/archive-v0/kXL4cKhaO5P88NA2/ruff-0.9.9.data/scripts/ruff`;`check` and `format --check` passforworker,TQpolicy,single-controllerutils,andtestfile. `git diff --check` passes. `systemd-run --user --scope` and `systemd-run --scope` couldnotconnecttobus; boundedCPUcommandsusedtimeout600sinstead,withnoGPUallocation.

FirstselectedteststartupabortedintheglobalRayfixturewithGRPC_CALL_ERROR_INVALID_METADATA; the subsequenttestreachedexplicitCUDAallocationsonCPU. Neither is a validnumericalRED. ExistingpureCPUfixtureandscalar-allocationprecedentsisolateonlythemockedstate-machinefabric. ThevalidREDlogis`loss-scale-red-cpu-isolated-20261114.log`; GREEN:`loss-scale-green-20261114.log`; nativepreflight:`formal-native-resume-1-native-preflight-latest-only-20261114.log`; dry-run:`formal-native-resume-1-dry-run-latest-only-20261114.log`,allunderthetasktmpdirectory.

Nativecheckpointstateevidence:`evidence/formal-packed32-scaler-adam-state-20261114.json` and `evidence/formal-packed32-common-state-20261114.json`. Thirty-tworeadonlybyte-rangequeriesretrieved52,576byteswithoutcopyingtensors. Step1scale65536/growth0/hysteresis1;step10scale128/growth0/hysteresis−8,onall8scalers. Adamparam_groupsaresavedbutnosuccessfulstepstateexists;schedulernum_steps2560,lr_decay_steps51200,constantLR1e−6/WD0.01. These observations explain persistentnativeoverflowbackoff; finiteunscaledgradientsandactualformalupdatesremainunverified.
