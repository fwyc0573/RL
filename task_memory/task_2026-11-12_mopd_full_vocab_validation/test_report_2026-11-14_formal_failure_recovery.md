## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Record actual resumed overflow attempts and preserve the eight-unit TQ restore layout for expanded generation. |
| 2026-11-14 | Verify latest-only deletion, native scaler/scheduler state, loss-scale regression, and corrected resume preflight. |
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

## Active native resume: observed state and optimizer attempts

Expected configuration: the original student and five teachers, vocabulary size 151936, 256 fresh trajectories per update, a 65536-token sequence ceiling, 200 successful optimizer updates, native save/resume, and final training/scorer/vLLM response-logprob MAE at most 0.02 nats/token.

Job `exp-1114-021713-015045`, UID `4637305a-eb69-469e-96a7-9e2002abe37e`, was Running with four Running replicas and restart count 0 at 2026-11-13T18:56:10Z (November 14, 2026, 02:56:10 HKT). Personal creator, company image and the local NFS source match the recorded submission. Training launcher 60173, independent reader 59190 and monitor 79639 remain alive.

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Native restore identity | Load the latest formal step10 without resetting the scaler | Trace names the exact retained step10, synchronizes weight version 10, and records restore completion | PASS for the observed boundary |
| Restored loss scale | First attempted update uses the saved scale of 128 | Step11 records `train/loss_scale=128` | PASS |
| First resumed attempt | Record native overflow and count only successful updates | Step11: loss 0.20187801122665405, scale 128, optimizer success 0, no returned grad norm | OVERFLOW; no successful update |
| Second resumed attempt | Observe native backoff and a complete batch | Step12: loss 0.1982083022594452, scale 64, optimizer success 0, 256 samples and 2,009,960 valid tokens | OVERFLOW; no successful update |
| First batch coverage | 256 valid samples, all five teacher groups, correct masks and weight version | 256 samples, 2,275,836 valid response tokens, all five groups, zero reported issues | PASS |
| First batch scorer minus vLLM | Finite identical-token differences with MAE <= 0.02 | MAE 0.0013568862733754916; signed mean -3.453098465679549e-6; p95 0.0055446625; p99 0.0099423647; max 0.3230872154 | PASS for the pre-update scorer/generation mean |
| Second batch scorer minus vLLM | Same criteria and complete group coverage | MAE 0.0013624504824679; p99 0.0100831985; max 0.4857079983; all five groups; zero issues | PASS for the pre-update scorer/generation mean |
| Latest-only checkpoint namespace | One complete current-task checkpoint | At 18:40:45Z, only original step10 remains: 31 files and 24,884,408,375 bytes | PASS |
| Formal successful updates | 200 actual successful optimizer updates | 0/200 after these two resumed attempts | INCOMPLETE |
| Final training/scorer/vLLM comparison | Actual training forward and final saved weights meet MAE <= 0.02 | No final checkpoint or formal successful update yet | UNVERIFIED |

Selected evidence is stored under `evidence/` as `formal-native-resume-1-first-attempt-20261114.json`, `formal-native-resume-1-step12-summary-20261114.json`, both `formal-native-resume-1-step11-parity-20261114.json` and `formal-native-resume-1-step12-parity-20261114.json`, the retention receipt, and the exact-job live-status receipt. No checkpoint tensors were copied.

Exact local inspection commands use the previously prepared platform Python 3.10.6/brainpp 2.8.6 and actor Python 3.13.14/Torch 2.13.0+cu130 environments:

```bash
source /data/ycfeng/tmp/nemo-mopd-20261112/personal_env_20261113.sh
/data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/inspect_formal_native_resume_1_20261114.py
/data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/read_formal_native_resume_1_logs_20261114.py
/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/read_formal_native_resume_1_metrics_20261114.py
```

The head log selector initially returned no selected messages because the tail contained normal rollout, teacher, compilation and communication messages. A bounded query of the actual head tail confirmed active backward without an observed training error in that snapshot. The event `group_discarded` is also emitted when the legacy Gym path successfully commits a group and releases its recovery-ledger entry (`rollout_manager.py`). Its name alone does not establish a dropped training sample. Native recovery reports 512 unfinished siblings restarted, with zero reused or continued. The first 256 published training samples have verified masks and weight versions. This does not classify every later lifecycle event.

First-batch group counts are logic 47, math 56, if 40, stem 46 and code 67. Sequence lengths range from 300 to 23103, so this batch does not exercise the 65536-token ceiling. Prior D evidence covers the separate actual 65536-token execution. The resumed first step took 583.5801 seconds: exposed generation 456.1321 seconds, policy training 96.3912 seconds and scoring 19.9070 seconds. The second total step time was 527.1949 seconds. These observations do not establish final training-gradient health.

## Prepared next segment with more generation replicas

The previous same-layout 11 formal attempts had mean exposed generation time 534.1738 seconds, mean policy training time 75.6929 seconds, and mean total step time 632.0742 seconds. This measured generation cost motivates the next native allocation: 7 nodes with 8 H800 GPUs each, consisting of 8 student GPUs, 32 generation GPUs, five 2-GPU teachers and 6 spare GPUs.

The student remains TP2/PP4/CP1; generation engines remain TP2. All six model paths, full vocabulary, 256 trajectories, the sequence ceiling, FP16 execution and constant scheduler horizon 200 are preserved. The next segment explicitly sets `data_plane.simple.num_storage_units=8` to match native recovery. Its final resolved differences from the current configuration are exactly:

| Field | Current | Prepared next segment |
| --- | --- | --- |
| `cluster.num_nodes` | 4 | 7 |
| `policy.generation.colocated.resources.num_nodes` | 1 | 4 |

The existing native configuration/dtype/optimizer/input-identity preflight returned all seven PASS markers, including the new TQ-count assertion, and exited 0. The existing personal submission wrapper dry-run also exited 0. New training has **not** been submitted. Current training continues until its native publication and terminal boundary. Actual expanded-layout throughput, native restore and post-refit parity remain UNVERIFIED.

```bash
export UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
export CUDA_HOME=/data/ycfeng/toolchain/cuda-13.0.2
export LD_LIBRARY_PATH=/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export TMPDIR=/data/ycfeng/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
timeout 600s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -u /data/ycfeng/tmp/nemo-mopd-20261112/preflight_formal_native_resume_2_generation32_20261114.py
source /data/ycfeng/tmp/nemo-mopd-20261112/personal_env_20261113.sh
/data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/submit_formal_native_resume_2_generation32_20261114.py --dry-run
```

The exact arguments, resolved configuration and two-field difference are archived as `evidence/formal-native-resume-2-generation32-*-20261114.json`. Logs with matching preflight and dry-run names remain under task temporary storage.

## Native TQ restore count checked before submission

The retained step10 data-plane checkpoint sets `storage_saved=true` and contains eight entries in `simple_storage/storage_unit_info.json`. Increasing cluster nodes initially made the existing resolver choose 14 TQ units. The installed actor TQ loader, `AsyncSimpleStorageManager.load_checkpoint`, requires the current count to equal the saved count.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Incompatible native count | Reject saved 8/current 14 before storage RPC | `Storage unit count mismatch: checkpoint has 8, current system has 14.`; zero RPC calls | EXPECTED FAIL |
| Matched count and mapping | Saved 8/current 8 maps all eight stored shards | Eight correct positional restore calls | PASS for the CPU contract |
| Corrected configuration | Preserve eight TQ units while expanding generation | Native preflight and final two-field difference pass | PASS CPU preparation |
| Actual next-segment recovery | Restore weights, scaler, TQ and sampler and continue successful training | Next segment not submitted | UNVERIFIED |

The count test calls the real native loader with the saved manifest. Storage RPC payload loading is replaced to isolate the count and mapping contract. No tensor files are read or written, and the temporary metadata fixture is removed. A first invocation used an incorrect class name and raised ImportError before reaching the loader; that startup failure is not a valid RED result. Its traceback is retained in `formal-native-resume-2-tq-count-contract-import-error-20261114.log`.

```bash
export UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
export TMPDIR=/data/ycfeng/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
timeout 60s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -u /data/ycfeng/tmp/nemo-mopd-20261112/check_formal_native_tq_count_contract_20261114.py
```

Exit 0; the actual expected RED and GREEN results were printed. Source path and mapping evidence are in `evidence/formal-native-resume-2-tq-count-contract-20261114.json`; native checkpoint structure is in `evidence/formal-native-step10-data-plane-structure-20261114.json`. This correction uses an existing config field; native TQ code, checkpoint format and training arithmetic are unchanged.
