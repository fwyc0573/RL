## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Verify seven successes, actual72-H800 restore/statistics and synchronized final continuation. |
| 2026-11-14 | Verify six saved successes, latest-only step23 and scheduled72-H800 continuation. |
| 2026-11-14 | Record three formal successes and prepare latest-only120-H800 native continuation. |
| 2026-11-14 | Verify native step18 success and pooled rollout-statistic RED/GREEN on real sample lengths. |
| 2026-11-14 | Record native scheduler RED/GREEN, failure RCA and latest-only retention. |
| 2026-11-14 | Record actual resumed overflow attempts and preserve the eight-unit TQ restore layout for expanded generation. |
| 2026-11-14 | Verify latest-only deletion, native scaler/scheduler state, loss-scale regression, and corrected resume preflight. |
| 2026-11-14 | Recover formal failure, unsuccessful-update metrics and exact rank3 preemption; native state verification remains in progress. |

# Formal MOPD failure recovery

Current status: **7/200 successful formal optimizer updates** are verified on steps 18–24, with no skips in those successful segments. The 72-H800 continuation is Running with nine replicas and restart count zero. Native checkpoint/TQ load and all five full-vocabulary teacher heads are verified. Step 24 uses 256 samples, 2,051,505 valid response tokens and weight_version 23; loss is 0.1714246124, grad norm 0.9186215401 and scale 1. Sampled scorer/vLLM MAE 0.0012360684 meets 0.02, and all 24 pooled median/stddev checks pass in the resumed worker. The only published checkpoint remains step 23 with native Adam 6; step 24's seventh update remains in live memory until the next native save. Final 200-update acceptance and actual-training/scorer/vLLM parity remain UNVERIFIED. Older sections preserve observations at their respective recovery checkpoints.

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
| Third resumed attempt | Observe continued native backoff, five teachers and valid response coverage | Step13: loss 0.1890251785516739, scale 32, success 0, 256 samples, 1,890,880 valid tokens, no returned grad norm | OVERFLOW; no successful update |
| Fourth resumed attempt | Observe native backoff and complete five-teacher sample coverage | Step14: loss 0.19013960659503937, scale 16, success 0, 256 samples, 2,103,101 valid response tokens, no returned grad norm | OVERFLOW; no successful update |
| Fifth resumed attempt | Native backoff, finite loss and complete batch | Step15: scale 8, loss 0.1932196021080017, success 0, 256 samples and 2,240,865 valid response tokens | OVERFLOW; no successful update |
| First batch coverage | 256 valid samples, all five teacher groups, correct masks and weight version | 256 samples, 2,275,836 valid response tokens, all five groups, zero reported issues | PASS |
| First batch scorer minus vLLM | Finite identical-token differences with MAE <= 0.02 | MAE 0.0013568862733754916; signed mean -3.453098465679549e-6; p95 0.0055446625; p99 0.0099423647; max 0.3230872154 | PASS for the pre-update scorer/generation mean |
| Second batch scorer minus vLLM | Same criteria and complete group coverage | MAE 0.0013624504824679; p99 0.0100831985; max 0.4857079983; all five groups; zero issues | PASS for the pre-update scorer/generation mean |
| Third batch scorer minus vLLM | Finite identical-token differences with MAE <= 0.02 | MAE 0.0013389612643139941; signed mean -3.758083974557135e-6; p95 0.0055510998; p99 0.0101517689; max 0.2507520020; all five groups; zero issues | PASS for the pre-update scorer/generation mean |
| Fourth batch scorer minus vLLM | Finite identical-token differences with MAE <= 0.02 | MAE 0.0013298019411208636; p99 0.0101060867; max 0.4830121994; all five groups; zero issues | PASS for the pre-update scorer/generation mean |
| Latest-only checkpoint namespace | One complete current-task checkpoint | At 19:05:36Z, only original step10 remains: 31 files and 24,884,408,375 bytes | PASS |
| Formal successful updates | 200 actual successful optimizer updates | 0/200 after these five resumed attempts | INCOMPLETE |
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

The third resumed batch covers lengths 272–23462; total step time was 502.1598 seconds and policy training 60.3622 seconds. Exact selected metrics, all sample/group parity summaries and current checkpoint inventory are retained as `evidence/formal-native-resume-1-step13-metrics-20261114.json`, `formal-native-resume-1-step13-parity-20261114.json` and `formal-native-resume-1-retention-current-20261114.json`. Platform query at 2026-11-13T19:04:47Z confirms four Running replicas and restart count 0. A head-tail snapshot includes an allocator allocation-failed warning; it contains no fatal training traceback and does not establish a failed job.

## Sixth resumed attempt and localized error-tail inspection

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Native overflow accounting | Record used scale and actual optimizer success | Step16: scale4, loss0.1900885552, success0; formal successful updates0/200 | OVERFLOW / INCOMPLETE |
| Formal batch and teachers |256valid trajectories; five distinct teachers; positive response mask |256samples,2,145,439valid response tokens,5distinct teacher aliases/models;256/256generation successes;zero reported mask/weight-version issues | PASS for this attempt |
| Scorer/generation parity | Finite identical-token MAE<=0.02 |MAE0.0013085080,p990.0098823595,max0.3649930954 | PASS pre-update mean; final training comparison pending |
| Latest-only retention |One complete checkpoint; no additional retained weights |2026-11-13T19:29:45Z:onlystep10,31files/8shards/24,884,408,375bytes | PASS structure; physical quota recovery UNVERIFIED |
| Step15 error tails |Investigate localized maximum without loosening mean threshold |Top code/if/if samples have maxima0.8163185120/0.4316844940/0.2824678421; retained tokenpositions andneighbor deltas | OBSERVED; complete causal explanation UNVERIFIED |

Commands reuse the local personal exact-job inspector, mounted reader and actor-environment metricreader listed above. The namespace inventory ran `timeout 60s ssh -o BatchMode=yes -o ConnectTimeout=10 shai-ycfeng python3 - < /data/ycfeng/tmp/nemo-mopd-20261112/inventory_latest_only_checkpoints_20261114.py`, with output retained in tasktmp/formal-native-resume-1-retention-20261114-current.json. All reads were scoped to this task's personal namespace. No production behavior change orGPUsubmission was made for these inspections.

## Step17 native boundary and second formal resume segment

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Native completion |Publication and terminalstatus agree |exp-1114-021713-015045:step17published;RJob/fourreplicasSucceeded/allcontainerexit0/restart0 at19:42:00Z | PASS native boundary |
| Formal updates |200actualsuccesses, finite gradients |steps11–17allunsuccessful;step17loss0.1972962320/scale2;totalformal0/200 | INCOMPLETE; gradienthealth unresolved |
| Batch andscorer/generation |256validsamples/fivegroups/MAE<=0.02 |2,194,353response tokens,maxlength40,481;MAE0.0013301397,p990.0100550652,max1.0676422119;zeroissues | PASS pre-update mean; tailcause/finaltrainingpending |
| Latest-only retention |Only newestpublishedcheckpoint;oldremovedpromptly |onlystep17,31files/8shards/24,884,408,379bytes;step10absent | PASS structure/publication;backendphysicalreclamationunverified |
| Native optimizer/scaler/scheduler |Restore actualcontinuationstate |8scalers1/growth0/hysteresis−15;Adamgroupsno successstep;schedulernum_steps4352/constantLR1e−6/decay51200;8TQentries/sampler17 | PASS smallstate inspection;GPUrestorepending |
| Nextsegment preparation |Exactmodels/full151936/256/65536/latest-only/native scheduler |7nativePASSmarkers;dry-runexit0;controllerhorizon217/schedulerhorizon200;7nodes8student/32generation/10teacherGPUs | PASS CPU preparation |
| Nextsegment runtime |Personalcreator/localmount/7workers/native restore |exp-1114-034353-134003/UID405b8c82-9752-4a57-9443-b48c1cd32b5e;Starting/7replicas/currentNFS;restore/scale1result not yet observed | IN PROGRESS |

CPUcommands usedPython3.13.14/Torch2.13.0+cu130/UV0.13.7 in theexistingMegatronactorenvironment, withTMPDIR=/data/ycfeng/tmp,offline/no-sync,OMP/OPENBLASthreads1:

```bash
timeout 60s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -u /data/ycfeng/tmp/nemo-mopd-20261112/inspect_latest_formal_native_state_20261114.py
timeout 60s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -u /data/ycfeng/tmp/nemo-mopd-20261112/preflight_formal_native_resume_2_generation32_20261114.py
source /data/ycfeng/tmp/nemo-mopd-20261112/personal_env_20261113.sh
/data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/submit_formal_native_resume_2_generation32_20261114.py --dry-run
/data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_formal_native_resume_2_generation32_20261114.py
```

State-readerexit0;551,606bytesretrieved,33non-tensor storageitems plus smallnativefiles,0model/Adam tensorchunks. Metadata includesserialized scalar tensors, notweights. No newfullfinite scanorweightcopy wasperformed. Preflight/dry-run/launchlogs:tasktmp/formal-native-resume-2-step17-{native-preflight,dry-run,launch}-20261114.log. PlatformPython3.10.6/brainpp2.8.6. Independent0GPUreader/monitor use existing scripts adapted only to the exact newjob and local output paths; they do not controlhealthytraining.


## Current second-resume runtime verification

At 2026-11-13T19:49:08Z (November 14, 2026, 03:49:08 HKT), the personal API confirms the second resume job and seven replicas are Running, restart count 0. Creator, current local NFS source and every full image digest match the request. The 19:49:16Z query returns H800 and `MOPD_PREPARED_INPUTS_CHECK_PASS 41` for all seven replicas. Student NCCL and existing conversion-cache loading are observed; restoration of step17 and the first scale1 update remain UNVERIFIED. Local TensorBoard has zero new scalars; startup does not count as an optimizer update.

Exact inspection commands:

```bash
source /data/ycfeng/tmp/nemo-mopd-20261112/personal_env_20261113.sh
/data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/inspect_formal_native_resume_2_20261114.py
/data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/read_formal_native_resume_2_logs_20261114.py
/data/ycfeng/envs/nemo-mopd-actors/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/read_formal_native_resume_2_metrics_20261114.py
```

Platform/log inspection uses Python3.10.6/brainpp2.8.6. Local metric inspection uses Python3.13.14/Torch2.13.0+cu130. Commands completed with exit0. Evidence: `evidence/formal-native-resume-2-live-status-20261114.json` and `evidence/formal-native-resume-2-selected-logs-20261114.json`. No model, Adam tensor or new checkpoint was copied. Required training launcher46542 and reader62396 are still active.


## Weight-decay scheduler mismatch: native reproduction and correction

Expanded resume exp-1114-034353-134003 failed before any optimizer attempt. Native ConfigContainer computes weight-decay horizon as217×256=55552 while the saved checkpoint has51200. Explicit lr_decay_iters200 preserved only the LR horizon. The native loader rejects that mismatch. The head traceback first appears in the returned incident window at2026-11-13T19:49:04.290258Z; head exits1 at19:49:14Z, platform finishes19:49:16Z. Two replicas Failed and five Stopped. The initiating Python failure precedes native critical-task cleanup. No new checkpoint or optimizer update was produced.

The FP16 recipe now sets the existing native `use_checkpoint_opt_param_scheduler=true`. That selects complete saved scheduler restoration. It preserves the approved LR/WD and counters when the controller extends its attempt horizon; it does not override saved values or reset optimizer/scaler state. Setting scheduler.max_steps200 would fail native validation against train_iters217 and was rejected from source inspection without GPU submission.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Native CPU RED | Reproduce actual saved/configured mismatch | Same55552/51200 assertion at native load_state_dict; exit1 | EXPECTED FAIL |
| Native CPU GREEN | Restore all saved fields and continue | Entire state equals saved; num_steps4352→4608; LR/WD horizons51200; groupLR1e−6 and WD0.01/0 unchanged; exit0 | PASS CPU |
| Corrected full preflight | Configuration, dtype, model identity, topology, retention and actual scheduler load pass before GPUs | Prior seven markers plus scheduler restore/next-step and final preflight marker; exit0 | PASS CPU |
| Personal dry-run | Existing company image/localsource/H800codesign/RDMA/latest-only request |7×8H800,32CPU/768GB per node, native source and config; exit0 | PASS preparation |
| Checkpoint retention after failure | Only latest published boundary |19:55:11Z:step17only,31files/8shards/24,884,408,379bytes | PASS structure; physical quota recovery unverified |
| Actual GPU continuation | Restore step17, then finite scale1 gradients and successful update | Replacement not yet submitted at this checkpoint | UNVERIFIED |

Both CPU numerical runs use the unchanged Python3.13.14/Torch2.13.0+cu130/UV0.13.7 actor environment, offline/no-sync, CUDA_HOME=/data/ycfeng/toolchain/cuda-13.0.2, existing cuda-compat library path, TMPDIR=/data/ycfeng/tmp, OMP/OPENBLASthreads1:

```bash
timeout 60s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -u tests/manual/mopd/check_native_scheduler_resume.py --config examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-4n8g-megatron-tp2pp4cp1-pack-single-controller-fullvocab-fp16.yaml --native-state task_memory/task_2026-11-12_mopd_full_vocab_validation/evidence/formal-latest-step17-native-state-20261114.json --train-iters 217
timeout 60s /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python -u /data/ycfeng/tmp/nemo-mopd-20261112/preflight_formal_native_resume_3_generation32_20261114.py
source /data/ycfeng/tmp/nemo-mopd-20261112/personal_env_20261113.sh
/data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/submit_formal_native_resume_3_generation32_20261114.py --dry-run
```

RED ran before the one-line recipe correction; GREEN ran afterwards. They call the real Bridge scheduler calculation/factory and native MCore loader on recorded small state. CPU optimizer groups have no parameter tensors; this is scheduler validation, not a CUDA/Adam update test. No model/Adam tensor payload or new checkpoint is read/written. Ruff0.9.9 check and format checks pass for the new manual script. Logs:`formal-native-scheduler-resume-{red,green}-20261114.log`, `formal-native-resume-3-step17-{native-preflight,dry-run}-20261114.log` under tasktmp. First appended-preflight invocation failed to import tests.manual before reaching the new check; its error is preserved separately, then absolute-script runpy execution passed. That import failure is not scheduler RED evidence.

Exact failure collector command:

```bash
timeout 60s /data/ycfeng/tmp/stepmind-env/bin/python -u -B /data/ycfeng/stepcode-codex-home/.stepcode/codex/skills/stepmind-gpu-worker/scripts/collect_failure_evidence.py exp-1114-034353-134003 --replica exp-1114-034353-134003-cfcd2084 --expected-creator i-fengyicheng --credential-dir /data/ycfeng/tmp/brainctl-auth-i-fengyicheng --start 2026-11-13T19:48:00Z --end 2026-11-13T19:52:00Z --timeout 20 --output /data/ycfeng/tmp/nemo-mopd-20261112/rca-exp-1114-034353-134003
```

Collector exit1 means two all-operation audits timed out after20seconds; six other query categories succeeded. Incidentlogs return711rows, belowlimit3000, with four scheduler assertions. Exactobjects/ownerUID/node,live/historicalEvents and zeroexistingDeepScan flows are retained. Manualstop actor/reason/requesttime remainUNKNOWN; no post-start preemption/OOM trigger is established by returned records. The old Running snapshot remains historical; latest terminal evidence is separate. Selected evidence: scheduler-failure-rca, terminal, failure-selected-logs, retention-after-resume-failure and scheduler-red-green JSONs under evidence/.


Corrected replacement exp-1114-035754-642199, UIDf7b3451c-6b0d-4687-9b0d-91e6f0ec5e10, was locally submitted at19:57:55Z after the corrected preflight and dry-run passed. At19:58:20Z the exact API reports Starting/sevenreplicas/personalcreator/currentlocalNFS. Traininglauncher80619,reader4033 andmonitor20311 remainalive. ActualGPUrestore/scale1updates remainUNVERIFIED. Launchlog:tasktmp/formal-native-resume-3-step17-launch-20261114.log; arguments/resolvedconfig/live-status/read-onlyreaderrequest under evidence/. No extra weights were written.

## Corrected replacement runtime and checkpoint inventory

At2026-11-13T20:02:12Z,exp-1114-035754-642199 and all seven replicas are Running,restart0. Creator,currentlocalNFS and fullimageID match. Every worker reports H800 and prepared-inputcheckPASS41. Model initialization is observed,zero new optimizer attempts recorded. Actual native GPU restoration and scale1 gradient health remain UNVERIFIED. Startup contains an optional nixl_ep import warning; no terminal workload failure is established by this observation.

At20:03:18Z,the read-only personal namespace inventory confirms only step17,31files,8positive tensor shards,24,884,408,379bytes. Verdict:PASS latest-only structure; physical quota reclamation UNVERIFIED. No tensorchunk,newcheckpoint orHFexport was copied. Exact commands and runtime versions are in the corresponding progress entry; small evidence is under evidence/formal-native-resume-3-{live-status,selected-logs,retention}-20261114.json. All commands exit0. Traininglauncher80619 and independentreader4033/monitor20311 remain alive.

## Actual policy and TQ checkpoint restoration

The20:05:14Z exact head query returns797rows. The native loader confirms successful distributed loading from step17/policy/weights,t0/2,p0/4,iteration0. Native TQ restoration and validation follows,groups0. Teacher initialization then proceeds. Verdict:PASS native policy/TQ load progression and actual recovery beyond the previous scheduler failure. First scale1 gradient/Adam result and successful post-resume updates remain UNVERIFIED. The exact personalPython command and filtered markers are recorded in progress.md and evidence/formal-native-resume-3-restore-selected-logs-20261114.json. Queryexit0;zero new checkpoints or weightcopies.


## First successful formal update and pooled rollout statistics

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Actual post-resume update | Finite loss/gradient, success1/skipped0 | step18:loss0.19038212299346924,pre-clippinggrad1.1475327014923096,scale1,success1/skipped0,LR1e−6 | PASS for this update; formal1/200 |
| Native sample coverage |256validfreshtrajectories,allfivegroups,finitepositivecoverage |256samples/2,019,541response tokens;code42/math42/stem49/logic57/if66;version17;issues0 | PASS step18 |
| Same-token scorer/vLLM | MAE≤0.02,finitegroupvalues,correctmasks/versions | MAE0.0013267311125374486,p990.010008955001831077,max0.2572273015975952,allfivegroupsfinite,issues0 | PASS sampled mean; finalactualtrainingforwardpending |
| Aggregator regression RED | Fail on actual nonlinear summary bug | Native producer→aggregator:[1,2,10] reportsmedian4.333333333333333 instead of2;assertionfailure,exit1 | EXPECTED FAIL |
| Aggregator regression GREEN | Pooledmedian/stddev correct,othersemantics preserved | Existing TestAggregateRolloutMetrics:15PASS,11.91s,exit0 | PASS CPU |
| Original actual-batch replay | Correct overall/five-domain total/response statistics |12distributionsPASS;totalmedian7799.5/stddev6180.540175503602;responsemedian7559/stddev6108.047592100288 | PASS CPU |
| Current workload unaffected | ExistingGPUwriterandlaunchercontinue |20:27:14Z:sevenreplicasRunning/restart0;20:29:13Z:nextbatchTQretrieval | PASS live progression; nextsuccesspending |

The gradnorm is reported before clipping; configuredclip_grad1.0 makes normal clipping of1.1475 expected. No arbitrary loss scaling or scaler reset was applied. The lengthstddevNaN came from averaging singleton sample standarddeviations. Median was also averaged pergroup. The fix reuses native calculate_single_metric on the already flattened histogram and replaces only existing /median and /stddev keys. Mean semantics remain unchanged. The active driver retains its loaded code; subsequentresume loads the fix.

Exact CPU commands (run fromrepo with UV_PROJECT_ENVIRONMENT set to the existing Megatronactorenvironment, TMPDIR=/data/ycfeng/tmp/nemo-mopd-20261112, PYTHONPATH=repo, PYTHONDONTWRITEBYTECODE=1):

```bash
uv run --no-sync python -m pytest --noconftest tests/unit/algorithms/test_grpo.py::TestAggregateRolloutMetrics::test_histogram_statistics_use_pooled_observations -q --disable-warnings
uv run --no-sync python -m pytest --noconftest tests/unit/algorithms/test_grpo.py::TestAggregateRolloutMetrics -q --disable-warnings
uv run --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/check_pooled_step18_lengths_20261114.py
```

Python3.13.14/Torch2.13.0+cu130; effectiveRED/aggregationGREEN/actualbatchGREENexit1/0/0. Two earlier invocations failed before assertions at unrelated Rayfixture setup and are not RED evidence. The class-level reset_env_calls override follows the neighboring pure-tensor test precedent, avoiding module-level remote mockfixtures. No local sharedRayservice orGPUjob was stopped. No weights or checkpoints were read/copied/written by these tests.

Evidence:formal-native-resume-3-step18-{metrics,parity,coverage,pooled-statistics}-20261114.json under evidence/. Runtime inspection uses exact personalPython scripts recorded inprogress.md. Logs remain intasktmp. The15testGREEN and12actualdistributionchecks establish this reporting repair onCPU; they do not establish200successfulupdates orfinalactual-trainingparity.


## Second successful formal update and post-update refit observation

At2026-11-13T20:29:59Z (November14,2026,04:29:59HKT), step19 reports success1/skipped0,loss0.19377940893173218,pre-clippinggradnorm1.1761640310287476,scale1,LR1e−6,256validsamples and2,237,967response tokens. Two formal optimizer successes are now observed; no new skips occurred in this segment.

The second batch uses weightversion18,after step18's successful update/refit. All256samples coverstem50/logic61/if48/code45/math52. Exactsame-token scorer/vLLM comparison:MAE0.0013295876747451956,signedmean−3.4402633044347716e−6,p500.00042831897735595703,p950.005485057830810547,p990.0099334716796875,max0.29484084248542786;allfinite;mask/versionissues0. Verdict:PASS sampled means after formalupdate/refit;longerstability,finalactualtrainingforward and200-successacceptance remainUNVERIFIED.

Commands:existing read_formal_native_resume_3_metrics_20261114.py under UV/no-sync actorenvironment;the existing independent mountedreader summarized the native published step19samples,with receiptfiles10/steps2/failures0. BothbusinessJSONs are retained as evidence/formal-native-resume-3-step19-{metrics,parity}-20261114.json. Ruff0.9.9checkandformatPASS for grpo.py/test_grpo.py after formattingonlythemodifiedtestfile;gitdiffwhitespacePASS. Thepinnedtoolpath is/data/ycfeng/tmp/uv-cache/archive-v0/4Z7oFB0RosbDJ1nO/bin/ruff, invokedthroughuv run --no-sync. No checkpoint orweightcopy was created.


## Third formal successful update and next-segment preparation

Step20 succeeds atscale1/success1/skipped0:loss0.18652436137199402,pre-clippinggradnorm1.1433717012405396,LR1e−6,256validsamples/2,190,389response tokens. Formaltotal3/200. All256sampledcomparisons useweightversion19 and coverfivegroups(stem47/logic63/code44/math51/if51). Scorer/vLLMMAE0.0013174562888005074,signedmean−4.813743656178109e−6,p990.010034031867980964,max0.3331570625305176;allfinite,mask/versionissues0. PASS sampledmean afterthe secondformalrefit;finalactualtrainingforward and200updates remainINCOMPLETE.

The next segment is prepared for120H800,student8/generation96/teachers10/spare6,TP2/PP4/CP1,eightTQunits,256/65536/full151936,controllerhorizon217,save_period10,keep_top_k0/ft_keep_latest_k1,samecheckpointdirectory,checkpoint_must_save_byNone. Nativeconfig/schedulerCPUpreflight,personalbackenddry-run and41file/sixmodel cheapidentity checkPASS. No repeatedfullHFscan,no newcheckpoint/weightcopy/HFexport. Actualthroughput andGPUresume areUNVERIFIED. Atstep19/20 exposedgeneration takes474.10/541.64seconds versus571.23/640.98total; addedgenerationcapacity addresses the observed dominant stage. Runtime remainsH800codesign/personalauth/currentlocalNFS.

Inspectioncommands andversions are unchanged and recorded inprogress.md. Artifacts:formal-native-resume-3-step20-{metrics,parity}-20261114.json andformal-native-resume-4-{capacity-preparation,cheap-inputs,generation96-submit-args,generation96-resolved-config}-20261114.json under evidence/. The initially selected historicalstep10inventoryassertion is an auxiliaryinspectionfailure;healthytraining continues. Exactnextjobname andnativepublication remainpending; do not infer successfulsubmission from prepared files or waiterprocesses.


## Fourth formal update and consistency result

Step21:success1/skipped0,loss0.18771615624427795,pre-clippinggradnorm1.126503348350525,scale1,LR1e−6,256samples/2,054,407response tokens. Formaltotal4/200. Weightversion20 coversfivegroups(code66/stem38/if50/logic63/math39); sampledscorer/vLLM MAE0.0013121362624947182,p990.009957306385040277,max0.30077171325683594,allfinite,issues0. PASS this update and sampledrefitmean; final200/actualtrainingparityINCOMPLETE. Existing native metricreader UV/offline/no-sync command exits0; actualbatchmountedreader hasfoursteps/failures0. No checkpoint, weightcopy orHFexport wascreated.


## Parameter-change comparison baseline with latest-only retention

Expected: the existing initial conversioncache, cast to nativeFP16 as during modelload, equals the still-zero-success step17checkpoint on selectednorm/bias elements. Actual:144,896elements,changed0,maxabsolute delta0,allfinite;PASS. Absolute error0; relative error undefined forzero reference. The1MB-limitedsmallchunk reader loaded935,986bytes into remote memory and savedonlyJSONstatistics. This verifies a baseline for future trainedcheckpoint parameterchange without preservingoldweights; it doesnot prove finalall-parameter finiteness.

Two auxiliaryread errors were diagnosed beforeGREEN: remotePythoncannotdeserialize nativePython3.13 pathlibmetadata; use theexisting localmetadata-reader pattern. OriginalcacheBF16 andnativeFP16 require theexact runtimecast before comparison; same-shape assertion identified theprecision difference. No training behavior or environment waschanged. Exactcommand/version/error paths are recorded inprogress.md;receipt:evidence/formal-step17-initial-cache-small-delta-20261114.json. Reuse only after a newnativepublication; healthytraining andautomaticcontinuation do notdepend onthisauxiliaryverdict.


## Fifth formal update and finite error tails

Step22:loss0.18089325726032257,pre-clippinggradnorm0.9955323338508606,scale1,success1/skipped0,LR1e−6,256samples/2,155,691response tokens. Formal5/200. Weightversion21 coversfivegroups(code47/math53/stem53/if47/logic56). Sampledscorer/vLLMMAE0.0012718353314980088,p990.009858131408691406,max0.7927513122558594,allfinite,issues0. PASS thisupdate andsampledmean; final200updates/actualtrainingforward remainINCOMPLETE.

Exact tailcounts acrossall2,155,691response tokens:1948above0.02(0.0903655%),13above0.1(0.0006031%),1above0.5(0.0000464%). The largest is an interiorlogic token atresponseposition11328,ID2456, scorer−7.29484748840332/vLLM−6.502096176147461. Its full15906-tokenresponse MAE0.0014313242/p990.0093778372;only10tokensabove0.02. Observed masks/versions arecorrect andno batch-wide token shift is established. The exact cause ofthisisolated numericalspike remainsUNVERIFIED; meanerror passes theunchanged0.02contract,withtails reported separately. Tailmeasurement usedboundedremote read-onlyJSON/NumPy,exit0,withnoGPUallocation/weightcopy/cloudwrite. Exactreport:evidence/formal-step22-tail-characterization-20261114.json.


### Formal step23 publication and120-H800 continuation,2026-11-14

| Check | Expectation | Actual | Verdict |
| --- | --- | --- | --- |
| Formal optimizer success | Six successful updates since step17, no skips | Steps18–23 success1/skipped0; each of8 native Adam states step6 | PASS for this segment;200-update contract pending |
| Step23 training health | Finite loss and gradient,256 valid samples | Loss0.1785717756; pre-clipping norm0.9818502069; scale1; LR1e−6;1,945,807 response tokens | PASS |
| Sampled scorer/vLLM parity | MAE≤0.02; finite; correct masks/versions and five teachers | MAE0.0012762114;p990.0097942352;max0.1695599556;issues0 | PASS for sampled parity;actual-training comparison pending |
| Native publication/retention | One newest complete checkpoint | Step23,31files/8tensor shards/24,884,408,489bytes;step17 absent | PASS for logical retention;physical quota reclaim unverified |
| Native recovery counters | Controller/sampler and consumed samples agree | Step23,dispatch23,5888samples,8TQunits;8scalers at1/growth_tracker6 | PASS for saved state;actual new GPU restoration pending |
| Scheduler restore/next step |5888→6144;original decay settings |5888→6144;lr/wd decay51200;parameter-group LR1e−6 | PASS;absolute count error0,relative error0 |
| Selected model changes | Nonzero changes;finite;no extra weight files |4298/144896 norm/bias elements changed;max7.62939453125e−6;935986bytes read only into memory | PASS for selected elements;full final scan pending |
| Corrected continuation check | Native one-element scale tensor accepted at1 | Original scalar assertion FAIL;exact [1.0] assertion PASS across8states | PASS after check-only fix |
|120-H800 submission identity | Personal creator/current local source/one job |exp-1114-051358-764594;UID031a37db-db4c-4ac1-82cc-2cb608a6ee21;i-fengyicheng;100.96.128.195:/data/ycfeng:/data/ycfeng;Pending | PASS for submission;execution unverified |

Commands and environments are recorded in progress.md's matching entry and the exact argument JSON. The continuation uses the same latest-only checkpoint directory, save_period10, keep_top_k0 and ft_keep_latest_k1; it cannot produce additional diagnostic checkpoints or retained HF exports. The previous RJob is Succeeded and finished.json confirms native_execution_succeeded. Final200 successful updates, complete parameter/optimizer finite scan, final three-way parity and completion archive remain pending.


###72-H800 resource continuation replaces unstarted120-H800 request

The120-H800 job stayed Pending with only75 H800 quota, so a prepared72-H800 configuration was selected. CPU native topology/input/scheduler validation and personal dry-run PASS. The exact120-H800 job was stopped while Pending and confirmed Stopped before new submission; no training overlap. Replacementexp-1114-051900-202756 uses9×8H800 and was approved within approximately1.27 seconds of creation. The05:20:12HKT observation verifies9Running replicas,restart0,personal creator,current source and expected image digest. Allocation/identityPASS;actual H800 output,native restoration,subsequent successes,reporting fix and throughput remain unverified. Formal success stays6/200 and step23 remains the only retained checkpoint. Exact args and stop/runtime evidence are under evidence/.


## First actual 72-H800 resumed update and final evidence synchronization

Observed at November 14, 2026, 05:43 HKT. Current writer is exp-1114-051900-202756 / UID ae42005f-d314-4c86-bb10-c6f7bc385b92. Personal creator, all nine worker runtimes, NFS source 100.96.128.195:/data/ycfeng:/data/ycfeng and full d297 image digest agree with the approved launch. Local/worker actor Python is 3.13.14, Torch 2.13.0+cu130 and CUDA 13.0; personal platform client is Python 3.10.6 / brainpp 2.8.6. No new GPU submission was made for these checks.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Native restoration | Exact step 23 load, native TQ restoration, all five teacher heads | Native successfully loaded TP2/PP4 checkpoint; TQ restored/validated; math/code/logic/stem/if heads loaded | PASS |
| Successful resumed optimizer | success 1, finite loss/norm, scale 1, 256 valid samples, positive tokens | Step 24 at 21:42:01Z: loss 0.17142461240291595, norm 0.9186215400695801, scale 1, 256 samples, 2,051,505 response tokens | PASS; formal 7/200 |
| Group/version coverage | All five groups and pre-update weight_version 23 | math 57 / code 48 / stem 49 / logic 53 / if 49; all 256 version 23, mask/version issues zero | PASS |
| Sampled scorer/vLLM response parity | Finite response-token MAE ≤0.02 | MAE 0.0012360683777012415, signed mean −5.294858375691003e−6, p99 0.00977205276489257, max 0.20928828418254852 | PASS for sampled means; exact isolated-tail cause remains pending |
| Pooled median/stddev reporting | Independently recomputed sample lengths equal recorded event values | 24/24 global/per-group total/response-length statistics match native FP32 TensorBoard values; no NaNs | PASS |
| Latest-only native namespace | Only newest complete published checkpoint | 05:43:09 HKT: only step 23, 31 files, eight tensor shards, 24,884,408,489 bytes; step 17 absent | PASS for logical retention; physical OSS quota reclaim UNVERIFIED |
| First resumed batch time | Record actual costs | Total 779.5671s, exposed generation 653.1280s, policy training 93.0572s | OBSERVATION; steady throughput benefit unverified |
| Final evidence copy synchronization | Read metrics after exact reader finished with error-free copy receipt | Five CPU fault injections cover delayed reader, failed reader, wrong UID, copy failure and timeout; all PASS; metric reads follow terminal sync | PASS for auxiliary flow; final runtime pending |
| Final full tensor scan deadline | Permit four sequential two-file waves, each ≤65s | Scanner call permits 300s, above four 65s subprocess bounds; source compiles | PREPARED; final scan has not run |

The reused metric command is the existing monitor's `uv run --offline --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/read_formal_native_resume_5_metrics_20261114.py`, with UV_PROJECT_ENVIRONMENT set to the existing Megatron actor environment. The zero-GPU collector is `/data/ycfeng/tmp/nemo-mopd-20261112/collect_formal_native_resume_5_mounted_20261114.py`, running under exp-1114-051922-692747 / UID a7f3b178-aaa9-41b4-84c6-3e33576307c2. It reads raw samples on the cloud and writes small parity summaries/selected metadata to local mounted temporary storage. No model weights are copied.

Verification commands are the inline `python3 -` scripts recorded in the tool trace: (1) read exact step 24 metrics and mounted parity, recompute Python statistics median/stddev from 256 sampled lengths, encode the reference with `struct.pack('f',...)` and require exact equality; (2) AST-extract wait_for_reader and inject platform states/identity/copy-receipt errors into a local temporary directory; (3) perform a personal bounded read-only SSH inventory of the exact native checkpoint root. All exited zero. Evidence: formal-native-resume-5-step24-{metrics,parity,pooled-statistics}-20261114.json, formal-native-resume-5-retention-current-20261114.json and final-native-parity-reader-sync-check-20261114.json.

The temporary final waiter now awaits the exact existing reader before reading final metrics. Its full-scan timeout increased from 180s to 300s to cover the scanner's existing four waves of 65s bounded reads. Only previously waiting local PID 3756841 was terminated after confirming its exact cmdline and absence of final submission; no GPU, job controller or training launcher was touched. Restart command:

```bash
source /data/ycfeng/tmp/nemo-mopd-20261112/personal_env_20261113.sh
/data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/continue_to_final_native_parity_20261114.py >> /data/ycfeng/tmp/nemo-mopd-20261112/final-native-parity-continuation-20261114.log 2>&1
```

Current final waiter tool session 40518 replaces 56399. It allocates no GPU while the formal writer runs and cannot stop/control it. Final replay retains checkpointing.enabled=false, native loading and in-memory refit. No extra checkpoint or HF export was written.


Task-wide retention follow-up at 05:47:56 HKT: bounded read-only personal SSH inspection of every current-task `*/exp-*/checkpoints` namespace found exactly one native directory, step23, with eight tensor shards / 24,883,134,685 tensor bytes. The query included `tmp_step_*` directories and found none. Historical smoke/diagnostic checkpoint directories are absent. No cloud write or extra checkpoint was created. Evidence: current-task-native-checkpoint-inventory-20261114.json. This proves logical task-wide latest-only retention at that observation, while physical backend quota reclamation remains unverified.
