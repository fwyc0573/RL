## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Preserve the native NCCL failure, same-node library control, narrow launcher correction and unverified rerun. |

# Native NCCL startup diagnosis

The first nonpreemptible native control failed before any optimizer update. A same-node control reproduced its initiating NCCL error and identified forwarding of the head's platform network-device names as the cause. Full native verification of the launcher correction is pending.

## Scenario and expected result

- Original RJob: `exp-1112-162755-471889`, personal `i-fengyicheng`, nonpreemptible H800/codesign, 3×8 GPUs, local `/data/ycfeng` NFS source. RJob UID `46f87b35-1378-4293-87f4-12992e7662a4`.
- Runtime: Python3.13.14, Torch2.13.0+cu130, CUDA13.0, vLLM0.29.0; actual vLLM NCCL library2.30.7. The image's NCCL_VERSION environment label2.25.1-1 is not the loaded library version.
- The original native command and exact overrides are retained in `/data/ycfeng/tmp/nemo-mopd-20261112/self-diagnostic-submit-args.json`, the launch log and cloud `command.json`.
- Required startup outcome: every worker uses its provisioned node's network settings; TP2 engines initialize, initial refit occurs, then5successful optimizer updates and same-token logprob checks are inspected. The latter checks remain unverified.

## Observed failure and control

At08:45:17Z (16:45:17HKT), `RayWorkerProc.initialize_worker`, pid3623 on gpu-h800-0187 /10.191.9.11, raised `NCCL error: invalid usage` at `ncclGetUniqueId()`. This precedes communicator creation. ActorHandleNotFoundError cleanup at08:45:18Z is subsequent. Optional nixl_ep import warnings are not shown causal.

The provisioned head uses ens20np0. Generation node runtime-1.json uses ens22np0 and a different NCCL_IB_HCA order. Read-only process inspection found all8outer generation actors using ens20np0 on that node; its actual interface list has ens22np0 and no ens20np0.

The same vLLM NCCLLibrary call ran in the same mounted actor interpreter/library on that exact node, changing only NCCL_SOCKET_IFNAME and enabling WARN output:

| Interface | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| ens20np0, copied from head | Reproduce initiating symptom | Exit1; `Bootstrap : no socket interface found`; same invalid-usage traceback | PASS reproduction |
| ens22np0, provisioned locally | Unique-ID initialization succeeds | Exit0; `NCCL_UNIQUE_ID_PASS` | PASS control |

Exact script: `/data/ycfeng/tmp/nemo-mopd-20261112/probe_live_node_nccl.py`. Its existing-Ray inspection submission used the dashboard `/api/jobs/` entrypoint, `uv run --no-sync python`, no GPU resource request, and pinned base/actor environments. Request receipt: `nccl-inspection-ray-submission.json`; result: [native-nccl-interface-control.json](evidence/native-nccl-interface-control.json). This inspection used the approved existing RJob; it did not create another GPU worker.

Two local GCS attachment probes did not complete: the bounded second attempt reported no matching local node. Removing proxy variables did not settle that access limitation. The dashboard-hosted read completed. A subsequent native forwarding A/B inspection could not be submitted after the dashboard closed; it is UNVERIFIED, not PASS.

## Correction and verification

`tests/manual/mopd/run_on_rjob.py` keeps every node's platform variables in its Ray daemon environment and omits only NCCL_SOCKET_IFNAME and NCCL_IB_HCA from the native head driver's subprocess environment. Native worker snapshots can then inherit the local daemon's values. GPU allocation, collective transport, runtime packages and numerical assertions are unchanged.

CPU checks PASS: Ruff0.9.9 check/format on both launch scripts; `git diff --check`; native MasterConfig/SingleController validation of five-teacher hidden and200-step main candidates after the diagnostic checkpoint change. Exact commands use `/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --no-sync` with UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313. Config command: `python -u /data/ycfeng/tmp/nemo-mopd-20261112/validate_next_native_configs.py`; log `next-native-configs-validation-2.log`.

Replacement `exp-1112-165702-849504` uses the same24GPU native control and corrected launcher. Local session42091 is retained. Actual worker-local settings, engine initialization, refit, updates and parity are still pending. No launcher fix is counted as an end-to-end PASS yet.

## Terminal and RCA coverage

Original job Failed08:55:01Z. Head container Error/exit1/restart0 finished08:54:59Z; generation/teacher replicas became Stopped08:55:04/03Z. Their containerStates retain older running snapshots and do not establish an exit code. The launcher observed failure and cleanup08:55:06Z; no agent manual stop was initiated. Cloud finished.json records failed.

Personal exact object/owner checks and terminal statuses: [self-diagnostic-3-terminal.json](evidence/self-diagnostic-3-terminal.json). Preterminal collector: `/data/ycfeng/tmp/nemo-mopd-20261112/rca-self-native-3`, job plus exact generation replica UID `86bdfd60-104d-46b3-b745-d2c301fae761`, incident08:44–08:51Z. Live/historical Events, incident logs and existing DeepScan reads succeeded; no existing DeepScan; both all-operation audits timed out18s. Historical missing-ConfigMap08:41:26Z predates successful worker startup and the initiating error. Those returned sources do not establish preemption or OOM; incomplete coverage does not exclude unobserved events. Initiating manual-stop actor/reason/request-time remain UNKNOWN; later cleanup is observed separately.

Terminal head traceback and any terminal-near platform events still need reconciliation with retained Ray logs. Native successful updates remain0; five-teacher short run, actual long-context execution,200updates and checkpoint/resume are incomplete.

### Archive follow-up at 17:07 HKT

The bounded read of the exact old job's ray/rank-0 archive found no traceback in worker-*.err; finished.json records failed. The final head-driver exception is not recovered by that archive. This is a collection limitation, not a new cause or a native verification PASS. Local raw summary: /data/ycfeng/tmp/nemo-mopd-20261112/old-native-head-error-summary.json. The corrected job exp-1112-165702-849504 is Starting after09:06:26Z queue approval; its engine, refit, training and parity checks remain pending.
