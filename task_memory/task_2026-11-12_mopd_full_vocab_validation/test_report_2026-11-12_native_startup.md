## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Preserve native Hydra failure, local repair validation, agent TP interruption, and replacement-job identities. |

# Native startup validation

## Outcomes

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Original24-H800 native job | Ray cluster and native training | Three distinct nodes registered24H800; Hydra then rejected an absent override key before model initialization | FAIL |
| Local failure reproduction | Same Hydra failure as worker | OverridesError reproduced with env.nemo_gym.skip_venv_if_present=true | PASS (reproduction only) |
| Complete fixed worker overrides | Parse, resolve, validate native config | +env.nemo_gym.skip_venv_if_present=true with all native/diagnostic overrides passed MasterConfig and validate_single_controller_config | PASS (CPU preparation only) |
| First green probe | Run config check | Wrong probe module import caused ModuleNotFoundError; no production dependency was missing | FAIL; retained |
| Ruff0.9.9 | Check/format | Both adapter files pass | PASS |
| Original2-H800 TP reference |4PASS/0skip | Agent stopped newly Running worker after a failed state guard; numerical outcome not established | UNVERIFIED / interrupted |
| Replacement submissions | Personal creator, H800/codesign, local NFS and global preemptible queue | Both verified creator i-fengyicheng, preemptible=yes, quotagroup-global-default and100.96.128.195:/data/ycfeng:/data/ycfeng | PASS (submission only) |
| Native training acceptance | Real updates and parity | Successful updates0; no native logprob comparison | UNVERIFIED |

## Exact workload and environment

Original native job: exp-1112-153814-140170, head replica exp-1112-153814-140170-cfcd2084, node gpu-h800-0086.host.platform.shaipower.com. Expected Python3.13.14/Torch2.13.0+cu130 mounted environment, NVIDIA H800/570.124.06. Exact worker command is in /data/ycfeng/tmp/nemo-mopd-20261112/self-diagnostic-launch.log. Native Ray readiness occurred before the Hydra traceback. User command exited1 at2026-11-12T08:19:21Z; RJob Failed08:19:22Z.

CPU validation commands used UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 and /data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --no-sync python. The probes directly call load_config, parse_hydra_overrides, register_omegaconf_resolvers, MasterConfig and validate_single_controller_config against the selected recipe and all diagnostic overrides. Scripts do not launch Ray or access CUDA. Exact outputs: native-overrides-red.log, native-overrides-green.log and native-overrides-green-2.log in the task tmp directory. Adapter path: tests/manual/mopd/run_on_rjob.py; submission path: tests/manual/mopd/submit_on_rjob.py.

Original TP job: exp-1112-155645-933258, replica exp-1112-155645-933258-cfcd2084, node gpu-h800-0401.host.platform.shaipower.com. Worker started08:20:16Z; the agent's rjob_stop.sh call was08:20:26Z; RJob Stopped08:20:30Z. The guard assertion failed because original jobs had left Pending, but the invoking shell lacked set -e. This operational error interrupted the numerical slice. The shell correction is to require set -euo pipefail before dependent commands; no production numerical thresholds or error handling changed.

Exact stop command: bash /data/ycfeng/steptron/playground/users/ycfeng/tools/rjob_stop.sh --creator i-fengyicheng --reason 'Replace never-started MOPD diagnostics with existing preemptible global queue after sustained codesign GPU quota wait' exp-1112-153814-140170 exp-1112-155645-933258. Local stop receipts are in nonpreemptible-stop.log and the shared rjob-stop-audit.log under /data/ycfeng/tmp. The stated pending assumption was stale when that command ran.

## RCA coverage and limits

Selected exact objects, owner UIDs, container status, initiating evidence and source coverage are in evidence/native-startup-and-tp-interruption.json. Two collector commands used the stepmind-gpu-worker bundled collect_failure_evidence.py, exact primary replicas, expected creator,18s query timeout and incident windows08:18–08:21Z /08:19–08:22Z. Raw outputs are /data/ycfeng/tmp/nemo-mopd-20261112/rca-self-native-1/ and rca-tp2-interrupted-2/. All live/historical-event, log and existing-DeepScan reads succeeded. Existing flows total0. Both all-operation audits timed out for each case; missing audit rows are not negative stop evidence. Native manual-stop initiating fields remainUNKNOWN; controller stop labels are cleanup. TP initiating personal command and local reason are observed separately from unavailable server audit data.

## Active replacements

exp-1112-162239-053672:24H800, launcher17124, self-diagnostic-launch-2.log. exp-1112-162239-776626:2H800, launcher28672, tp2-reference-launch-3.log. Exact native arguments: /data/ycfeng/tmp/nemo-mopd-20261112/self-diagnostic-preemptible-args.json. Exact TP submission: /data/ycfeng/tmp/nemo-mopd-20261112/submit_tp2_reference.py; receipt tp2-reference-command.json. Only the selected queue and repaired Hydra prefix differ from the prior controls. Required numerical checks are unchanged. Read-only monitor81077 writes native-job-watch-2/. Continue actual execution verification; these submissions do not complete phaseD.
