## Modification History

| Date | Change |
| --- | --- |
| 2026-11-13 | Record user-authorized artifact cleanup, exact receipts, retained assets, and verification limits. |

# Current-task artifact cleanup

The user explicitly authorized Codex to clean current-task large, smoke, and unused artifacts. This report covers the cleanup milestone; MOPD's five-teacher training and 200-successful-update acceptance remain incomplete.

| Action | Expected condition | Observed result | Verdict |
| --- | --- | --- | --- |
| Rebuildable UV temporary/download/index cache | Listed temporary/index directories absent; installed environment and reusable cache intact | 94 directories deleted; available space +1,606,836,224 bytes (1.497 GiB), free inodes +29,678 | PASS |
| Duplicate archives | Every ordinary tar member exactly matches its retained extracted copy before archive deletion | 18 archives deleted, 403,136,717 logical bytes; byte-identical extracted copies retained | PASS |
| Obsolete wheel downloads | No active downloader references wheel directory; installed environments retained | 7 files deleted, 1,486,795,398 logical bytes, measured available space +1,044,639,744 bytes (0.973 GiB) | PASS |
| Obsolete cloud conversion caches and three BF16 smoke checkpoints | Every associated job is terminal and personal; inventory unchanged before deletion; approved paths absent afterward | 41 directories deleted, 263,072,167,440 logical bytes (245.005 GiB), all verified absent | PASS |
| Small checkpoint metadata | Preserve all non-distcp files from deleted smoke weights directories with complete byte comparison | 18 files retained in evidence/cleanup-retained-metadata/ | PASS |
| Required models/data/FP16 caches | All six source models, train.jsonl and two required conversion caches remain readable | Six config.json + safetensors inventories present; train.jsonl present; five-teacher and PP4 cache shard inventories present in receipt | PASS |
| Native terminal completion | Personal creator, local NFS source, business receipt, terminal platform/replica/container and launcher | exp-1113-094259-836505: creator i-fengyicheng, NFS100.96.128.195:/data/ycfeng:/data/ycfeng, RJob/replica Succeeded, container Completed/exit0/restart0; launcher exit0 | PASS |

Local cache/wheel deletions measured a combined available-space increase of 2,651,475,968 bytes (2.469 GiB); the archive deletion accounts separately for 403,136,717 logical bytes. These snapshots are local filesystem measurements, not a total disk-space accounting for other sessions. Approximately113GiB was already available before this task's cleanup and is not credited to this action. Sparse wheel downloads explain the logical/allocated difference. UV archive-v0 is mostly hardlinked to installed environments and remains intact. The local task Ray service remains running and its runtime is retained.

Cloud task logical size after cleanup is30,806,354,945 bytes (28.690 GiB). This report establishes namespace deletion and the amount of logical file content removed. It does not measure JuiceFS/S3 physical quota reclamation or trash garbage collection. No shared trash directory was deleted.

## Commands and environment

Local inventory and byte comparison use host Python3, shutil, statvfs and tarfile. The exact deleted paths are in evidence/cleanup_2026-11-13_local_cache.json, cleanup_2026-11-13_duplicate_archives.json and cleanup_2026-11-13_local_wheels.json.

Cloud submission:

```bash
bash /data/ycfeng/tmp/nemo-mopd-20261112/cleanup_personal_env.sh   /data/ycfeng/tmp/nemo-mopd-20261112/submit_cleanup_cloud.py
```

Submission uses local StepMind Python3.10.6/brainpp2.8.6 RJobBackend with personal i-fengyicheng credentials, codesign,0GPU/4CPU/16GiB, one replica and explicit company image. The worker uses uv run --no-sync python in the locally mounted /data/ycfeng/envs/nemo-mopd-py313 environment. The cloud volume is juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp; every cloud mutation is under this task's personal ycfeng prefix. SSH was used for read-only inventory.

Worker command and request are preserved in evidence/cleanup_2026-11-13_cleanup-cloud-request.json. Exact candidates, sizes and the45 associated terminal personal RJobs are in evidence/cleanup_2026-11-13_cloud_manifest.json and cleanup_2026-11-13_cloud_job_states.json. The receipt lists every deletion,18 retained metadata files, original model paths and retained cache shards. Native completion is in cleanup_2026-11-13_cloud_terminal.json and the launcher log. Initial Pod API inspection returned403; the established custom Replica API subsequently supplied the verified owner-bound terminal object. No repeat cleanup job was submitted.

## Retention limits

Old BF16 smoke tensor checkpoints are intentionally deleted under the user's authorization. Recorded measurements, native small training state and numerical reports remain available, but those deleted weights cannot be replayed directly. Future verification uses the retained FP16 caches and new native checkpoints. Source assets, training pools, installed Python environments, unresolved current numerical captures, raw error evidence, trajectories, metrics and small business/runtime/terminal receipts remain intact.
