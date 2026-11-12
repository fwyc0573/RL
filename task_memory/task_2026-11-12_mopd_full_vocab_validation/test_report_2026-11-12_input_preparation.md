## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record actual input preparation, native terminal state, and corrected metadata checks. |

# Input preparation validation

Status: PASS for input preparation. Runtime configuration, numerical kernels, GPU training, train/inference parity, and checkpoint/resume remain unverified.

## Environment and exact execution

- Repository: `/data/ycfeng/sota-engine/RL`, branch `mopd-sys`, baseline planning commit `843d679b` plus the task-owned recipe and preparation script.
- Source worktree: `/data/ycfeng/steptron/.worktrees/yc-mopd-learning-dynamics-s4-branch`, source commit `6536accb03f0e494f136232d33b49e7f948f42c6`.
- Local submitter: `/data/ycfeng/sota-engine/verl/tests/manual/agent_opd/rjob_run.py`, launched with `/data/ycfeng/tmp/stepmind-env/bin/python`, `STEPMIND_BACKEND=rjob`, and personal credentials loaded without printing them.
- Submit arguments: `--command-file /data/ycfeng/tmp/nemo-mopd-20261112/prepare-worker.sh --gpus 0 --replicas 1 --cpus 8 --memory-gb 32 --name nemo-mopd-inputs-assets-20261112`.
- Image: `hub.i.basemind.com/stepmind/megatron-step:test-optimus-3.24.0-stepccl-0.0.5.post5-vllm-0.11.0.post37-20260817-2733176`.
- Worker: Python 3.10.12 / PyYAML 6.0.2 for this standalone metadata conversion only. The training environment is separately being prepared with Python 3.13.14.
- Worker command:

```bash
python3 tests/manual/mopd/prepare_inputs.py \
  --recipe examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-7n8g-megatron-tp2cp4-pack-single-controller-fullvocab.yaml \
  --source-dir /mnt/codesign-exp/ycfeng/steptron/task_2026-09-26_mopd_learning_dynamics/data \
  --data-manifest /data/ycfeng/steptron/.worktrees/yc-mopd-learning-dynamics-s4-branch/task_memory/task_2026-09-26_mopd_learning_dynamics/s0_common/data_manifest.json \
  --output-dir /mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/data \
  --seed 20260926
```

## Expected and observed results

| Check | Passing condition | Actual | Result |
| --- | --- | --- | --- |
| Asset vocabulary | All six complete token-to-ID, added-token, normalized BPE and decoder mappings agree | All six agreed; 151,387 merges each | PASS |
| Weight files | Nonempty safetensors for each model | Six model paths contain nonempty weights | PASS for existence; tensor integrity is not yet checked |
| Source identity | SHA256 and row count match every manifest input | All 11 train/probe/reserve files matched | PASS |
| Split isolation | No shared source identity or exact prompt between train/probe/reserve | No overlap with identity including optional source subset | PASS |
| Fresh rollout input | Only user messages retained, source metadata preserved | Native Gym tasks contain user input and original task metadata | PASS for conversion; Gym schema still pending |
| Group sampling | All source rows retained and equal group slots | 54,600 unique training source rows; 55,000 slots, 11,000/group | PASS |
| Evaluation | Original 64 probes/group remain separate | 320 rows, 64/group | PASS |
| Completion | Business PASS and native successful exit | RJob and replica Succeeded, exit0/restart0 | PASS |

The IF checkpoint keeps its template in `chat_template.jinja`, not the optional embedded config field. Code/stem/IF post-processors can insert BOS, unlike the student. Every model's template and post-processor is recorded separately; the intended runtime shares student token IDs. This asset check does not establish numerical parity.

The code source contains independently numbered `primeintellect` and `taco` subsets. Omitting subset from the identity falsely reported 21 train/probe overlaps despite distinct prompts. The corrected identity is `(group, source, subset, source_row)`; no training row was removed.

## Native evidence

- RJob: `exp-1112-025041-145201`; replica: `exp-1112-025041-145201-cfcd2084`.
- Creator: `i-fengyicheng`; quota: H800/codesign; 0 GPUs.
- NFS: `100.96.128.195:/data/ycfeng:/data/ycfeng`.
- Node: `gpu-h800-0331.host.platform.shaipower.com`.
- Business exit0: `2026-11-11T18:51:34Z`; native finish: `2026-11-11T18:51:36Z`; launcher observed success and then cleaned up.
- Selected evidence: `evidence/input-preparation.json`, `evidence/input-preparation-platform.json`.
- Cloud outputs: task prefix `/data/train.jsonl`, `/data/probe.jsonl`, `/data/preparation.json` under `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation`.
- Raw worker log: `/data/ycfeng/tmp/nemo-mopd-20261112/prepare-assets-v3-worker.log`; launch log: `prepare-assets-launch.log` in the same directory.

Earlier failures and their original tracebacks are recorded in progress.md and the two RCA directories. The preparation script passed Ruff 0.9.9 lint and was formatted with that version. No GPU or loss/gradient test is counted in this report.
