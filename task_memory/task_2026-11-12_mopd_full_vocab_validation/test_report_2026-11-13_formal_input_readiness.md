## Modification History

| Date | Change |
| --- | --- |
| 2026-11-13 | Record full CPU input validation, prepared-input checks, and formal submission preparation. |

# Formal MOPD input readiness

Full CPU input validation: PASS. Corrected formal GPU backward and the 200 successful optimizer updates remain unverified.

The local personal Python RJobBackend submitted `exp-1113-150603-564709`, UID `5d250f1d-0607-44a2-904e-197ceddcc0f8`, creator `i-fengyicheng`, codesign, zero GPUs, eight CPUs, 32 requested GB, company image `hub.i.basemind.com/stepmind/megatron-step:test-optimus-3.24.0-stepccl-0.0.5.post5-vllm-0.11.0.post37-20260817-2733176`. Local NFS source is `100.96.128.195:/data/ycfeng:/data/ycfeng`; the personal cloud volume is dynamically mounted. The platform reports Succeeded, finishTime `2026-11-13T07:10:15Z`; launcher56785 exited0. Python3.13.14/Torch2.13.0+cu130 execute through `/data/ycfeng/envs/nemo-mopd-py313` using `uv run --no-sync`.

Exact worker command:

```bash
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --no-sync python -u tests/manual/mopd/validate_run_inputs.py \
  --config examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-7n8g-megatron-tp2pp4cp1-pack-single-controller-fullvocab-fp16.yaml \
  --preparation /mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/data/preparation.json \
  --source-dir /mnt/codesign-exp/ycfeng/steptron/task_2026-09-26_mopd_learning_dynamics/data \
  --data-manifest /data/ycfeng/steptron/.worktrees/yc-mopd-learning-dynamics-s4-branch/task_memory/task_2026-09-26_mopd_learning_dynamics/s0_common/data_manifest.json \
  --tensor-rows 256 \
  --report /data/ycfeng/sota-engine/RL/task_memory/task_2026-11-12_mopd_full_vocab_validation/evidence/formal-inputs-cpu-validation-20261113.json
```

Expected: every source safetensor loads and contains finite values, unchanged model configurations and file sizes, identical complete student/teacher token mappings, and exact reproduction of every prepared training/probe row. Actual: all six models PASS; 2,034 tensors and 10,662,528,000 elements are finite; all selected model metadata and weight files have whole-file SHA256 records. Rebuilt train55,000 and probe320 rows exactly match, including the original source hashes and cross-split exclusion checks. No native publisher READY evidence was reused. Full values and paths are in `evidence/formal-inputs-cpu-validation-20261113.json`; platform evidence is in `evidence/formal-inputs-cpu-validation-terminal.json`.

Prepared-input startup checks use the recorded file sets, size and mtime_ns. They detect changes without rescanning tensors or datasets. The submission option `--prepared-inputs` requires a PASS CPU report before submission. Every GPU worker performs the inexpensive check; any checker failure, including import or interpreter failure, records UNVERIFIED and leaves healthy native training running. A missing-report direct execution returns0 and prints `MOPD_PREPARED_INPUTS_UNVERIFIED`, as required by the auxiliary soft-gate policy. This does not alter native training or I/O failure handling.

The actual native formal configuration passes in the mounted Megatron actor environment: original six model paths match the CPU report, 151,936 vocabulary, TP2/PP4/CP1, float16 training/generation/teacher payloads, FP32 gradient reduction, 256 trajectories/update, 65,536 sequence limit, native optimizer and data-plane save/resume, max_num_steps200. `grpo.max_num_epochs=null` permits later native attempt-horizon extension if FP16 overflow skips prevent200 actual successes; the constant scheduler retains horizon200 and LR=minLR1e-6. Personal submission dry-run passes with seven replicas, 56 H800, 32 CPUs and768GB/node, original RDMA requests and local source. Preparation commands and logs are `/data/ycfeng/tmp/nemo-mopd-20261112/preflight_formal_fp16_pp4_split.py`, `formal-split-config-preflight.log`, `submit_formal_200updates_fp16_pp4_split.py --dry-run`, and `formal-split-submission-dryrun.log`.

Ruff0.9.9 lint and format pass using the cached exact executable through `uv run --no-sync`. The initial public-index Ruff lookup was canceled after discovering the existing cached executable; the first native config probe selected the base environment and reported missing Megatron, then passed using the installed Megatron actor environment. These CPU-host probe errors did not allocate GPUs or change dependencies.

The final zero-GPU current-identity check `exp-1113-151346-746393` reports PASS at2026-11-13T07:16:07Z for41files across all six model roles. A deliberately changed recorded file size is rejected; no source file is modified and no full tensor scan is repeated. Business output is `evidence/formal-inputs-cheap-check-20261113.json`. The actual generated submission shell continues after an injected checker ImportError, and a FAIL CPU report is rejected before submission, recorded in `evidence/formal-prepared-input-submission-checks.json`. The packed backward repair remains CPU-validated only until the corrected formal job executes.
