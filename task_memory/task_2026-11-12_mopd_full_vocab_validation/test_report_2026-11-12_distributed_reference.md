## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Archive the successful H800 TP loss/gradient reference and exact terminal evidence. |

# Distributed full-vocabulary tensor reference

## Result

PASS: four native distributed tensor-reference cases, with zero errors, failures, or skips. These cases validate global vocabulary normalization and gradients for distinct teacher/student logits at TP1 and TP2. They use synthetic vocabulary size 256; actual Qwen model vocabulary, teacher transport, training, and trainer/vLLM parity remain unverified by this test.

## Command and environment

Worker command is preserved in `/data/ycfeng/tmp/nemo-mopd-20261112/tp2-reference-command.json`. The business test command was:

```bash
uv run --no-sync python -m pytest tests/unit/distributed/test_distributed_logprob.py::test_student_teacher_divergence_kernels -q --tb=short --junitxml="$output/pytest.xml" -o cache_dir="$TMPDIR/pytest-cache"
```

- Local personal Python RJobBackend submission; creator `i-fengyicheng`.
- Job `exp-1112-162239-776626`, one replica, two H800 GPUs, codesign billing with preemptible queue placement.
- Node `gpu-h800-0377.host.platform.shaipower.com`; H800 driver 570.124.06, 81,559 MiB per GPU.
- Python `/data/ycfeng/envs/nemo-mopd-py313/bin/python`, 3.13.14; PyTorch 2.13.0+cu130, CUDA 13.0; actual NCCL 2.30.7+cuda13.3.
- Company image digest `sha256:d29740f438d49f543989925ccee2e1e0c6a739f2254c9acc7430363a570a6177`.
- Source mount `/data/ycfeng` from this host; request recorded in `tp2-reference-command.json` and launch log `tp2-reference-launch-3.log`.
- Worker temporary root `/tmp/nrl-$KUBEBRAIN_RJOB_NAME`; retained output under the personal cloud prefix.

## Expected and observed checks

Each case compares divergence and student-logit gradients against direct full-vocabulary log-softmax arithmetic. Student and teacher logits are independently sampled; shape is batch2 × sequence16 × vocabulary256. Existing assertions retain `rtol=1e-4`, `atol=1e-4` for both divergence and gradients.

| Case | Chunk size | Actual outcome |
| --- | --- | --- |
| TP1 reverse KL | 5 | PASS |
| TP1 cross entropy | 5 | PASS |
| TP2 reverse KL | 4 | PASS |
| TP2 cross entropy | 4 | PASS |

JUnit records 4 tests, 0 errors, 0 failures, 0 skips, total 53.510 seconds. Numeric maximum errors are not emitted by the existing test; no invented maximum or relative error is reported. All covered elements met the existing assertions.

## Terminal and artifact evidence

- Platform job Succeeded, finished `2026-11-12T08:24:57Z` (16:24:57 HKT).
- Exact replica Succeeded, owner UID matches job UID; container exit0, restart0, reason Completed.
- Business marker `MOPD_TP2_REFERENCE_TESTS_FINISHED` and all four PASSED cases appear in retained terminal logs.
- Cloud JUnit: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/distributed-reference/exp-1112-162239-776626/pytest.xml`.
- Selected local evidence: `evidence/tp2-reference-pytest.xml`, `evidence/tp2-reference-terminal.json`, `evidence/tp2-reference-terminal.log`.

The prior interrupted TP attempt stays UNVERIFIED; this successful replacement does not erase its agent-stop RCA. No rerun of this passing slice is required.
