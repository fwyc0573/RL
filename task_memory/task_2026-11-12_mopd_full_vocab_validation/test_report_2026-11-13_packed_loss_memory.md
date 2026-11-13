## Modification History

| Date | Change |
| --- | --- |
| 2026-11-13 | Recover the second formal failure and validate the packed-loss gradient allocation repair on CPU. |

# Packed MOPD loss memory validation

Status: CPU regression PASS. Original formal GPU job FAIL. Corrected formal GPU backward remains pending. The full 200-successful-update contract is unfinished.

## Exact failed scenario

RJob `exp-1113-121031-998701`, UID `2003e09a-e896-4ecd-ac29-453b3dcaf327`, creator `i-fengyicheng`, H800/codesign, 7 nodes / 56 GPUs / 32 CPU and 768 requested GB per node (750 Gi platform limit). Local source is `100.96.128.195:/data/ycfeng:/data/ycfeng`. Actual runtime is Python3.13.14 / Torch2.13.0+cu130 / CUDA13.0 / H800 / driver570.124.06. Exact command and recipe are in evidence/formal-200updates-fp16-pp4-memory-submit-args.json.

Native backward raises CUDA OOM for MegatronPolicyWorker rank7 on `10.191.9.69` (`gpu-h800-0200.host.platform.shaipower.com`): request18.55GiB, free18.38GiB, allocated42.97GiB, reserved-unallocated14.64GiB, total79.19GiB. Native worker container exits1/Error/restart0 at06:31:17Z. RJob task-fail-policy stopTimestamp06:32:24Z and finishTime06:32:26Z precede launcher cleanup stop14:32:33HKT. This failure is distinct from prior CPU-memory pressure. No manual initiating stop actor/reason/request time is established; those fields are UNKNOWN.

The bounded personal Python collector verified the exact replica UID13d66ed5-c4bd-4bf2-bfb8-44fde6890308 and its owner. It queried live Events, historical Events, incident logs06:29–06:33Z, all-operation audits, and existing DeepScan. All queries succeeded with complete pagination: historical RJob28/replica1, audit RJob62/replica1, live Events0/0, DeepScan0. Historical startup configmap error at06:12:40Z predates the running workload and is not the backward initiating error. Audit rows do not supply a manual initiating reason; controller cleanup is not assigned as the CUDA failure cause. Raw scoped responses remain in `/data/ycfeng/tmp/nemo-mopd-20261112/formal-memory-cuda-rca-20261113/`.

## Recovered business progress

Command: `/data/ycfeng/envs/nemo-mopd-py313/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/read_formal_fp16_pp4_memory_metrics.py`.

TensorBoard reconstruction yields0 recorded optimizer attempts and0 successful updates, with no recorded train loss/grad norm. Cloud inspection confirms no checkpoint. Unpublished `train_data_step1.jsonl.part-0` contains256 trajectories /2,157,723 total tokens, length303–20,185. Group counts: math60/code56/stem53/logic36/if51. These samples demonstrate rollout/scorer coverage but are not an accepted optimizer update.

For2,086,520 response-loss tokens, scorer-minus-vLLM MAE0.0013139912591959128, signed mean-4.454962343792446e-7, p950.005482196807861328, p990.009916305541992188, max0.2215239405632019, allfinite. The mean meets0.02; tails are reported without declaring actual training-forward parity. Evidence: evidence/formal-memory-first-batch-summary.json.

## Hypotheses and repair

Before modifying behavior, rank the hypotheses: (1) full packed-logit gradient peak, (2) tensors retained across multiple microbatches, (3) allocator fragmentation. Inspect the native wrapper's per-sequence narrow calls. Each narrow backward produces a full packed gradient, then accumulates it into the original logits gradient. The CPU regression exercises the real SequencePackingLossWrapper, hidden-state teacher reconstruction, distributed reverse-KL kernel with singleton collectives, teacher routing, masks, padding, and normalization.

The repair replaces independent narrow operations with one torch.split for the ordinary logits path. PyTorch joins all sequence gradients once. The existing shorter fused-linear-logprob path retains its clamped slicing. No objective, dtype, vocabulary, sequence limit, teacher route, or global batch changes.

RED command:

```bash
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 \
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --no-sync \
pytest --noconftest -q tests/unit/algorithms/test_opd_full.py::test_packed_full_loss_avoids_repeated_full_logit_gradients
```

Expected: no repeated full-sized slice-gradient buffers, and loss/gradients within rtol1e-5/atol1e-6 of direct log-softmax reverse KL. Observed RED: full-size slice backward allocated111024bytes for37008bytes of packed logits (3x); loss and gradient numerical checks already passed. Exact symptom assertion failed,1failed/13.65s, test call0.77s. An earlier ordinary pytest invocation entered the repository-wide Ray fixture and was interrupted; --noconftest executes this CPU seam without that unrelated fixture.

GREEN command:

```bash
PYTHONPATH=/data/ycfeng/sota-engine/RL \
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 \
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --no-sync \
pytest --noconftest -q --disable-warnings -o addopts='' \
tests/unit/algorithms/test_opd_full.py \
tests/unit/algorithms/test_single_forward_grpo.py \
tests/unit/algorithms/test_sequence_packing_fusion.py \
-k 'not test_sequence_packing_fusion'
```

Observed118PASS/47deselected/17warnings/14.92s. The new memory assertion and direct mathematical loss/gradient checks pass. No GPU result is inferred from CPU profiler. The CPU profiler reports gpuGetDeviceCount code35 on this CPU host; CPU checks finish successfully. Distributed GPU fusion cases were not selected by this command.

## Remaining validation

The new user input-preparation rule requires full validation before allocating training GPUs when native publisher READY evidence is unavailable. RJob exp-1113-150603-564709 is a0GPU CPU validation job for all six original HF sources and complete dataset reproduction. Its result is now PASS:2034tensors/10662528000finiteelements andexact55000train/320proberows; platformSucceeded07:10:15Z. See test_report_2026-11-13_formal_input_readiness.md. Worker identity checks are nonblocking auxiliary checks; native training and I/O retain their own errors.

After CPU input readiness, rerun the unchanged formal scenario with the repaired wrapper, inspect first backward health and native checkpoint/resume, then finish200actualsuccessfulupdates and final-checkpoint three-way parity. Original65536long-context and five-teacher D results remain recorded PASS; this repair still needs the full formal GPU scenario.
