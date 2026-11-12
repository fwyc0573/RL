## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record the completed six-model HF payload/loss/gradient control and native terminal evidence. |

# Initial model reference validation

**Result: PASS within the HF initial-checkpoint scope.** Megatron, vLLM, payload transport, optimizer updates, refit and resume remain unverified.

## Execution

- Repository commit: c2cfe5f0, plus the tested `tests/manual/mopd/check_model_payloads.py`.
- Local launcher: personal StepMind Python RJobBackend; creator `i-fengyicheng`; H800/codesign; one GPU, 8 CPU, 48 GB; source NFS `100.96.128.195:/data/ycfeng:/data/ycfeng`.
- Job: `exp-1112-125639-935155`; replica: `exp-1112-125639-935155-cfcd2084`; node: `gpu-h800-0582.host.platform.shaipower.com`.
- Image digest: `sha256:d29740f438d49f543989925ccee2e1e0c6a739f2254c9acc7430363a570a6177`; image label: `hub.i.basemind.com/stepmind/megatron-step:test-optimus-3.24.0-stepccl-0.0.5.post5-vllm-0.11.0.post37-20260817-2733176`.
- Worker runtime: Python 3.13.14, Torch 2.13.0+cu130, CUDA 13.0; H800 81559 MiB, driver 570.124.06. Mounted base environment: `/data/ycfeng/envs/nemo-mopd-py313`; CUDA compatibility library: `/data/ycfeng/envs/cuda-compat-13-0-580.178.04/usr/local/cuda-13.0/compat`.
- Exact shell wrapper: `/data/ycfeng/tmp/nemo-mopd-20261112/model-payload-worker.sh`; launcher log: `/data/ycfeng/tmp/nemo-mopd-20261112/model-reference-launch.log`.

Executed business command (wrapper selects the above environment and local caches):

```bash
uv run --no-sync python -u tests/manual/mopd/check_model_payloads.py \
  --recipe examples/configs/recipes/llm/mopd-r1distill-qwen-1.5b-7n8g-megatron-tp2cp4-pack-single-controller-fullvocab.yaml \
  --probe-data /mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/data/probe.jsonl \
  --output /mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/reference/hf-model-payloads.json \
  --response-tokens 16 --max-prompt-tokens 512
```

## Expectations and actual results

The control uses the shortest complete probe prompt in each group, rendered once with the student tokenizer, and a greedy student response. Each model scores identical token IDs; BF16 HF SDPA direct output and the native hidden-state/LM-head reconstruction are compared. All six model vocabularies contain 151936 logits. Every selected-token and full-distribution reconstructed/direct logprob error statistic (signed mean, MAE, p50, p95, p99, maximum) is exactly zero: PASS, including selected-token MAE <=0.02 nats/token.

The native full-vocabulary CUDA loss and logit gradient are compared with direct FP32 autograd on the same logits. Loss assertions: rtol=1e-5, atol=1e-6; gradient assertions: rtol=1e-4, atol=1e-5. Both direct and reconstructed teacher payloads pass for every model; their measured loss and gradient results agree.

| Model | Prompt / response tokens | Reverse KL | Absolute loss error | Relative loss error | Maximum gradient element error | BF16 / FP32 head full-distribution logprob MAE |
| --- | --- | --- | --- | --- | --- | --- |
| student | 42 / 16 | 0 | 0 | undefined (reference 0) | 1.09430403e-08 | 0.0506301485 |
| math | 42 / 16 | 0.0367072821 | 1.86264515e-08 | 5.0743231e-07 | 1.42608769e-08 | 0.0404233299 |
| code | 323 / 16 | 0.166921556 | 2.98023224e-08 | 1.78540917e-07 | 1.49011612e-08 | 0.0504580587 |
| stem | 31 / 16 | 0.0563248023 | 5.21540642e-08 | 9.25952872e-07 | 2.98023224e-08 | 0.0374853984 |
| logic | 22 / 16 | 1.12548423 | 2.38418579e-07 | 2.11836401e-07 | 1.8189894e-12 | 0.0437980406 |
| if | 18 / 16 | 0.0544073805 | 2.23517418e-08 | 4.10821701e-07 | 2.98023224e-08 | 0.0353089906 |

Maximum absolute loss error: 2.384185791e-7. Maximum gradient element error: 2.980232239e-8. All native logit gradients are finite. Self-distillation yields native loss and gradient norm zero; direct FP32 reference gradient norm is 2.21708234e-8, within the assertion tolerance. These are gradients with respect to logits, not the trainer's parameter gradient norm.

The BF16 versus FP32 LM-head distribution MAE is 0.035309–0.050630 nats/token. This measures all vocabulary entries and a different head precision; it is not trainer/vLLM selected-token parity and does not modify the 0.02-nat/token acceptance ceiling.

## Platform and artifacts

Business completion marker: `MODEL_PAYLOAD_REFERENCE_PASS`. Native RJob and replica: Succeeded. Container exit: 0; restart count: 0; finished November 12, 2026 at 05:05:50 UTC (13:05:50 HKT). Local launcher also completed normally. Direct core Pod read returned HTTP403; exact Replica CRD supplied the container state instead.

- Retained numerical evidence: `evidence/hf-model-payloads.json`.
- Retained platform evidence: `evidence/model-reference-terminal.json`.
- Cloud numerical JSON and log: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/reference/hf-model-payloads.{json,log}`.
- Native log snapshot: `/data/ycfeng/tmp/exp-1112-125639-935155.log`.

## Limits

This verifies reconstruction and the full-loss kernel on real model logits from short HF inputs, without distributed vocabulary/context shards or mixed-worker payloads. It does not verify Megatron forward, vLLM engine initialization, batch/packing effects, the actual parameter update, five-teacher worker routing, long context, checkpoints, refit, resume, or any of the 200 required updates. All remain pending in phases D/E.
