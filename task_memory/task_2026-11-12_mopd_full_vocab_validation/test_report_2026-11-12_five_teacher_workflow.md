## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record native five-teacher training, checkpoint contents, selected parameter changes, and domain parity limits. |

# Native five-teacher diagnostic

The requested initial student and five distinct teachers completed 5 native optimizer updates. This establishes the short training workflow and nontrivial parameter changes. IF-domain BF16 parity still fails the 0.02 nats/token criterion. Actual checkpoint reload and the formal 200-update run remain outstanding.

## Exact execution

- RJob: `exp-1112-175309-039353`, UID `84a2ec3c-9eac-40b4-b0c9-bc88c1810f70`.
- Personal creator `i-fengyicheng`, H800/codesign, non-preemptible, 7 replicas × 8 GPUs. Source `100.96.128.195:/data/ycfeng:/data/ycfeng`.
- Submit: `PYTHONPATH=/data/ycfeng/steptron /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/nemo-mopd-20261112/submit_five_teacher_diagnostic.py`.
- Exact arguments: `/data/ycfeng/tmp/nemo-mopd-20261112/five-teacher-diagnostic-submit-args.json`.
- Worker uses mounted base Python 3.13.14 through `uv run --no-sync`, Megatron actor environment under `/data/ycfeng/envs/nemo-mopd-actors`, CUDA toolkit 13.0.2, and existing CUDA compatibility libraries. Native policies use TP2/CP4, sequence packing, BF16 outputs, and the 151936-token vocabulary without padded logits.
- Native short overrides: 5 updates, 32 fresh prompts/update, one response/prompt, 128 generated tokens, 8192 total-token limit, native checkpoint every 5 updates. These overrides do not satisfy the formal 256-trajectories/update or 65536-token criteria.
- Durable root: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/five-teacher-diagnostic/exp-1112-175309-039353`.
- Analysis: `UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 uv run --no-sync python /data/ycfeng/tmp/nemo-mopd-20261112/read_five_teacher_scalars.py` and the same invocation of `analyze_five_teacher_diagnostic.py` and `read_five_teacher_parameter_change.py`.

## Observed acceptance checks

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Native/platform completion | Native completion and all replicas succeeded | RJob Succeeded at 11:11:24Z; 7 succeeded replicas and native finished receipt | PASS |
| Optimizer updates | 5 successful, 0 skipped | 5 successful, 0 skipped, trainer version 5 | PASS |
| Teacher activity | All five distinct teachers | `teacher_alias_unique=teacher_model_unique=5` at every update | PASS |
| Positive accepted response coverage | Every accepted update has valid samples/tokens | 32 samples and 4096 response tokens/update, 160 samples and 20480 tokens total | PASS |
| Full-vocabulary objective | Native full reverse-KL, 151936 output entries | Native full-loss route with exact-divisibility model override; loss and full reverse-KL metrics agree | PASS; exact shard-shape GPU observation was covered by the earlier self control |
| Token/mask/version identity | Matching prefix, response masks, versions 0–4 | No mismatches in the 160 retained samples | PASS |
| Loss/gradient health | Finite, no unexplained skipped updates | Loss 0.1115033–0.1711059; pre-clipping gradient norm 1.777639–3.039701 | PASS for observed metrics |
| Parameter changes | Nontrivial changes on teacher batches | 429 of 144896 inspected BF16 values changed, max absolute change 7.629395e-6 | PASS for selected tensors |
| Scorer/vLLM overall mean | ≤0.02 nats/token | 0.01450814, p95 0.08372284, p99 0.12352643, max 0.38544261 | PASS |
| IF-domain parity | ≤0.02 nats/token | 0.02242192 | FAIL |
| Native checkpoint contents | Model/optimizer/scheduler and sampler/data state | Native step_5 weights and train-state, dataloader, replacement/recovery/replay and data-plane files present | PASS for contents; actual reload pending |
| Every model parameter finite | Entire model covered | Four selected tensors are finite | UNVERIFIED for the entire model |

The optimizer reports its norm before applying max-norm clipping. The native `clip_grad_norm` implementation calls `clip_grad_by_total_norm_fp32` with configured clipping 1.0, then returns the original norm. The observed values above 1.0 are consistent with this path; they are not proof that clipping was bypassed. No clipping or gradient scale was changed. LR remained 1e-6, weight decay 0.01, and policy-version lag zero.

Loss varies across fresh prompts and teachers. The short series does not establish convergence. Native full-loss decomposition error reached 1.430511e-6; this scalar diagnostic is separate from the independent arithmetic reference tested in the earlier precision report.

## Parameter comparison

The comparison reads native initial and step_5 checkpoint metadata using the local Python 3.13 runtime, then obtains only exact selected byte ranges through read-only SSH. It does not download full model weights or write to cloud storage. A first remote Python 3.12 metadata read failed with `ModuleNotFoundError: pathlib._local`; the final analysis uses the runtime that created the checkpoint.

| Tensor | Elements | Changed | Max absolute change | Max relative change |
| --- | ---: | ---: | ---: | ---: |
| QKV bias | 57344 | 370 | 7.629395e-6 | 1.802198 |
| QKV layer norm weight | 43008 | 9 | 7.629395e-6 | 0.06321839 |
| MLP layer norm weight | 43008 | 50 | 7.629395e-6 | 0.1224490 |
| Final layer norm weight | 1536 | 0 | 0 | 0 |

Relative changes use the initial values as references. The selected references contain no zeros; other comparisons with a zero reference must record relative error as undefined. These are four selected tensors, not a full-model or optimizer-state comparison. An unchanged final layer norm does not negate the measured changes in the other tensors.

## Evidence and limits

Selected evidence: `evidence/five-teacher-terminal.json`, `five-teacher-training-metrics.json`, `five-teacher-parity.json`, and `five-teacher-parameter-change.json`. Native `training_info.json` records current_step=5, trainer_version=5, consumed_samples=160, total_valid_tokens=20480, sampler_name=in_order, and sampler_dispatch_index=5.

Domain means: math 0.01025350, code 0.01595377, stem 0.01152203, logic 0.01320290, IF 0.02242192. This run uses the ordinary controller, so the dump scorer forward is independently measured; actual training-forward agreement was measured in the earlier observer control. Teacher routing follows the inspected strict source-group mapping; this run's ordinary dump lacks an explicit per-sample selected-teacher field. Do not present the five activity counters alone as an independent per-sample routing audit.

The independent paired-FP32 native save/resume diagnostic `exp-1112-191947-030955` uses the same initial identities, a 5-update horizon in both stages, native deadline `00:00:00:01` for the first stage, and deadline=null for the resumed stage. Its actual recovery and parity are pending.
