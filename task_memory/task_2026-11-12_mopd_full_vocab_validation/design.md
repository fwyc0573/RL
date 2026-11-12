## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Pin inspected sources and document the proposed runtime, input mapping, and correctness checks. |
| 2026-11-12 | Reconcile verified model/input assets and selected native sampling, vocabulary and failure handling. |

# MOPD validation design

## Evidence and model identity

Target: `/data/ycfeng/sota-engine/RL`, branch `mopd-sys`, inspected commit `d51b2842`.
Source: `/data/ycfeng/steptron/.worktrees/yc-mopd-learning-dynamics-s4-branch`, current inspected commit `6536accb03f0e494f136232d33b49e7f948f42c6`. The November 8 export names the older `4af84b6c` revision; it is not the current source HEAD.

Authoritative source files:

- `playground/users/ycfeng/MOPD/r1distill1p5b_mopd_5teacher_fullyasync_tis_no_kl.py`: `STUDENT`, `TEACHERS`, `TEACHER_REVISIONS`, `MIXED_WEIGHTS`, and `FULL`.
- `task_memory/task_2026-09-26_mopd_learning_dynamics/s0_common/assets_manifest.json` and `data_manifest.json`: pinned downloaded assets and selected data files.
- The same directory's `test_report_2026-09-26_s0-01_stage_and_verify_assets.md`: historical architecture and tokenizer observations.
- `/data/ycfeng/paper-data/qwen-dsv4-distill-1.5b-r32161-full/mopd_5teacher_settings_and_results.md`: readable historical settings and outcomes; corroborated against the source files above.

| Role | Model and revision | Existing cloud path |
| --- | --- | --- |
| Student | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B@ad9f0ae | `/mnt/codesign-exp/ycfeng/opd-assets/oneshot-r1distill1p5b/student` |
| math | hbx/JustRL-DeepSeek-1.5B@0637e40 | `/mnt/codesign-exp/ycfeng/opd-assets/oneshot-r1distill1p5b/teacher` |
| code | agentica-org/DeepCoder-1.5B-Preview@103033da | `/mnt/codesign-exp/ycfeng/steptron/task_2026-09-24_mopd_medium_step3_2teacher/models/DeepCoder-1.5B-Preview@103033da` |
| stem | nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@a8e647c7 | `/mnt/codesign-exp/ycfeng/steptron/task_2026-09-24_mopd_medium_step3_2teacher/models/Nemotron-Research-Reasoning-Qwen-1.5B@a8e647c7` |
| logic | Huawei-RLVE/Deepseek-R1-Distill-Qwen-1.5B-env-256-1600@2e381f50 | `/mnt/codesign-exp/ycfeng/steptron/task_2026-09-24_mopd_medium_step3_2teacher/models/Deepseek-R1-Distill-Qwen-1.5B-env-256-1600@2e381f50` |
| if | Thinking-Space/UltraData-IF-1.5B@be67d8ba | `/mnt/codesign-exp/ycfeng/steptron/task_2026-09-24_mopd_medium_step3_2teacher/models/UltraData-IF-1.5B@be67d8ba` |

The historical asset report observed Qwen2, 28 layers, hidden size 1536, 12 attention heads, 2 KV heads, and model vocabulary size 151936 for the four downloaded teachers and student. Its tokenizer test checked one fixed string, not the entire token-to-ID mapping. DeepCoder's rendered chat template differed. Current cloud assets and complete tokenizer equivalence passed input preparation; all six initial models also passed the HF reconstruction/full-loss/gradient control. The corresponding input and model-reference reports retain exact evidence. Native Megatron/vLLM distributions and training remain unverified.

## Data and sampling

Proposed training inputs are the existing cleaned files at `/mnt/codesign-exp/ycfeng/steptron/task_2026-09-26_mopd_learning_dynamics/data/{group}/train.json`: math has 10,600 prompts; each other group has 11,000. Each group has a separate `probe.json` with 64 prompts. Preserve these evaluation exclusions and the math reserve. Do not turn the historical replay modes into extra runs.

Use fresh student responses and equal group sampling weights, one response/prompt, 256 trajectories/update. Preserve prompt identity, source row, group, and teacher identity through the native `NemoGymDataset` / NeMo-Gym processor. Adapt the existing input files to its JSONL/materialized-task contract without introducing a second trainer or a registry redesign. Use supported taskset or agent attribution and strict teacher matching; an unknown group must not silently select another teacher.

## Runtime proposal

Reuse `examples/run_grpo_single_controller.py` and derive the recipe from `examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack-single-controller-fullvocab.yaml`.

Main path: Single-Controller + TransferQueue + Megatron training and teacher scoring + vLLM generation. Start resource planning from seven 8-H800 nodes: one student trainer, one generation node, and five dedicated teacher nodes. This 56-GPU layout follows the existing allocation mechanism; it is a proposal, not a measured requirement or an upper limit. Select final TP/DP/CP and queue/microbatch sizes from the preflight measurements. Use idle quota for independent checks where it shortens the task, within platform limits.

Use BF16, temperature 1, top-p 1, no top-k truncation, and full-vocabulary reverse KL. Main teacher payload is `hidden_states`; compare it against the existing `logits` path on small identical inputs. The hidden-state route is conditional on full tokenizer and architecture checks passing. Padding added for model parallelism is not an extra semantic token category. Validate output-vocabulary treatment against the actual checkpoint model and reference forward.

Start the optimizer from the historical constant learning rate `1e-6`, Adam beta values `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.01`, and gradient clipping `1.0`. Preserve global token normalization. These are task configuration values, not new code defaults. The existing full loss rejects policy-gradient importance-sampling and ratio options; do not copy the StepTron TIS configuration into this objective.

Use the original 65,536-token ceiling in the main run. Short diagnostics exercise 8K and shorter sequences first; include an actual long-sequence case before declaring the original ceiling covered. Record prompt/response lengths, truncation and valid-loss-token counts rather than hiding overlong examples. The current recipe selects the existing in-order sampler with max_lookahead_versions=0 for same-version initial/refit/resume comparisons; record generation and learner weight versions separately. vLLM uses raw_logprobs. The existing make_vocab_size_divisible_by=64 override for student and teachers makes TP2 vocabulary padding exactly151936; inspect actual GPU dimensions before counting this as verified.

Formal isolated-sample handling uses existing retry/replacement configuration: max_skipped_prompts=128 for data-budget exhaustion, max_consecutive_dropped_prompts=1 for infrastructure-budget exhaustion, on_dropped_prompt=replace, and min_step_batch_fraction=1.0. This preserves 256 fresh trajectories for every accepted update. Formal rollout/router deadlines are7200s and the stall watchdog is9000s; the short diagnostic uses300/300/600s. The263 native failure/timeout/replacement/trace checks passed; actual run accounting must still be reconciled. Native system failures retain their original handling.

## Checks that determine correctness

1. Compare full-vocabulary loss and gradients to a direct FP32 reference on identical small tensors and masks. Test distinct teachers together, sequence packing, vocabulary shards, next-token alignment, and normalization across microbatches. Existing `tests/unit/algorithms/test_opd_full.py` supplies the nearest regression coverage.
2. Run an initial-checkpoint self-distillation control through the same payload transport, separate teacher worker, and LM-head reconstruction. It checks the path used by the requested run, including the Qwen2 checkpoint, rather than only the existing Qwen3 example.
3. Score exactly the same token IDs and response positions with the same frozen student weights in trainer and vLLM. Compare repeated inference/shape controls, trainer logprob forward, and the actual training forward. Repeat after an optimizer update/refit and after resume. Report numerical mismatch separately from stale rollout weights.
4. Compare teacher direct logits with hidden-state/LM-head reconstruction for all five teachers and mixed-teacher batches. Do not use the entropy/cross-entropy identity as proof that the teacher payload is correct.
5. Record successful optimizer updates, loss, exact per-token reverse KL by group, pre-clipping gradient norm and clipping activity, learning rate, finite parameters, valid sample/token counts, response lengths, teacher coverage, rollout failures, weight versions, memory, and step time. For a nontrivial fixed training batch, verify a real parameter update and the expected local loss response; fresh-batch loss need not fall at every step.

The full-loss implementation explicitly replaces its metric set in `nemo_rl/algorithms/loss/loss_functions.py:617`, and `_opd_full_call` does not emit the ordinary policy-gradient mismatch diagnostics. Reuse the existing Single-Controller `DebugWriter` and collected columns for focused checks; add instrumentation only for demonstrably missing evidence. Do not interpret missing metrics as zero mismatch.

## Environment and storage

The local shell has Python 3.10.6 and no `python` command. The current repository declares `>=3.13.14,<3.14` in `pyproject.toml`; old StepTron images are not established NeMo-RL environments. Follow `skills/build-and-dependency/SKILL.md`, inspect the existing image/uv mechanisms, and freeze the actual dependency/submodule revisions before GPU training. No dependency installation or image operation has been performed during planning.

Submission must use local StepMind Python RJobBackend as `i-fengyicheng`, H800/codesign, current-host source mounts, and the required cloud volume. Follow `/data/ycfeng/stepfun-env-handbook/stepmind-python-rjob.md` and the `stepmind-gpu-worker` skill before operations. Keep the launcher alive and preserve platform NCCL settings. Use the mandated `rjob_stop.sh` for job stops.

New durable artifacts belong under `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/`. Use `/data/ycfeng/tmp` for host temporary material and worker-local temporary storage for high-frequency caches. Store only selected small evidence and scripts locally. At 32 or more GPUs, auxiliary validation errors remain non-blocking soft gates: mark artifacts unverified and diagnose while healthy training continues. An unverified measurement still cannot support final acceptance.

No ADR is warranted: this task reuses existing runtime mechanisms and makes reversible experiment choices. Domain terms are recorded in the root `GLOSSARY.md`.
