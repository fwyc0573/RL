## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Draft bounded execution milestones and verification criteria for the core acceptance interview. |

# Plan

Status: acceptance confirmed by `continue`; B/C in progress. Execution settings are in [design.md](design.md), and explicit user constraints are in [requirements.md](requirements.md).

## Dependency graph

`A -> Q -> {B, C} -> D -> E -> F`

- A: inspect original model/data sources and current implementation; document the proposal.
- Q: one user answer confirms the engineering completion contract or redirects it.
- B: prepare and verify runtime, assets, native dataset conversion, and recipe.
- C: prepare focused numerical and metric checks by reusing existing tests and debug output.
- D: execute the small real-model baseline and fix demonstrated defects.
- E: complete the five-teacher main run, checkpoint/resume, and post-update consistency checks.
- F: reconcile all evidence, commit verified code steps, and deliver the report and archive.

B and C can progress independently after Q, using batched independent reads and checks. D requires their runtime and observability outputs. Read-only planning does not require GPU allocation. No sub-agents are required. Keep each phase's observed failures and corrections in `progress.md` and the test report.

## Milestones and acceptance

| Phase | Work | Completion evidence |
| --- | --- | --- |
| A | Resolve the actual initial student, five teachers, data pools, full-vocabulary route, and existing validation mechanisms. | Source table and inspected code references; no claims of current GPU or cloud validation. |
| Q | Confirm the proposed 200-update engineering contract. | Actual user answer recorded verbatim in requirements. |
| B | Verify all six model directories, entire token-to-ID mappings and special tokens, checkpoint configs, data split exclusions, compatible Python/uv/backend environment, and native configuration. | Resolved config, versions, model/data identities, five valid teacher routes, and successful CPU import/config checks in the intended runtime. Current cloud checks must replace historical availability assumptions. |
| C | Reuse full-loss, routing and checkpoint tests; prepare trainer/vLLM comparisons and group/gradient metrics that the full loss does not already emit. | Small loss/gradient reference cases and explicit observation fields. No alternate trainer or generic monitoring framework. |
| D | Run self-distillation and all-five-teacher short training; compare hidden states with direct logits; exercise actual long context. | Finite reference comparisons, every teacher exercised, correct masks/packing/normalization, nontrivial parameter updates, measured same-weight logprob errors, and root-cause repair of failures before promoting the affected path. |
| E | Run fresh-rollout MOPD to 200 successful optimizer updates; save native training state and resume it at least once; test refit and final checkpoint inference. | 200 actual successful updates, five-teacher coverage, model/optimizer/scheduler and native sampler/data-plane recovery, continued training after resume, measured post-refit and post-resume parity, finite final parameters, completed artifacts and native terminal status. |
| F | Analyze per-group curves and anomalies; run affected regressions; archive the exact run. | Required test report with actual values and errors, all required checks resolved, signed scoped code commits, and English summary with exact deliverable/cloud paths and remaining limits. |

## Confirmed numerical contract

- FP32 full-loss arithmetic reference: use the existing test precision (`rtol=1e-5`, `atol=1e-6`) where applicable, and the existing gradient assertions for their covered cases. Report absolute errors as well as relative errors; relative error is undefined when the reference is zero.
- Real-model self-distillation: retain the existing suite's broad sanity limits, including median absolute loss/RKL below 0.02 and per-token RKL above -1e-3. These are smoke guards, not a substitute for the direct-logit and same-weight comparisons. Keep the suite's maximum-RKL check as additional evidence.
- Same-weight trainer/vLLM comparison: proposed mean absolute response-token logprob error ceiling is 0.02 nats/token, anchored to the original task's scorer/trainer check. Report signed mean, absolute mean, p50/p95/p99/max, length and group breakdowns. Investigate systematic tails, token shifts, vocabulary/template mismatch, and batch-shape dependence even when the mean passes. Do not loosen the ceiling after seeing failures; quantify the BF16 control error before the main run. Bitwise equality is not assumed across backends.
- Teacher reconstruction: check direct logits and reconstructed logits on identical hidden states and LM-head weights, followed by distribution, full-loss and gradient comparisons against the reference. FP32 arithmetic uses the existing reference assertions above; real-model response-token logprobs must meet the same 0.02-nat/token mean absolute ceiling, with full-distribution and error-tail results reported separately. Measure independent FP32/BF16 controls before the main run; an unexplained full-distribution or gradient discrepancy cannot be counted as PASS merely because sampled-token means pass.
- Training health: no unexplained non-finite loss, gradients, parameters, or skipped updates; positive valid-loss-token coverage for every accepted update; correct teacher routing for every inspected sample; actual parameter changes on nontrivial teacher batches. Explain grad norm relative to clipping and token normalization. Report trends and spikes; do not require every fresh batch or every domain to improve monotonically.
- Save/resume: verify native state contents and progress counters, reload optimizer and scheduler rather than restarting them, complete further successful updates, and compare reloaded model outputs to the saved state. Do not claim bitwise identity of future asynchronous sampling.

## Execution controls

Use 5-update diagnostic runs before the full run, following the existing suite's duration. Start main-run planning from 56 H800 GPUs, 256 fresh trajectories/update, 65,536 total tokens, and one response/prompt; finalize parallelism and queue sizes from measurements. Save recovery state at an early verified boundary and periodically thereafter. Reuse native checkpoint mechanisms; retain source assets and task outputs unless cleanup is explicitly authorized.

Full-vocabulary training is mandatory in the main run. Do not replace it with sampled-token OPD, top-k distillation, self-distillation only, or a separate implementation to obtain a passing result. Do not suppress a demonstrated inconsistency with filtering or an arbitrary scale factor. At 32+ GPUs, auxiliary validators cannot stop healthy training; complete missing validation separately before final acceptance.

Run `check-progress` after each completed phase or repeated failure threshold. Load contribution references before code changes and testing guidance before selecting regression commands. Read `requirements.md` again before final delivery.

## Planned deliverable locations

- Recipe: an entry beside the current full-vocabulary LLM recipes under `examples/configs/recipes/llm/`.
- Reusable verification/launch scripts, if needed: the appropriate existing `tests/` subdirectory; determine exact filenames from the smallest required additions.
- Task evidence: this directory's `progress.md` and `test_report_YYYY-MM-DD_<phase>.md`.
- Durable run artifacts: `/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/`.
- Completion: `summary.md` only after overall acceptance is met.
