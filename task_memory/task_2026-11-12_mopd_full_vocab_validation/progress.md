## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Record planning evidence, environment observations, and the pending core acceptance answer. |

# Progress

## Current status

| Phase | Status | Evidence |
| --- | --- | --- |
| A | completed | requirements.md, design.md, plan.md; source/model/data and current full-loss implementation inspected. |
| Q | in-progress | Recommended engineering acceptance contract prepared; no user answer yet. |
| B | pending | No runtime preparation or current cloud asset verification yet. |
| C | pending | Existing tests and DebugWriter inspected; no code written or tests executed. |
| D | pending | No GPU diagnostic job submitted. |
| E | pending | No main training or resume performed. |
| F | pending | Final acceptance and completion archive outstanding. |

## 2026-11-12 — source and implementation investigation

- Read target `AGENTS.md`, MOPD algorithm documentation, the full-vocabulary recipe and parents, and the source task's model/data records before choosing the implementation route.
- Invoked `grill-with-docs` by reading its local SKILL.md and loading `grilling` and `domain-modeling`; no callable Skill tool is exposed in this session. Loaded glossary format, repository documentation conventions, build guidance, StepMind worker guidance for the planned run, and `check-progress` for the phase checkpoint.
- Target Git working tree was clean on `mopd-sys`, HEAD `d51b2842`. The source worktree's current HEAD is `6536accb03f0e494f136232d33b49e7f948f42c6`; its unrelated untracked files were not edited.
- Source recipe confirms initial R1-Distill-Qwen-1.5B student, five teacher paths/revisions, equal source weights, and a historical 200-update full-data mode. The manifest confirms 54,600 training prompts and 320 separate probe prompts across five groups.
- The historical asset check tested matching token IDs on one string. That evidence is insufficient to prove whole-vocabulary equivalence. DeepCoder's chat template differs; B owns current full tokenizer/config verification and common tokenized-input checks.
- Inspected `OnPolicyDistillationFullConfig`, `_opd_full_call`, native payload reconstruction tests, Single-Controller DebugWriter, NemoGymDataset, and the checkpoint functional test. Full-vocabulary loss replaces sampled-token optimization and the ordinary PG mismatch metric set. C owns missing observability; no bug has yet been established by execution.
- The existing full-vocabulary shell suite contains cleanup commands. It was read only; future verification must reuse its checks without executing unauthorized deletions.
- Host date was `2026-11-12T02:12:07+08:00`; `python3 --version` returned Python 3.10.6. Bare `python` returned `command not found`. `pyproject.toml` requires Python `>=3.13.14,<3.14`, so B must select a compatible NeMo-RL runtime rather than reuse the historical StepTron environment.
- A bounded public documentation search returned no usable result; all decisions here rest on inspected local primary artifacts. No external API behavior is claimed verified by that search.

## Changes and their purposes

- `requirements.md`: preserve the raw request, explicit authorization, and one pending acceptance question.
- `design.md`: make exact inputs, existing mechanisms, numerical hazards and runtime prerequisites reviewable.
- `plan.md`: define finite milestones and acceptance, including the reference checks needed to distinguish a real fix from hidden error.
- Root `GLOSSARY.md`: distinguish full-vocabulary distillation, sampled-token feedback, train/generation mismatch and policy staleness without implementation details. No ADR is warranted for the reversible run choices.
- Verification method for this phase: static source and document consistency checks only. GPU training, current cloud availability and numerical acceptance remain pending.

## Pending handoff

Phase-A validation is recorded in `test_report_2026-11-12_plan_sources.md`: source group counts, 54,600 training/320 probe rows, document links and source premises passed. `review.md` records the required check-progress self-check: PASS 1 / TODO 6, no detected drift. Applied its route: finish the local handoff, ask only Q, then advance B/C after the answer; do not expand the investigation or claim runtime validation.

1. Obtain the single core answer about the proposed 200-update engineering completion contract, then record it verbatim.
2. Advance B and C: verify runtime/current assets and prepare the minimal recipe plus focused numerical/metric checks.
3. Execute D, then E; diagnose actual failures, make scoped fixes and rerun affected checks.
4. Re-read requirements, reconcile every acceptance item, and write the final report and English archive. Do not mark the overall task complete at the planning checkpoint.
