## Modification History

| Date | Change |
| --- | --- |
| 2026-11-14 | Record actual resumed overflow attempts and preserve the eight-unit TQ restore layout for expanded generation. |
| 2026-11-14 | Apply the user's latest-only checkpoint constraint before native resume and final parity. |
| 2026-11-14 | Replace stale queue state with actual zero-update failure and exact rank3 preemption evidence. |
| 2026-11-13 | Verify corrected32GPU capacity and reservation order; advance replacement while preserving formal acceptance. |
| 2026-11-13 | Reject partial-node GPU proposals after platform documentation and native CPU checks; preserve the exact queued 56-H800 formal run. |
| 2026-11-13 | Replace stale queue state with second formal CUDA backward OOM and recovered zero-update evidence. |
| 2026-11-13 | Record formal first-batch CPU OOM and resource correction. |
| 2026-11-13 | Accept completed five-teacher PP4 GPU save/resume, finite checkpoint, and full diagnostic parity. |
| 2026-11-13 | Recover latest PP4 checkpoint failure and reproduce uninitialized native Adam state. |
| 2026-11-13 | Record PP4 PASS, five-teacher overflow FAIL, and authorized artifact cleanup. |
| 2026-11-13 | Recover failed gradient probes, CPU correction, actual queue state, and formal FP16 preflight. |
| 2026-11-13 | Reconcile complete FP16 three-way parity and remaining training/gradient frontier. |
| 2026-11-13 | Recover actual decode controls, padding RCA, and ordinary native replay frontier. |
| 2026-11-12 | Draft bounded execution milestones and verification criteria for the core acceptance interview. |
| 2026-11-12 | Reconcile FP32 RoPE terminal results and retain the remaining long-context failure. |
| 2026-11-12 | Recover TQ serialization failure and native empty-batch diagnostic. |
| 2026-11-12 | Record rejected no-prefix-cache and first-layer GEMM controls; advance actual decode comparison. |

# Plan

Status: A/Q/B/C/D and authorized cleanup are complete; E/F remain outstanding. D's five successful diagnostic updates do not count toward the formal 200.

Formal job `exp-1113-174329-659007` (UID `da3fd441-bc51-4d5b-aa27-ad216d7fe445`) ran on 4 × 8 H800, 32 CPU / 768 GB per node, then Failed at **2026-11-13T13:38:07Z (21:38:07 HKT)**. All four runtime receipts verify 8 × H800, Python3.13.14, Torch2.13.0+cu130 and the expected company image/current local NFS. Personal API at2026-11-13T17:42:14Z confirms two Failed and two Stopped replicas. Training launcher82605 exited1; reader53011 and monitor5931 exited0. That failed request is no longer active. Native recovery job `exp-1114-021713-015045` (UID `4637305a-eb69-469e-96a7-9e2002abe37e`) is Running on4×8H800 after verified latest-only cleanup and CPU regression checks. Its four runtime receipts and41input checks per worker pass. The first restored attempt at controller step11 used loss scale128, returned finite loss0.2018780112 and successflag0. Actual restoration is observed. The latest confirmed restored attempt is controller step16: loss0.1900885552, scale4, successflag0,256samples and2,145,439valid response tokens. Formal successful updates remain0/200. At2026-11-13T19:29:45Z, exact namespace inspection confirms only step10 remains,24,884,408,375bytes. The first successful formal update remains pending.

Recovered business evidence covers **11 optimizer attempts, zero successful updates**. Last loss0.1954945773 is finite, with256 samples and2,242,562 valid response tokens. Grad norm and scaler values were not recorded by TensorBoard for these unsuccessful steps. Five-group scorer/vLLM mean error across all11steps is0.00128222–0.00135436nats/token, below0.02, with all summaries reporting zero issues. This is pre-update scorer/generation evidence; actual training forward and final post-update parity remain outstanding.

The exact rank3 historical Events query on2026-11-14 binds replica`exp-1113-174329-659007-eccbc87e`, UID`160eb189-67d9-461e-93ce-b808243a4b8f`, to node`gpu-h800-0320.host.platform.shaipower.com`. At13:37:14Z it records both `CleanPolicyTriggered` (`GPU浪费2小时可抢占`, action marked as preemptible) and `Preempted by a pod on node gpu-h800-0320.host.platform.shaipower.com`. This precedes missing Ray heartbeats/node death13:37:30Z and later controller/launcher cleanup. Preemption is established; the requesting pod and initiating human actor remain UNKNOWN. CUDA allocator warnings before the final finite step do not establish the termination cause. Exact-node Events/incident/DeepScan queries completed; all-operation audits timed out and are not empty results. Existing Sherlock session rejected authentication; a deeper authenticated read is no longer necessary to establish the returned preemption event, though attribution limits remain recorded.

Native checkpoint boundaries step1 and step10 and the early save/resume phases exist. Resume preparation must verify native scaler/optimizer/scheduler/data-plane state before restoring step10; unsaved attempt11 is rolled back and remains recorded as an attempted, unsuccessful update. FullCPUreadiness2034tensors/10,662,528,000finite elements and55,000training/320monitoring rows remains reusable. Do not repeat full input scans, completedD diagnostics, the completed cache/smoke cleanup, or rejected precision/backend probes. Continue native latest-only checkpoint pruning.

Checkpoint storage: retain only the latest complete native recovery checkpoint. Verify `keep_top_k=0` and `ft_keep_latest_k=1`; delete older current-task checkpoints before resume and after each publication. Do not retain extra checkpoint or HF export copies. The user's2026-11-14 storage instruction overrides prior unrestricted retention and extra five-step save preparation.

Latest-only retention, checkpoint state inspection, native loss-scale metrics, and resume preparation are verified. Recovery jobexp-1114-021713-015045 is running. Native step10 restoration and the first overflow attempt are observed. Next: verify the first successful update/loss/grad/teacher routes → complete200successfulupdates (successflags andAdamstep) → finalfinite/save-state/refit/training-scorer-vLLMparity. Any resource change must address observed workload placement/utilization rather than disable platform policies or increase Ray heartbeat timeouts. Auxiliary validators remain nonblocking soft gates. Next-segment preparation uses7×8H800 with8student/32generation/10teacherGPUs and explicitly preserves8TQstorageunits; CPUpreflight/dry-run and the native count contract pass. Do not submit until the current job reaches its native save/terminal boundary and the latest checkpoint is ready.

## Dependency graph

`A -> Q -> {B, C} -> D -> E -> F`

- A: inspect original model/data sources and current implementation; document the proposal.
- Q: one user answer confirms the engineering completion contract or redirects it.
- B: prepare and verify runtime, assets, native dataset conversion, and recipe.
- C: prepare focused numerical and metric checks by reusing existing tests and debug output.
- D: execute the small real-model baseline and fix demonstrated defects.
- E: complete the five-teacher main run, checkpoint/resume, and post-update consistency checks. The independent short native save/resume diagnostic can run during D; the formal 200-update run waits for D acceptance.
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


## Evidence-based parallelism and remaining numerical diagnosis

TP2/CP1 backward originally requested18.55GiB gradient and OOMed. TP4 reduced that allocation and completed the original65536update, but sameweight scorer/vLLM MAE0.06853FAIL and gradnorm70.85 prohibit its formal promotion. Native TP2/PP2/CP1 subsequently completed the same actual65536scenario with finite loss-8.18e-9/grad0.00337. TP2/PP1 and PP2 replay outputs are identical and nativefreshscorer reproduces both; all have MAE0.05936FAIL. CP2 was attempted and no attentionbackend was available.

The no-prefix-cache TP2/PP2/CP1 control exp-1112-231515-172060 completed 1 update/0 skipped with loss-1.10e-8 and grad0.00338, but MAE0.06052 FAIL. Historical D controls included actual eager decode versus identical newly generated tokens prefill in completed exp-1112-234539-001675 (235tokens,MAE0.02390397FAIL, first layer attention); native same-newtokens observer exp-1112-235130-765032 Failed on EOS padding; ordinary replay exp-1113-000248-302886 subsequently Succeeded; exp-1112-233623-543941 failed before forward at its ordinary-pickle boundary, launcher45185/monitor50481 have exited. The invariant configuration is rejected on its measured prefill result; its training job exp-1112-225944-064229 failed during initial NCCL broadcast and supplies no numerical training verdict. Do not retry that rejected path solely to obtain another verdict.

Matched prefill captures show exact first-layer QKV/RoPE/attention on all 32 response prediction positions of eight samples. Sparse BF16 RMSNorm differences first appear in post-attention normalization or layer-1 input normalization. Captures do not cover every prompt position and do not establish a full-model causal repair. The completed one-H800 GEMM shape probe exp-1112-232421-248453 has 16 full records/256 exact comparisons and rejects first-layer QKV shape/reduced-precision reduction as the cause. Do not alter training precision or expand that rejected probe; actual decode capture targets the remaining normalization/projection/attention boundaries.

Formal FP16 preparation remains CPU evidence only. Submit after D acceptance; retain the original five teachers, full 151936 vocabulary, 256 fresh trajectories, 65536 ceiling, 0.02 limit and ordinary resume entrypoint. The paired teacher control explains TP4 gradient 70.84613; native mixed-precision GEMM explains the raw FP32 reconstruction mismatch. Actual masked-objective gradient comparisons passed. PP4 subsequently passed the actual65536 update and scorer/vLLM mean-error criterion. The native five-teacher scaled-overflow and save/resume frontier passed on exp-1113-104851-054395; E now owns the formal dimensions and final-checkpoint checks.

## Authorized cleanup, 2026-11-13

1. Verify duplicate archives against extracted bytes, then delete redundant local files and rebuildable temporary UV cache.
2. Inventory obsolete cloud conversion caches and three prior BF16 smoke checkpoints; verify their exact RJobs are terminal. Preserve small checkpoint metadata and historical measurement evidence before deleting tensor shards through a local 0-GPU codesign RJob.
3. Verify deleted paths are absent and retained data, source assets, active caches and selected evidence remain; record exact paths and logical bytes.
4. Resume D at the native FP16 overflow repair, then five-teacher save/resume and E.

Acceptance: all manifest-approved obsolete tensor/cache paths absent, required retained artifacts accessible, exact deletion receipts and local space changes recorded. Cloud logical bytes removed are distinct from backend physical storage reclaimed.

Cleanup status: completed and verified by test_report_2026-11-13_artifact_cleanup.md. D is also complete. Proceed at E; no repeat cloud cleanup or rejected BF16/TP4 diagnostic is required.

Completed D diagnostic:56H800,TP2/PP4/CP1,16attempts,5successful updates after native scaler backoff;32 trajectories/update and8192 ceiling remain diagnostic only. Native retained cache resharding and early save/resume succeeded. Formal target and dimensions remain unchanged.

Formal preparation now has a reusable TP2/PP4/CP1 FP16 recipe and successful native CPU preflight/dry-run, retaining200/256/65536. The native optimizer success flag, not the controller or scheduler tick count, settles update acceptance. The configured scheduler is constant (LR=min_LR=1e-6, WD0.01, warmup0); explicit lr_decay_iters200 preserves its settings across measured-deficit native resume segments. This does not promote the formal run before D passes or change the200-successful-update contract.


Final checkpoint verification preparation: replay_checkpoint_logprobs.py now uses2nodes/16H800 for allfivefresh sourceprompt responses, nativecheckpoint refit, scorerlogprobs and an independent native model.train() diagnostic on those exacttokens before its singleoptimizerattempt. The observer reuses prepared native LOGPROB results; existingOPD_FULL gradientreference behavior remains. Diagnostic attempts are excluded from formal200success accounting. CPU callback/tokenoffset/mask/lossinputs/mountedregistration checks pass; finalGPUvalidation awaits the finalformalcheckpoint. Do not extend the preparedharness while the existing formaljob is queued.
