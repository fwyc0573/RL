## Modification History

| Date | Change |
| --- | --- |
| 2026-11-15 | Record completion-FIFO trajectory selection, explicit staleness, and GitHub push authorization. |
| 2026-11-15 | Add exact comparison against the wyx/full_vocab hidden-state and full-vocabulary storage implementation. |
| 2026-11-14 | Record the batch/FIFO and source-task historical timing follow-up. |
| 2026-11-14 | Add the requested parallel deployment, bottleneck, co-location, and performance-switch investigation. |
| 2026-11-14 | Require only the latest checkpoint to remain, with timely deletion of older checkpoints. |
| 2026-11-13 | Record explicit authorization to clean current-task large, smoke, and unused artifacts. |
| 2026-11-12 | Record the new request, explicit constraints, source identity, and pending acceptance decision. |
| 2026-11-12 | Record user confirmation and authorize execution of the proposed acceptance contract. |

# Requirements

## [Original Request] 2026-11-12

> $grill-with-docs
> 这是一个new task，当前task主要目标是完整测试 mopd，使用/data/ycfeng/steptron/.worktrees/yc-mopd-learning-dynamics-s4-branch 对应task中的student+teachers，并使用全词表。使用h800集群（无gpus限制）。核心是跑通训练workflow，检查训推一致性问题是否存在，如果在请修复；检查关键指标参数是否正常（norm grad，loss等）。对具体的并行设定无要求，你决定。减少提问，只允许提问核心关键问题。

## [Original Request] Invoked skill

The user supplied `grill-with-docs`, which invokes `grilling` and `domain-modeling`.
The loaded grilling skill states: “Do not enact the plan until I confirm we have reached a shared understanding.”
The interview gate was satisfied by the user's `continue` after the concrete acceptance question. The request authorizes in-scope training, diagnosis, ordinary fixes, and verification; do not ask for those permissions again.

## Explicit requirements and decisions

1. This is a new task in the current NeMo-RL workspace, not a continuation of the StepTron learning-dynamics study. The target repository is `/data/ycfeng/sota-engine/RL`; the referenced StepTron worktree supplies model identities and experimental inputs.
2. Use the initial `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B@ad9f0ae` student and the five source-group teachers recorded in that task: math, code, stem, logic, and if. Exact paths and revisions are in [design.md](design.md). Do not substitute the Qwen3 self-distillation example or a historical trained student for the requested production run.
3. The training objective must use the full vocabulary. The original study's `Full / M-prime` data mode and its sampled-token RKL diagnostic do not establish full-vocabulary training.
4. Run on H800 / codesign. The user imposes no task GPU-count ceiling; choose parallelism from actual memory, throughput, and scheduling evidence. Platform quota still applies; the handbook's 128-GPU figure is historical capacity, not currently verified idle capacity.
5. Complete the training workflow; detect and fix train/inference inconsistency; inspect loss, gradient norm, and related training health. A successful process exit is insufficient evidence.
6. Ask only core questions. Resolve routine parallelism, implementation, instrumentation, and environment choices autonomously from evidence.
7. Keep source assets intact and write new artifacts under this task's personal cloud prefix. Existing unrelated changes in the source worktree are outside this task.

## Confirmed completion contract

Recommended completion contract: five-teacher fresh-rollout training for 200 successful optimizer updates, checkpoint save and resume, full-vocabulary loss/gradient reference checks, same-weight train/inference checks before and after weight synchronization, and a documented diagnosis of all training-health anomalies. Use the existing five training pools and original 256 trajectories/update and 65,536-token sequence ceiling, with shorter diagnostic cases before the full run.

This is an engineering validation task. Historical accuracy gains are context, not an acceptance target. The numerical and workflow criteria are specified in [plan.md](plan.md).

## [Original Request] Execution confirmation, 2026-11-12

> continue

This answers the immediately preceding question about proceeding with 200 successful updates, consistency fixes, and save/resume validation. Proceed with B through F, including ordinary in-scope code/configuration changes and approved H800 submissions. Parallelism remains delegated to the agent.

## Delivery

- Reusable NeMo-RL recipe and minimal launch/verification additions using existing mechanisms.
- Any required root-cause fixes, with affected regression checks and signed commits.
- Exact model/data/config/runtime identities, job identities, artifact paths, metric curves, numerical comparisons, and a test report distinguishing PASS, FAIL, and unverified items.
- English `summary.md` at overall completion. The task remains incomplete until execution and verification finish.

## [Original Request] Cleanup authorization, 2026-11-13

> 授权，请你清理当前任务的大产物、smoke产物，无用产物，用codex执行清理

Codex is authorized to delete current-task redundant caches, duplicate archives, and obsolete smoke checkpoint tensors. Preserve source models, training data, installed environments, the FP16 conversion caches needed for subsequent verification, and selected business/runtime/terminal/numerical evidence. Cloud deletion runs through a locally submitted personal Python RJobBackend workload. Historical measurements remain valid recorded observations; removed weights cannot be replayed directly. This cleanup does not cancel the unfinished MOPD validation contract.

## [Original Request] Checkpoint storage constraint, 2026-11-14

> 不允许你额外保存ckpt，因为当前oss的磁盘空间不足，只允许你保留最新的ckpt，及时删除不是最新的ckpt

Only the latest complete native recovery checkpoint may be retained for this task. Delete older current-task checkpoints promptly after the latest checkpoint is confirmed published. Do not create additional checkpoint copies, diagnostic checkpoints, or retained HF exports. Reuse native retention (`keep_top_k=0`, `ft_keep_latest_k=1`) after verifying its behavior. Preserve source assets and required conversion inputs. Save only required native recovery/segment/final boundaries; this overrides the earlier all-checkpoint retention and extra five-step save preparation. Final parity uses the existing native checkpoint loader and in-memory weight synchronizer; it must not write additional checkpoint copies or HF weight exports.

## [Original Request] Parallel performance investigation, 2026-11-14

> 【并行确认】我认为每一步 9–11 分钟明显超过了预期。请你确认：
> - 当前的集群的部署分配是怎么样的？存在多少个分离的集群角色（按照角色分，例如rollout集群、teacher集群），分配的gpus比例，对应的并行度
> - 当前的主要耗时bottleneck是什么？占据了多少耗时？
> - 当前架构是否存在切换为 co-located模式？即时分复用共卡策略。
> - 是否存在明显的遗漏开关导致了耗时增加？允许你调研社区和repo相关的手册和经验

Deliver the actual role placement and parallelism, measured wall-time attribution with timer semantics, current-entrypoint co-location support, and evidence-ranked performance candidates grounded in repository and primary documentation. This investigation runs alongside the existing formal training. It does not replace the 200-successful-update contract or latest-only checkpoint constraint. The current investigation leaves the running configuration unchanged and distinguishes source-backed candidates from measured speedups.

## [Original Request] Batch semantics and historical performance comparison, 2026-11-14

> 继续深入分析和探讨：
> 1. 如何理解 “等待完整 rollout 批次就绪”，每次student 训练要等待多少个rollout批次？student一个update的批次是按照fifo取用的吗？
> 2. yc-mopd-learning-dynamics-s4-branch 下是否留存 当前student+teacher comb的mopd运行记录，是否记录了相关rollout侧耗时？我希望你比较已有的数据记录，找到系统性能耗时差异的关键点。我认为当前架构的单步耗时明显过慢

Extend P with inspected batch/admission/selection semantics and a source-task historical comparison aligned by successful update ordinals. Distinguish the historical fresh-data mode from full-vocabulary loss, exposed consumer waiting from GPU decode time, and observations from causal hypotheses. Preserve E/F and the original dimensions/checkpoint storage decisions.

## [Original Request] Hidden-state mechanism and exact branch comparison, 2026-11-15

> 我们在过去是否确认了 当前架构 使用的全词表处理方案是否和 ds-v4一致？即引入了load hidden state，然后再通过lm head还原？

> 请你比较当前架构的 “teacher hidden states，在 student 训练侧用对应 teacher 的 LM head 重建全词表 logits” 和 /data/ycfeng/steptron/.worktrees/wyx/full_vocab 的全词表压缩方案的异同

Compare the two actual local implementations: captured representation, teacher/head identity, row selection, transport/parallel alignment, projection and distribution memory, backward storage/replay, precision, objective wrappers, and supported configuration. Bind claims to inspected source and distinguish tensor-size arithmetic and primitive tests from paired production GPU throughput. This extends P; it does not substitute a different student/teacher combination, change the formal completion contract, or authorize a large core-interface migration.

## [Original Request] Trainer trajectory FIFO and staleness, 2026-11-15

> 继续主线任务。先修改 trianer集群的获取traj的逻辑，将其与 steptron 的逻辑对齐（fifo取traj，而不是必须等待指定的trajs，只要凑满数量即可，并给定staleness）。对于该需求，优先确认是否存在已有的控制参数来实现，如果没有，请补充该部分的控制参数

> 及时提交和push到 remote github

This is the next E implementation step, before resuming formal training. Inspect existing controls first. Consume complete, training-ready trajectory groups in completion FIFO order across dispatch batches; do not require a particular target_step or a single weight-version cohort to fill the batch. Retain 256 trajectories per optimizer update. Configure staleness explicitly and document whether it bounds admission backlog or actual weight age. Reuse the sampler registry, buffer ownership, and native recovery mechanisms; add only the missing selection control. Commit each verified step and push it to the existing GitHub origin/mopd-sys branch. This explicitly authorizes the branch push.

Initial operational choice: use the existing windowed weight-age policy with max_staleness_versions=5 and add completion_fifo as a selection_order option. Keep max_inflight_prompts=256 and cap unconsumed groups at (5+1)*256=1536. The teacher-enriched publication boundary defines readiness. StepTron's backlog-based staleness does not strictly expire individual stragglers; the NeMo window additionally excludes/evicts groups older than five trainer versions. Preserve the full-vocabulary reverse-KL objective and its existing precision; do not introduce TIS silently.


Native recovery decision: allow only in_order -> windowed as a sampler-name migration, with all other authoritative metadata checks intact. Retain historical group target-step stamps for ownership recovery; windowed selection ignores them. Reject migration when the checkpoint has shortfalls or allowed-short steps at or after the restored trainer version. Actual step40 has no such pending state. Unstamped samplers already require max_skipped_prompts=0 and max_consecutive_dropped_prompts=0; the recipe uses on_dropped_prompt=shrink with a full-batch floor and retains existing row retry behavior. No new importance-sampling setting or loss change is introduced.
