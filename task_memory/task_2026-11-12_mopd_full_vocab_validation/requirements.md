## Modification History

| Date | Change |
| --- | --- |
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
