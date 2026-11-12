## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Check completed input/HF/native preparation and preserve prior checks in progress and phase reports. |
| 2026-11-12 | Reconcile TP PASS, native preemption, and the sole queued replacement; carry forward the stop-control correction. |

## 1. 规划完成情况

**汇总：** PASS4｜BLOCKED/FAILED/SKIPPED0｜TODO3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A 来源与实现核对 | PASS | 指定student、五teachers、数据和全词表路线已确定。 |
| Q 核心验收确认 | PASS | continue已确认200updates、运行与修复。 |
| B 环境、资产与recipe | PASS | 输入、tokenizer、锁定runtime imports和配置准备完成。 |
| C 数值与指标准备 | PASS | CPU/HF reference、原生token/mask/version与optimizer成功标志准备完成。 |
| D 原生短跑与修复 | TODO | TP4PASS/0skip已核实；原生nonpreemptible24GPU作业Pending，updates与parity未验证。 |
| E 正式训练与恢复 | TODO | 五teacher短跑、实际长context和200updates/native resume未完成。 |
| F 验收交付 | TODO | TP报告完成；总体运行验收和English archive未完成。 |

## 2. 偏离检测

**结论：** 已偏离：命中D7，已应用纠正；本次续接无新增偏离。
**检测窗口：** 前次16:24HKT检查点及TP完成记录至16:31HKT｜**当前前沿：** D

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | TP只收取既有JUnit和终态；后续只检查D所需logprob/version/resume路径。 |
| D2 细节反复重试 | CLEAR | Hydra完整CPU修正已通过；当前新提交依据明确Preempted Event，不再切换队列。 |
| D3 支线问题修补 | CLEAR | TP证据与原生preemption RCA均服务D。 |
| D4 可并行任务停滞 | CLEAR | TP已完成；原生留队等quota，独立检查其验收数据及resume路径。 |
| D5 超范围镀金 | CLEAR | 没有扩展trainer、registry、schema或准确率目标。 |
| D6 分析空转 | CLEAR | 本窗口完成真实TP4PASS并确认Preempted initiating Event；原生仅待调度。 |
| D7 路线违背 | HIT | 原TP作业的状态guard失败后shell仍继续stop，已在native-startup报告记录；严格shell控制纠正已应用。 |

**文档检查：** S5 review原Running描述与当前TP终态、原生preemption及新Pending作业矛盾；其余S1–S4/S6未发现明显缺陷。
**归因：** D7执行偏离。S5为状态记录滞后；不改变验收或资源授权。

## 3. 纠正建议

**分支：** B：状态文档存在S5，先更新文档再继续。

**文档更新（先执行）：**
1. review.md当前状态与路线：替换旧Running/launcher描述为TP Succeeded与新原生Pending；保留误停记录（可直接执行，已完成）。
2. plan.md状态行、progress.md：记录TP完成及exp-1112-162755-471889唯一nonterminal作业（可直接执行，已完成）。

**需停止的偏离行为：**
- D7：停止在状态guard失败后继续mutation；依赖性shell明确set -euo pipefail，并依据当前实际状态行动。

**接下去的路线：**
1. D：保留launcher76679/monitor52990与exp-1112-162755-471889，检查实际151936词表、loss、grad norm、成功updates和initial/refit parity → 完成标志：5-update原生control与数值阈值满足。
2. 并行：等待原生调度 ‖ 核对同token/same-weight分析与native checkpoint stop/resume路径；不重跑TP。
3. D/E：指定五teacher短跑、同context直接logits/hidden payload和实际长context；通过后使用从开始固定200-step horizon的正式checkpoint/resume运行 → 完成标志：requirements所有执行验收满足。

## 4. 最值得关注的 3 点

1. 原生成功updates仍0；TP kernel PASS不能证明模型训练或训推一致性。
2. exp-1112-162755-471889非抢占式作业排队，quota0、rank2；保持launcher存活，不重复提交。
3. 第二次原生失败已有明确Preempted Event；不修NCCL、Ray lifecycle或重新安装环境。
