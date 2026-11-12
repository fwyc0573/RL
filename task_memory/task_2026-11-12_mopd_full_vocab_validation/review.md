## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Check phase A and acceptance confirmation; refresh after preparation failures. |
| 2026-11-12 | Reconcile completed HF control, native terminal evidence, locked dependency repair and the remaining Triton prerequisite. |

## 1. 规划完成情况

**汇总：** PASS 2｜BLOCKED/FAILED/SKIPPED 0｜TODO 5

检查人：当前主 agent。依据：requirements.md、plan.md、progress.md、提交 c2cfe5f0、输入/CPU/HF 报告和实际 RJob/Replica CRD。此检查衡量路线与状态，不是正确性审查。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A 来源与实现核对 | PASS | 六个模型、五组输入和全词表代码路径已有来源。 |
| Q 核心验收确认 | PASS | continue 已确认200updates及普通修复/运行授权。 |
| B 环境、资产与 recipe | TODO | 资产/recipe/Gym/toolkit/TE完成；锁定Triton及MCore实际imports未完成。 |
| C 数值与指标检查 | TODO | CPU数值/routing和六模型HF对照PASS；native parity待完成。 |
| D 小规模实际验证与修复 | TODO | 原生Megatron/vLLM训练尚未执行。 |
| E 正式训练与恢复 | TODO | 200updates及checkpoint/resume未执行。 |
| F 验收交付 | TODO | 最终训练证据与验收仍缺失。 |

## 2. 偏离检测

**结论：** 已偏离：命中 D2。
**检测窗口：** 上次HF对照提交至完成证据核验｜**当前前沿：** B、C。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | HF脚本直接验证C，最小Ray适配是D必需入口。 |
| D2 细节反复重试 | HIT | Triton传输跨多个恢复窗口未完成B；反复尝试下载路线占用主线时间。 |
| D3 支线问题修补 | CLEAR | 锁定checkpoint依赖和imports是D的实际前提。 |
| D4 可并行任务停滞 | CLEAR | HF控制独立于Triton完成；继续在下载期间准备native入口。 |
| D5 超范围镀金 | CLEAR | 无替代trainer、通用监控框架或accuracy研究。 |
| D6 分析空转 | CLEAR | 已得到HF数值及平台exit0/restart0证据。 |
| D7 路线违背 | CLEAR | 保留五teacher、全词表、H800、200updates和0.02阈值。 |

**文档检查：** S5：progress/review仍描述已完成HF作业为运行中，已修正。S1/S2/S3/S4/S6均未发现明显缺陷。
**归因：** D2执行偏离；S5是恢复时的记录滞后。

## 3. 纠正建议

**分支：** B：先更新实际结果及环境状态，再继续。

**文档更新（先执行）：**

1. progress.md及HF报告：保留数值、container终态及验证边界（可直接执行，已完成）。
2. review.md：反映Triton尚未完成和原生训练尚未执行（可直接执行，已完成）。

**需停止的偏离行为：**

- D2：停止反复变换下载路线；保持唯一checksum-enforced续传，并在等待期间推进必要Ray入口。不得将预分配大小或下载worker成功当作依赖完成。

**接下去的路线：**

1. B：完成锁定wheel/hash、必要actor imports和实际输入校验 → 完成标志：runtime可进入native初始化。
2. 并行C：准备读取native token/logprob/weight-version字段 → 完成标志：同版本一致性可以测量。
3. D → E：实际短训练/修复，再完成200updates及原生resume → 完成标志：满足plan中的数值、update和状态恢复要求。

## 4. 最值得关注的 3 点

1. 六模型HF全词表loss/gradient和reconstruction已PASS；不能外推为Megatron/vLLM parity。
2. Triton是当前环境前提；继续主线入口工作，不改变锁定版本或验收阈值。
3. 当前原生训练updates为0；200updates、长序列、refit和resume均未验证。
