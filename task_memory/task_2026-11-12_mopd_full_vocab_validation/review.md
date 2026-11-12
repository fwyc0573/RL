## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Reconcile matched FA4, retain per-sample/tail and compiled/fresh-generation acceptance. |

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A 来源 | PASS | 原student和五teachers身份已核对。 |
| Q 验收 | PASS | 用户continue确认200updates及严格parity。 |
| B 环境 | PASS | native短/长及保存恢复均实际运行。 |
| C 数值准备 | PASS | loss/gradient/forward有可执行比较。 |
| D 短跑与修复 | TODO | 五teacher短检查通过；RoPE与匹配FA4使firstattention exact。长总体MAE0.01905通过，3/5samplemean略超；norm稀疏BF16差异已观察；compiledFAIL，选定eager，fresh/refit待验。 |
| E 正式训练恢复 | TODO | 短native恢复/finite通过；正式200updates未提交。 |
| F 交付 | TODO | 需要D/E验收和已验证代码commit。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** commitd9b967b8至norm/compiled终态与最终runtime预检查｜**当前前沿：** D实际CP1训练与fresh/refit

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 增加实际norm边界只为判定第一处MLP前差异；使用既有capture机制。 |
| D2 细节反复重试 | CLEAR | 匹配FA4检验预先记录的attention假设并使firstattention exact；接下来验证norm/projection，未随机调参。 |
| D3 支线问题修补 | CLEAR | NIXL可选warning未转为支线；CPU/GPUroundoff被记录。 |
| D4 可并行任务停滞 | CLEAR | compiledFA4/norm已完成；24/6/56GPU实际workflow检查可并行，正式run依赖其验收。 |
| D5 超范围镀金 | CLEAR | 使用现有patch/hf_overrides和nativecontroller。 |
| D6 分析空转 | CLEAR | 匹配FA4给出firstattention exact和MLP前稀疏差异，直接决定norm边界观察。 |
| D7 路线违背 | CLEAR | 原模型frequency提交错误已纠正；fullvocab/0.02和200updates不变。 |

**文档检查：** S5旧plan/review仍称匹配FA4待运行，已更新为overall0.01905及3/5samplemean超线；S1–S4/S6无明显缺陷。
**归因：** 无执行偏离；状态文档过期。

## 3. 纠正建议

**分支：** B：状态文档存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan/progress/matchedFA4report记录actualoverallPASS和sample/tail未通过；更新compiled已提交及norm配对下一步（可直接执行，已更新）。

**需停止的偏离行为：无**

**接下去的路线：**
1. D：并行验证最终24GPUselfteacher/scorer/refit、6GPU长训练、56GPU五teacher恢复 → 完成标志：原长场景MAE≤0.02，尾部解释，freshrollout/refit实际通过。
2. D通过后E→F：200successfulupdates、每步256fresh、native恢复、finalrefit与finite检查、报告和scopedcommit。

## 4. 最值得关注的 3 点

1. 匹配FA4使firstattention/projection exact；overall0.01905通过但不能忽略3/5samplemean超线。
2. compiledFA4 MAE0.03366FAIL；实际norm已捕获，选定eagerFA4。
3. 五teacher短恢复与有限性已有证据，正式200updates仍未开始。
