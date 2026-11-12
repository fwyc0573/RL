## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Reconcile completed core probes, diagnose submitted frequency error, and retain strict long-context acceptance. |

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A 来源 | PASS | 原student和五teachers身份已核对。 |
| Q 验收 | PASS | 用户continue确认200updates及严格parity。 |
| B 环境 | PASS | native短/长及保存恢复均实际运行。 |
| C 数值准备 | PASS | loss/gradient/forward有可执行比较。 |
| D 短跑与修复 | TODO | 五teacher短检查通过；长cache干预修复首次Q/K，endLP仍FAIL。core原frequency修复已验证，FP32head的长LP仍FAIL。 |
| E 正式训练恢复 | TODO | 短native恢复/finite通过；正式200updates未提交。 |
| F 交付 | TODO | 需要D/E验收和已验证代码commit。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** commit7a052f0b至原frequency GPU验收、重复实现清理及匹配backend准备｜**当前前沿：** D长序列一致性

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | sequence-parallel分析修正仅为复用已有GPUcapture。 |
| D2 细节反复重试 | CLEAR | 已排除SwiGLU；core首次放大经实际cache定位为提交frequency错误，只重跑受影响两项。 |
| D3 支线问题修补 | CLEAR | NIXL可选warning未转为支线；CPU/GPUroundoff被记录。 |
| D4 可并行任务停滞 | CLEAR | 修正eager/compiled完成；下一native/vLLMbackend配对并行，正式run依赖D通过。 |
| D5 超范围镀金 | CLEAR | 使用现有patch/hf_overrides和nativecontroller。 |
| D6 分析空转 | CLEAR | 最新capture给出frequency实测证据，随后恢复原参数提交。 |
| D7 路线违背 | CLEAR | 原模型frequency提交错误已纠正；fullvocab/0.02和200updates不变。 |

**文档检查：** S5plan仍称修正frequency待验证，已更新为实际0.02658/0.03851FAIL和匹配FA4下一步；S1–S4/S6无明显缺陷。
**归因：** 无执行偏离；状态文档过期。

## 3. 纠正建议

**分支：** B：状态文档存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan/progress/RoPEreport记录corefirstQ/KPASS及全LPFAIL；原模型frequency正确，匹配FA4为下一步（可直接执行，已更新）。

**需停止的偏离行为：无**

**接下去的路线：**
1. D：并行执行native/vLLM匹配FA4，比较firstattention与完整LP → 完成标志：原长场景MAE≤0.02，尾部解释，freshrollout/refit实际通过。
2. D通过后E→F：200successfulupdates、每步256fresh、native恢复、finalrefit与finite检查、报告和scopedcommit。

## 4. 最值得关注的 3 点

1. 原模型rope_theta10000，前次提交1000000；actualcache已证实这次错误，不扩大为modelconversion问题。
2. 两个修正作业完成；生产cache已验证firstQ/K，nativeFA3未具备独立包，FA4配对准备中。
3. 五teacher短恢复与有限性已有证据，正式200updates仍未开始。
