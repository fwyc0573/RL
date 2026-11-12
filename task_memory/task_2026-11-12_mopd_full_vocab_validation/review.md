## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Check progress after native resume/state/finite audit and collected long-context controls. |

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A 来源 | PASS | 原student和五teachers身份已核对。 |
| Q 验收 | PASS | continue确认200updates和严格parity。 |
| B 环境 | PASS | 多项native短/长工作流已运行。 |
| C 数值准备 | PASS | CPU/TP/真实forward比较可执行。 |
| D 短跑与修复 | TODO | 五teacherFP32逐组短parity通过；长parity仍FAIL，native逐层输出已取得。 |
| E 正式训练恢复 | TODO | 独立短save/resume和finite通过；正式200updates尚未提交。 |
| F 交付 | TODO | 全体验收尚未完成。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** commit1b61f3b9至本次resume审计｜**当前前沿：** D长序列原因定位；E独立短恢复已验证

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 逐层capture用于定位首次差异；有限metadata/chunk读取直接完成finite验收。 |
| D2 细节反复重试 | CLEAR | RoPE/FA4/prefill均有执行前假设和排除结果；停止配置变化，进入逐层对照。 |
| D3 支线问题修补 | CLEAR | serialization阻断必要probe，复用nativeworker既有设置，不修无关nixl警告。 |
| D4 可并行任务停滞 | CLEAR | 两个layerjob同时提交，resume状态分析同时完成。 |
| D5 超范围镀金 | CLEAR | 无core接口/schema扩展。 |
| D6 分析空转 | CLEAR | actualsave/resume/finite均取得验收证据；下一步对照已执行。 |
| D7 路线违背 | CLEAR | 全词表/原teachers/0.02阈值保留，未提前启动正式run。 |

**文档检查：** S5plan/progress/review仍称resumeRunning和FA4未测，与终态冲突，已更新；S1–S4/S6无明显缺陷。
**归因：** 无执行偏离；状态文档过期。

## 3. 纠正建议

**分支：** B：状态文档存在缺陷，先更新再继续。

**文档更新（先执行）：**
1. plan/progress/review同步resume和finite完成、各长对照FAIL及vLLMprobe故障（可直接执行，已更新）。
2. 保存native_resume报告和原始matched2048输出，保留实际training-forward/nonemptyTQ未覆盖限制（可直接执行，已更新）。

**需停止的偏离行为：无**

**接下去的路线：**
1. D：修复已确定的probe环境遗漏并完成vLLM逐层capture，比较同一predictionpositions，定位最先分歧的模块；完成标志是原长场景0.02通过且尾部原因明确。
2. D通过后E→F：200successfulupdates/256fresh/native恢复/finalrefitparity和参数finite，归档提交。

## 4. 最值得关注的 3 点

1. 五teacherFP32短save/resume5successful/0skipped、逐组parity和1777088000model元素finite已验证。
2. FA4及matched2048不能修复长parity，native层输出已取得，避免再重复配置猜测。
3. vLLMprobe失败原因为缺少nativeworker既有callable序列化设置，修复后只重跑受影响probe；正式200updates尚未开始。
