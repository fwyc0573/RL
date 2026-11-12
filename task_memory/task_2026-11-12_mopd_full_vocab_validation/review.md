## Modification History

| Date | Change |
| --- | --- |
| 2026-11-12 | Reconcile completed FP32 long control, record failed CP replay initialization and apply native configuration preparation. |

## 1. 规划完成情况

**汇总：** PASS4｜BLOCKED/FAILED/SKIPPED0｜TODO3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A 来源 | PASS | student、五teacher、数据身份已固定。 |
| Q 验收 | PASS | continue确认200updates与resume/parity。 |
| B 环境 | PASS | 实际native短跑和长序列已成功训练。 |
| C 数值准备 | PASS | CPU/HF/TP与actual-forward观测可执行。 |
| D 短跑与修复 | TODO | 精度修复已验证；FP32长序列MAE0.13106失败；CP replay配置修复后重跑；五teacher Running。 |
| E 正式训练恢复 | TODO | 尚未提交200updates。 |
| F 交付 | TODO | 验收与completion archive未完成。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 精度修复完成至FP32长序列结果和CP初始化修复｜**当前前沿：** D

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | replay直接隔离已失败的长context CP变量，配置修复只补native准备调用。 |
| D2 细节反复重试 | CLEAR | long FP32检验预先记录的head假设；结果维持FAIL后转固定token CP变量。 |
| D3 支线问题修补 | CLEAR | 未修无关Rayfixture/NIXL可选import；瞬时ConfigMap等待已自行解决。 |
| D4 可并行任务停滞 | CLEAR | 原五teacher已Running；CP独立8GPU重跑已提交，当前排队。 |
| D5 超范围镀金 | CLEAR | 未扩大核心接口或添加替代训练实现。 |
| D6 分析空转 | CLEAR | 实际long数值排除仅靠head修复的路线，已推进CP实验。 |
| D7 路线违背 | CLEAR | 保持151936全词表、原五teacher和0.02阈值，正式主跑仍等D验收。 |

**文档检查：** S5 plan/progress仍写FP32 long queued，与Succeeded及FAIL结果不符；本次直接更新。S1–S4/S6未发现明显缺陷。
**归因：** 无执行偏离；S5为状态刷新滞后。

## 3. 纠正建议

**分支：** B：状态文档存在缺陷，先更新再继续。

**文档更新（先执行）：**
1. plan/progress当前D状态：写入FP32 long成功update但parity失败、CP初始化失败及affected rerun（可直接执行，已完成）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. D CP replay：收CP4/CP1固定token误差，完成标志为可复现CP差异或证据排除CP主因，再验证受影响长context路径。 
2. 并行D五teacher：继续原Running job，完成标志为5次成功update、正确routing、非平凡参数改变与native checkpoint。
3. E→F：D通过后200成功updates及native save/resume、initial/refit/resume parity与最终归档。

## 4. 最值得关注的 3 点

1. 短序列actual trainer/vLLM MAE0.00991，teacher投影修复已通过CPU和GPU；长序列仍失败0.13106。
2. FP32长序列worker退出是成功训练后的cleanup；CP失败原因是遗漏train_iters准备，实际native guard修复验证通过。
3. 五teacher已Running；200updates、checkpoint/resume和完整交付尚未完成。
