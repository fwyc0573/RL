## Modification History

| Date | Change |
| --- | --- |
| 2026-11-16 | 按用户暂停决定，分析最后完整 step176 的训练指标、样本组成和耗时，并保留验证边界。 |

# 暂停后的 MOPD 训练指标分析

训练已按用户要求暂停。最后完整记录是 **step176**，不是上次检查的 step174，也没有计算未完成的 step177。主 writer 和 observer 均由个人账户 `i-fengyicheng` 主动停止，九个 GPU replicas 和一个 observer replica 全部为 Stopped；72 张 H800 已释放。最终验收等待器 PID2019267 在停止 writer 前已终止，没有启动新的 GPU replay。

**观察结论：最新 loss、grad norm 和训练有效性指标没有显示明显失稳；数据组成与输出格式指标存在需要关注的问题。** 最明显的是等比例 admission 经 FIFO 与 age-window 筛选后，logic 的实际训练样本比例降到12.86%，最近20步为11.15%。另外，malformed-thinking 标记率由早期稳态约1.05%升至最近20步7.17%。这两项不能由整体 loss 下降排除。

## 分析范围和数据来源

- 权威恢复链：steps18–176，共159次尝试、158次成功 update，唯一 skip 是已诊断的 step25 FP16 overflow。原 writer 恢复时丢弃的 steps41–46 不纳入该链。
- 当前 FIFO writer：steps41–176，136次连续成功，零新增 skip；34816个不同 sample IDs。权威全部尝试共40704条样本、243350200 response tokens；实际成功 updates 使用40448条样本、241266002 response tokens。
- 三种证据共同确认终点：TensorBoard 全部所需 scalar events、256条 step176 样本摘要、rollout trace 的最后 `step_trained=176`。
- 原 student、五个 teachers、full vocabulary151936、256 trajectories/update、65536-token ceiling 保持；配置为 FP16、FP32 grad reduction、completion FIFO、max_staleness_versions5、inflight256、buffer1536。
- 原生唯一 checkpoint 为 **step170/Adam152**；八个 optimizer shards、每个两个 param groups 的16个计数全部152。31文件、55182356048bytes。只读取5605452bytes metadata/scalars/replay metadata，未读取模型或 Adam tensor payloads。steps171–176的六次成功仅为观察记录，未持久化，暂停时没有额外保存 checkpoint。

## 最新指标与趋势

| 指标 | step176 | 最近20步157–176 | 判读 |
| --- | --- | --- | --- |
| Full-vocabulary RKL / loss | 0.0243255217 | mean0.02506212，range0.02249219–0.02803249 | 有限，整体持续下降；各步 full-RKL 与 loss 完全一致 |
| Pre-clipping grad norm | 0.0819471106 | mean0.09537221，range0.07350215–0.13134272 | 非零且有限；无持续升高或爆炸 |
| Successful optimizer update | 1 | 20/20；FIFO全段136/136 | 没有新增 overflow skip |
| Valid samples / sequences | 256 / 256 | 每步都是256 / 256 | 没有缩小 batch 或样本全部失效 |
| Valid response tokens | 1249356 | mean1398641.7 | 每步正值，与摘要 response-loss token 总数相等 |
| Learning rate / weight decay | 1e-6 / 0.01 | 保持不变 | 与既有 constant scheduler 配置一致 |
| Loss scale | 1 | 全段1 | 既有 FP16 配置，无新增 scaler skip |
| Masked sequences | 0 | 全段0 | 不存在通过屏蔽异常样本使健康指标改善的现象 |
| Natural termination / truncation | 100% / 0% | 全段100% / 0% | 没有记录到达到65536 ceiling后的截断 |
| Teacher models / aliases | 5 / 5 | 每步五组均有样本、每条 prompt→group匹配 | 已覆盖全部教师；这不代表五组占比相等 |

从 step18 到176，loss由0.19038212降到0.02432552，端点下降87.22%。最近20步均值比前20步137–156的0.02980849低15.92%；最近10步均值0.02377696。FIFO最大的相邻步 loss 上升为 step71的17.21%，随后未持续发散。新鲜且组成变化的数据没有逐步单调下降要求。

Grad norm从1.14753270降到0.08194711。该指标是 **clipping前** 的 norm；全链只有steps18–21超过clip1.0，最大1.17616403。FIFO阶段最大0.70465547，最近20步最大0.13134272，未显示 clipping 饱和。小于初始值本身不能证明梯度消失；这里仍有非零 norm和持续成功的 optimizer flags。没有记录 update norm，不能由这些值证明全部最终参数已被独立检查。

Full-RKL最小值为step176的−3.81354e−7，全段最小−4.92493e−7；量级在FP32聚合舍入附近，未达到既有负KL sanity界限−1e−3。step176单token RKL最大10.66125，FIFO全段最大34.62521，最近20步最大22.11275。极值与平均值量纲相同，但极值没有显示持续增加；没有保留 per-token RKL分位数或定位信息，不能对这些尾部token的语义作判断。

## 明显问题：实际训练混合比例改变

输入准备有五类各11000 slots，原始设计使用等组采样概率。新 admission trace中的42496个不同groups仍约20%/组。每条保留样本的prompt_idx与输入交错顺序均独立核对，且每步 trace的训练group IDs与样本摘要完全一致。

| Group | 新 admission 占比 | FIFO实际训练样本占比 | 最近20步样本占比 | 最近20步response token占比 | 最近20步平均response长度 |
| --- | --- | --- | --- | --- | --- |
| math | 19.87% | 21.55% | 22.11% | 26.33% | 6506 |
| code | 20.07% | 18.77% | 19.45% | 28.50% | 8004 |
| stem | 20.04% | 23.32% | 23.65% | 21.21% | 4900 |
| logic | 20.01% | 12.86% | 11.15% | 19.68% | 9642 |
| if | 20.01% | 23.49% | 23.63% | 4.28% | 989 |

**logic被明显低采样。** FIFO全段的12.86%相对20%目标低7.14个百分点，最近20步11.15%低8.85个百分点。实际 admission仍20.01%，因此偏移发生在派发之后的就绪、选择或年龄淘汰阶段，不能归因于初始数据池比例。logic回复最长，FIFO与有限age窗口优先消耗较早就绪的groups、淘汰过老groups，是与现象一致的机制；当前trace没有分组的淘汰原因，尚不能把每条缺失都唯一归因到某个环节。

IF平均response约989tokens，而logic约9642tokens。虽然IF最近20步占23.63%的样本，仅占4.28%的有效token。现有全词表目标按global valid tokens归一化，因此样本占比和token贡献比例都应报告。**按token归一化本身是现有目标的定义；额外的样本混合偏移会进一步改变比较条件。** 总loss下降不能代替每个教师领域的loss下降，也不能证明与原等组实验有相同数据组成。

如后续恢复并继续要求等组样本权重，应先讨论分组选择规则：在每组内保留completion FIFO和staleness约束，通过已有目标组权重决定各组消费数量。该选择可能重新暴露慢组等待，需实测其代价；本次暂停分析没有修改sampler或启动训练。

## Staleness与计算浪费

- FIFO实际age范围0–5，未越界。34816个样本中29719个为age5（85.36%）；最近20步4593/5120为age5（89.71%），其余527个为age4。
- step176为221个age5、35个age4，完全没有age0同权重样本。
- native `evicted_stale_prompt_groups` 全FIFO累计 **6229**，平均45.80/update；最近20步936，平均46.8/update；step176为40。当前没有stale inflight abort、dropped或replaced training groups，最终batch仍完整256。
- 6229/(34816+6229)=15.18%描述的是“已消费加已记录stale淘汰”的group数量比例。它不是GPU算力浪费百分比，也不是全部admission失败率；还有恢复和终点未消费groups，而且trace的`group_discarded`包含正常消费后的清理，不能把42559个discard事件全算作丢弃训练数据。

FIFO解决了consumer等待问题，但当前数据大多使用五个版本以前的生成权重，且有可观的stale淘汰。这符合已配置的age policy；它依然是训练分布和性能成本问题，不能因为age不越界就判为没有影响。配置没有新增TIS或改变full-RKL目标，本报告不推断off-policy bias的数值大小。

## 输出格式标记与零reward

`malformed_thinking_rate`在steps51–70为54/5120=1.05%，前20步137–156为321/5120=6.27%，最近20步为367/5120= **7.17%**，最近10步为205/2560=8.01%。step176为14/256=5.47%，FIFO最高单步30/256=11.72%。这是 **输出格式或parser兼容性需要检查的信号**，而不是loss/grad异常。

代码 `nemo_rl/environments/nemo_gym.py::_detect_invalid_tool_call_and_malformed_thinking` 将最终content中的thinking tags，或reasoning summary中的重复tags标为malformed。它不等价于答题错误率，也不证明训练token错位。现有compact summary未保留每条thinking标记和原文，完整trajectory dump已按授权删除；无法在本轮把367个标记定位到领域或区分实际格式退化与parser行为。没有为了降低这个指标而添加过滤或惩罚。

reward、total_reward、各组reward均为0，verification_skipped为1，zero_variance_fraction为1。这由 `skip_verification=true`、`skip_verification_reward=0.0` 和每prompt只有一个response决定，不能作为模型零正确率或没有学习的证据。训练loss实际为full-vocabulary teacher/student RKL，advantage/teacher_student_logprob_gap是另一个sampled-token统计，不能混作full-RKL。本run没有测量领域准确率或验证集性能。

## 历史日志NaN与当前数值健康

全链282个在该step区间实际出现的scalar tags中，发现210个历史非有限events，全部为35个`*/stddev` tags在steps18–23的NaN。它们属于已经记录并修复的singleton汇总统计问题；底层长度和reward数据有限，原实现对每个singleton求sample std再平均。该修复从恢复step24生效，实际batch的pooled median/std已核验。**steps24–176没有任何已记录scalar NaN/Inf，所有loss与成功update的grad均有限。** 不隐藏或用0替代历史NaN；详细tag/step列表保留在scalar-statistics JSON。

唯一跳过update是step25的已诊断FP16 overflow，恢复后step41–176没有新增skip。最终完整模型/Adam tensor的独立finite scan本轮未执行，不能从scalar有限推断全部tensor已通过扫描。

## 耗时：目前瓶颈在student训练

最近20步总耗时mean **70.167s**（step176为65.587s），其中student policy_training mean53.168s，占75.77%；exposed_generation mean0.581s，占0.83%。当前已不是先前每步9–11分钟的等待状态。

其他最近20步timer均值：policy/reference logprobs5.259s、advantage_calculation7.106s、logprob prep1.221s、weight sync0.613s。train_data_dump6.576s主要嵌套在advantage stage；不要与advantage耗时重复相加。checkpointing只在steps160/170出现，发生保存时均值11.616s；摊到这20步为1.162s/update。代码的checkpointing timer位于total_step_time外层计时块内部，已经包含在总耗时中，不应重复相加。保留dump是既有完整指标检查需求，本次没有改动日志开关。

Teacher logprob、inference、lock_wait均是自上一次metrics drain以来**并发请求的累加计时**。最近20步mean teacher_logprob557.622s、lock_wait485.834s、inference49.388s；不能把557.622s视为单步顺序耗时或与70.167s相加。较高lock wait显示teacher串行队列存在，但由于异步overlap，consumer只暴露0.581s等待。该现象与采样组成和age-window应一起评估。

与先前相同72卡的steps32–38相比，已记录的稳态steps51–130从650.300s降至70.424s（9.234×）。FIFO、staleness、graph设置和response长度同时变化，这不是单一开关的独立因果测量。

## 训推一致性判读

全链age0同权重观测6144samples/46227111tokens，token加权scorer/vLLM MAE0.00123752925，低于既有0.02标准。当前FIFO中的age0只在恢复初始出现256samples/856388tokens，pooledMAE0.00118336；最近steps没有age0 coverage。

step176的scorer/vLLM MAE0.00532999、p95绝对差0.02336836、p99为0.04941740、max0.92218590，都来自age4–5。这些值包含model weight drift，不能用它们证明或否定same-weight backend parity。当前批次的较低MAE不能代替最终训练态forward的检查。实际 `model.train()` / scorer / vLLM 最后三方GPU replay没有执行，本轮因用户暂停而不再作为完成前置条件，仍明确记为unverified。

## 命令、环境和产物

本次分析没有修改训练实现或配置，也没有新GPU作业或checkpoint。个人停止操作：

```bash
bash /data/ycfeng/steptron/playground/users/ycfeng/tools/rjob_stop.sh \
 --creator i-fengyicheng \
 --reason 'User requested pause and analysis at the last completed training step on 2026-11-16' \
 exp-1115-193456-282702
bash /data/ycfeng/steptron/playground/users/ycfeng/tools/rjob_stop.sh \
 --creator i-fengyicheng \
 --reason 'User-requested pause; observer collected all 136 completed FIFO steps through step176 without failures' \
 exp-1115-193548-376103
```

平台terminal查询复用：`cleanup_personal_env.sh current_job_state_20261115.py exp-1115-193456-282702 exp-1115-193548-376103`；脚本均位于 `/data/ycfeng/tmp/nemo-mopd-20261112/`，从本机个人Python client读取，没有使用本机平台CLI。停止前通过/proc验证PID2019267命令后发送SIGTERM，停止后再次确认三个launcher PIDs均不存在。

运行环境：local platform Python3.10.6/brainpp2.8.6；指标和绘图UV0.13.7/Python3.13.14/Torch2.13.0+cu130/Matplotlib3.11.2。相关代码规范和记录使用contributing、nemo-rl-docs、nemo-rl-session-memory；GPU控制使用stepmind-gpu-worker；路由自检使用check-progress。没有依赖安装或Docker操作。

```bash
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 \
PYTHONPATH=/data/ycfeng/sota-engine/RL \
UV_CACHE_DIR=/data/ycfeng/tmp/nemo-mopd-20261112/uv-cache \
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python \
 /data/ycfeng/tmp/nemo-mopd-20261112/read_fifo_observer_metrics_20261116.py
# The refreshed plot command is retained in test_report_2026-11-14_training_health_curves.md.
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 timeout 60s \
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python \
 /data/ycfeng/tmp/nemo-mopd-20261112/verify_paused_health_curves_20261116.py
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 timeout 60s \
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python \
 /data/ycfeng/tmp/nemo-mopd-20261112/analyze_paused_metrics_20261116.py
UV_PROJECT_ENVIRONMENT=/data/ycfeng/envs/nemo-mopd-py313 timeout 60s \
/data/ycfeng/tmp/nemo-mopd-20261112/uv-cli/bin/uv run --offline --no-sync python \
 /data/ycfeng/tmp/nemo-mopd-20261112/inspect_latest_formal_native_state_20261114.py
```

绘图使用30GiB address-space上限和60秒timeout；既有systemd MemoryMax scope需要互动授权，未重复尝试。完整数据文件由准备阶段验证，本次只cheap-read dataset前五个task_id并核对所有保留样本的交错index/group，不重扫模型或分配GPU进行准备。

| 验证 | 条件和实际结果 | 状态 |
| --- | --- | --- |
| Pause / resource release | 精确个人writer和observer全部Stopped，waiter消失、无新replay | PASS |
| Last complete step | TensorBoard所需tags、sample summary和trace三方为176 | PASS |
| Authoritative chain | 159rows/158successes；skip25；discarded旧链不重复计算 | PASS |
| Sample coverage | 40704distinct IDs；每步256；prompt/mask/group/age匹配，无摘要issues | PASS |
| Current numerical health | steps24–176的所有已记录scalar有限，最新loss/grad正常 | PASS inspected scalars |
| Equal teacher sampling after selection | admission约20%；训练logic12.86%，recent11.15% | ISSUE observed |
| Output format metric | recentmalformed7.17%，高于早期1.05%；原文原因无法确认 | ISSUE / RCA UNVERIFIED |
| Historical statistics | stddev NaN18–23保留；从24起native pooled修复生效 | EXPLAINED / FIX VERIFIED previously |
| Staleness | age0–5符合配置，但85.36%FIFO samples为age5且6229evictions | PASS bounds / COST observed |
| Save state | sole170/Adam152；16param-group counters匹配；无extra save | PASS metadata only |
| Same-weight scorer/vLLM | 既有age0 pooledMAE0.00123753≤0.02；最近无same-weight覆盖 | PASS historical coverage / latest UNVERIFIED |
| Final tensors / actual train-forward parity / domain accuracy | 本轮未执行/未记录 | UNVERIFIED |
| 200updates / finalGPU replay | 用户明确取消本轮前置要求 | SKIPPED, not PASS |

分析脚本首轮严格检查发现历史stddev NaN；重新核对到既有修复后，保留原值并分类而非屏蔽。Trace核对最初假设所有训练样本都有本次admission event，但256个来自已恢复的512-group ledger；随后用restore证据核对身份。Checkpoint统计最初把shard数等同param-group数；检查原生内容确认8shards×2groups后全部152。上述是CPU分析假设纠正，没有训练失败或改动训练行为；原始错误和修正记录在progress.md。

主要产物：`evidence/formal-training-health.png`、对应CSV、`formal-training-health-verification-20261116.json`、`paused-training-metric-analysis-20261116.json`、`paused-training-scalar-statistics-20261116.json`、`user-pause-terminal-20261116.json`。JSON包含精确输入文件路径/SHA256、每tag统计、各窗口和各领域的实际数值。Observer的136个compact summaries保留；云端原TensorBoard event文件13087791bytes，原始trajectory dumps已清空，未复制大权重。

Checkpoint云路径：`/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-five-teacher-fp16-resume/exp-1113-174329-659007/checkpoints/step_170`。

当前run日志云路径：`/mnt/codesign-exp/ycfeng/nemo-rl/task_2026-11-12_mopd_full_vocab_validation/final-five-teacher-fp16-resume/exp-1115-193456-282702/logs/exp_001/`。
