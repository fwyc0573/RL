## Modification History

| Date | Change |
| --- | --- |
| 2026-11-15 | Compare NeMo-RL and wyx/full_vocab source, verify small CPU loss/gradient and saved tensors, and quantify storage scaling. |

# NeMo-RL 与 wyx/full_vocab 全词表方案比较

比较已完成。两者都保存 teacher 输出投影的输入，再用对应 teacher head 恢复完整词表；压缩依据相同。关键差异是 `wyx/full_vocab` 将 response-row 选择、student/teacher projection、分布计算和 backward replay 一起分块，当前 NeMo-RL 只在 KL kernel 内分块，仍保留整段 vocabulary logits。这是实际源码差异，不能把相同的 `chunk_size=128` 解释为相同的显存限制。

## 对象与证据范围

- NeMo-RL：`/data/ycfeng/sota-engine/RL`，branch `mopd-sys`，检查前 HEAD `a2fca0c26366e47dcabf5bff2f2d697d6ebeb94b`。当前运行参数指最后正式 writer 的 resolved config：`evidence/formal-native-resume-6-generation48-resolved-config-20261114.json`；已准备而未提交的 graph candidate 不算当前实测部署。
- StepTron：`/data/ycfeng/steptron/.worktrees/wyx/full_vocab`，HEAD `a94dc8c98df4cb5e765f02ee8e42e697f5bf7741`。保留的 Step3p6 实验为 `playground/users/getianyi/mopd/step3p6/step3p6_mopd_full_vocab_rkl_r3_it100_ctx128k_c128.py`。该 worktree 的既有 `task_memory/task_2026-11-14_full_vocab_analysis/` 被用作定位线索；核心结论重新核对源码。
- 本轮验证是源码、维度算术和一个小型 CPU primitive 对照，没有启动 GPU 作业、读取 checkpoint tensor payload 或写入新 checkpoint。没有测两条架构的同配置 H800 峰值显存和吞吐。
- StepTron 历史 1.5B fresh-data `Full / M-prime` 耗时报告不等同于此 `wyx/full_vocab` 实现的全词表性能实测。

## 共同机制

对于指定 teacher k，缓存其输出 linear 的输入 h_k，用冻结的 W_k 重建 z_k = h_k W_k^T，随后在全部词表项上计算 KL(p_student || p_teacher)。它不需要先生成完整 logits 再执行 SVD，不使用训练出的压缩器，也没有在检查的 hidden-cache 路径中发现 top-k 概率截断、tail bucket 或专用 hidden quantization codec。

两条路径缓存的都不是缺少最后 norm 的原始 decoder 输出：

- NeMo-RL：[opd_full_capture.py](/data/ycfeng/sota-engine/RL/nemo_rl/models/megatron/opd_full_capture.py:21) 在 `output_layer` 前挂 hook，捕获已完成 `final_layernorm` 的输入。
- StepTron：[teacher_supervision.py](/data/ycfeng/steptron/.worktrees/wyx/full_vocab/steptron/core/mopd/teacher_supervision.py:251) 调用 [OutputEmbedding.prepare_projection_input](/data/ycfeng/steptron/.worktrees/wyx/full_vocab/steptron/model/llama3_model.py:1090)，完成 teacher 原有 norm 和 dtype 转换，然后缓存。

两者按样本来源使用对应 teacher，不把五个 teacher 的分布先平均。对于相同参数、输入、有效词表、温度、mask 和外层权重，两者的 raw reverse-KL 数学目标一致；这不保证浮点计算逐 bit 相同，也不保证生产训练 wrapper 完全一致。

## 逐项对照

| 项目 | 当前 NeMo-RL | wyx/full_vocab |
| --- | --- | --- |
| 持有的 hidden rows | canonical `[B,S,H]`，包含完整 input trajectory 的 prompt 和 response；padding 按原始宽度处理 | 只抽取 response rows；按 original row ID、teacher source、student CP owner 存储 |
| CPU／通信路径 | Teacher 先恢复 SP/CP canonical 布局，经 TransferQueue `simple` backend 的 8 个 storage units 交给 student；student 再切自己的 CP window | TP0 持有 pinned CPU cache；CP all-to-all 将 response rows 发往 student CP owner；actor 每块 H2D，再在 TP group broadcast |
| Teacher projection | student loss preparation 先生成 `[B,S_local,V/TP]`，然后交给 KL | actor 按 source 选 response rows，每块生成 `[C,V/TP]` |
| Student projection | native student forward 先生成完整 packed-sequence vocabulary logits，sequence wrapper 再拆分 loss | model 返回 deferred LM-head output，actor loss 按 response 块执行 student norm/head |
| chunk_size 的边界 | KL/log-softmax FP32 临时 working set | hidden transfer、两端 projection、distribution loss，以及 backward 重放 |
| backward 保存 | KL 自定义 autograd 保存完整 student 和 teacher logits；backward 分块重算 log-softmax，但输出完整 student logits gradient | non-reentrant activation checkpoint 只保留 hidden 块等小输入；重放时重新读 teacher hidden、执行两次 head projection 和分布计算 |
| Teacher head 来源 | 从各 teacher checkpoint 加载词表分片，plain `torch.matmul` 重建 | 拷贝原 teacher leaf projection 的实现、参数与必要 buffers，并冻结；live teacher 指向 resident storage，避免稳定态重复 head 权重 |
| Head 生命周期 | 训练开始把全部五个 head 分片移到 GPU，训练阶段后 offload 到 CPU；`offload` 不表示每次只驻留一个 head | 实验 `max_resident_teachers=5`；五个 frozen head 常驻，不是 GPU 单 head 轮换 |
| 当前 head 精度 | payload FP16；teacher `fp32_lm_head=true`，student 缓存的 teacher head 分片为 FP32，输入转 FP32 后 matmul | hidden cache 使用模型 params_dtype；此实验继承 BF16 参数，原 projection 使用 FP32 output 的 native Optimus GEMM 路径 |
| 并行契约 | teacher TP2/PP1/CP1 与 student TP2/PP4/CP1 独立；student SP=true，capture 会 undo SP；各 teacher hidden width 必须一致 | 拒绝 independent teacher_model_cfg；teacher/student 共享 model config；要求 SP=false、deferred head；当前实验 TP1/PP6/VPP2/CP8/EP8 |
| Raw loss 能力 | 此路径 reverse KL | registry 支持 reverse KL、forward KL、JSD；保留实验选择 reverse KL |
| 外层训练行为 | 以 token/sample mask 做 token mean；当前不做 IS，full-vocab validator 拒绝 TIS 参数；仍经过既有 scoring/advantage 阶段 | actor 同步获得 sampled logprobs；loss 可应用 token/turn TIS gate、teacher weight、loss_scale；保留实验启用 inference-training IS；flow 没有独立 actor-logprob/advantage 阶段 |
| 可选 loss kernel | 当前 chunked distributed Torch RKL | 可选 `triton_rkl`，但保留实验继承默认 `torch`；不能把 Triton 优化算作已开启 |

源码入口：

- NeMo canonical payload：[train.py](/data/ycfeng/sota-engine/RL/nemo_rl/models/megatron/train.py:1001)，尤其 1077–1177 行；投影 [loss/utils.py](/data/ycfeng/sota-engine/RL/nemo_rl/algorithms/loss/utils.py:152)，CP 对齐在 315–368 行。
- NeMo 先重建再 KL：[loss_input.py](/data/ycfeng/sota-engine/RL/nemo_rl/algorithms/loss/loss_input.py:240)；完整保存和 backward：[model_utils.py](/data/ycfeng/sota-engine/RL/nemo_rl/distributed/model_utils.py:2538)、2667–2705 行。
- NeMo head loading/dtype/lifecycle：[megatron_policy_worker.py](/data/ycfeng/sota-engine/RL/nemo_rl/models/policy/workers/megatron_policy_worker.py:2834)、2932–2972 行；loss wrapper：[loss_functions.py](/data/ycfeng/sota-engine/RL/nemo_rl/algorithms/loss/loss_functions.py:1188)。
- StepTron response-only CP routing、head 注册、按块读回：[teacher_supervision.py](/data/ycfeng/steptron/.worktrees/wyx/full_vocab/steptron/core/mopd/teacher_supervision.py:200)；pinned cache/budget：[teacher_state.py](/data/ycfeng/steptron/.worktrees/wyx/full_vocab/steptron/core/mopd/teacher_state.py:178)。
- StepTron response/source chunking 和外层 loss：[actor_loss.py](/data/ycfeng/steptron/.worktrees/wyx/full_vocab/steptron/core/mopd/actor_loss.py:43)；projection/distribution checkpoint：[full_vocab_distillation.py](/data/ycfeng/steptron/.worktrees/wyx/full_vocab/steptron/model/common/full_vocab_distillation.py:48)。
- StepTron contract：[exp/mopd.py](/data/ycfeng/steptron/.worktrees/wyx/full_vocab/steptron/exp/mopd.py:417)、[mopd_trainer.py](/data/ycfeng/steptron/.worktrees/wyx/full_vocab/steptron/core/trainers/mopd_trainer.py:403)；阶段 [mopd_train_step_flow.py](/data/ycfeng/steptron/.worktrees/wyx/full_vocab/steptron/core/trainers/mopd_train_step_flow.py:82)；实验赋值 [experiment](/data/ycfeng/steptron/.worktrees/wyx/full_vocab/playground/users/getianyi/mopd/step3p6/step3p6_mopd_full_vocab_rkl_r3_it100_ctx128k_c128.py:569)。

## 同维度显存算术

采用当前 1.5B 任务的 V=151936、H=1536、TP2、CP1，假设一个完整 65536-row 序列。以下是单张量大小，不是峰值显存测量，也不表示每个实际 microbatch 都有 65536 rows。

| 张量 | 大小 |
| --- | --- |
| 全序列 `[65536,1536]` FP16 hidden | 192 MiB |
| 同 dtype 完整 logits／hidden 存储比 V/H | 98.9167 倍 |
| FP32 logits／FP16 hidden 存储比 | 197.8333 倍 |
| 单个 `[65536,75968]` FP32 teacher logits 分片 | 18.546875 GiB |
| 两张同形状 FP32 student/teacher logits | 37.09375 GiB，梯度与其他临时量另计 |
| 一个 `[128,75968]` FP32 logits 块 | 37.09375 MiB |
| 当前一个 teacher FP32 head `[75968,1536]` | 0.434692 GiB |
| 当前五个 teacher FP32 head 分片合计 | 2.173462 GiB／student output-stage TP rank |

`wyx/full_vocab` 若处理相同模型和 TP2，可将每张 live logits 限制到 128-row 块；当前长序列对应单张量 512 倍的大小差。它还只缓存 response rows，因此 CPU cache 的进一步节约取决于 response／input 比例。相反，backward 多做两端 head projection、分布计算和 teacher hidden 读回，不能从显存大小直接推出速度提高。

`wyx/full_vocab` 自身实验 H=4096、V=128896、BF16 parameters、TP1：same-dtype V/H=31.46875，一张 BF16 head 为 0.983398 GiB，五张为 4.916992 GiB／output-stage GPU。这个数值与当前 1.5B 的 2.17 GiB 不可直接用来判断算法优劣，模型和 dtype 不同。

## 本轮 CPU 验证

初次完整双 repo import 用 NeMo Python3.13 runtime 超过 60s 超时（exit124），没有 numerical result，未作为机制 FAIL。原工具输出保留了 transformers compatibility 与 MLflow import 信息。随后最小化到纯数学 primitives，避免加载整套 model runtime；没有修改依赖或生产源码。

精确命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/ycfeng/steptron/.worktrees/wyx/full_vocab TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 60s /data/ycfeng/tmp/d3-mopd-execution-env/bin/python /data/ycfeng/tmp/nemo-mopd-20261112/compare_full_vocab_storage_20261115.py > /data/ycfeng/tmp/nemo-mopd-20261112/compare-full-vocab-storage-20261115.log 2>&1
```

环境：Python3.10.6、Torch2.1.0+cu121、CUDA unavailable；脚本设置 RLIMIT_AS=30 GiB。NeMo 部分从当前 source AST 原样提取 reconstruction 与 RKL 必要函数/class，使用真实 one-rank Gloo collective；StepTron 调用实际 `distillation_block`。两端使用同一小型 FP32 Linear 权重，seed7，7 input rows、5 response rows、H5、V17、chunk3、温度1、单位权重、无 TIS。

| 验收项 | 条件 | 实测 | 判定 |
| --- | --- | --- | --- |
| Raw per-response KL 对照 | max absolute error ≤1e-6 | 0；两端 mean loss 均 0.19033749401569366 | PASS |
| NeMo 对 dense log_softmax oracle | max absolute error ≤1e-6 | 2.60770320892334e-7 | PASS |
| Student hidden gradient 对照 | max absolute error ≤1e-6 | 5.587935447692871e-9 | PASS |
| Student head gradient 对照 | max absolute error ≤1e-6 | 1.4901161193847656e-8 | PASS |
| NeMo KL saved tensors | 保存完整 vocabulary logits | 两张 `[1,7,17]` FP32 | PASS，确认完整驻留 |
| StepTron checkpoint saved tensors | 没有 vocabulary-width 激活保存 | `[3,1,5]` 与 `[2,1,5]` hidden、空 tensor、行索引/标量向量 | PASS |
| StepTron teacher hidden replay | 每块 forward/backward 各读一次 | 两个块共4次 read，顺序 forward A/B、backward B/A | PASS |
| Frozen teacher / prompt masking | 无 teacher gradient；prompt gradient 为0 | 均符合 | PASS |

完整结果：[evidence/full-vocab-nemo-steptron-storage-20261115.json](evidence/full-vocab-nemo-steptron-storage-20261115.json)。这是 primitive 级数值和 autograd 存储验证，不是两个生产模型、完整 optimizer、TP2/CP8 或 GPU kernel 的配对验证。

## 对当前性能问题的含义与后续动作

可确认的机制缺口是 response-only projection 和覆盖 LM-head 的 chunk/replay。它们可以解释训练侧为什么仍出现很大的 vocabulary 分配；已有 full-vocab hidden传输功能不能作为整个显存方案都已完成的证据。原 head reconstruction FP32 方案通过既有 GPU 数值验证，迁移时必须保留该精度验收，不能为复用 BF16 head/module 路径而默默改变数值契约。

但当前 matched successful ordinals14–28 的 627.8369s/update 中，ready wait 为540.1195s，student training 为66.3933s。wait 包含 rollout 和teacher enrichment；此比较没有隔离两者。response-only/cache/replay 是有依据的训练和 teacher-stage 优化候选，尚不能声称其能解释或消除整个约9–11分钟的慢速，更不能用旧 sampled-token 的54.742s历史值作为此分支full-vocab的baseline。

后续 E 保持：先验证原定 FP16 graph/eager rollout candidate，解决已知 teacher deployment utilization 问题，准备 step40/Adam22恢复；如调整 full-vocab 内存路径，以本次差异为范围，逐项保持 token alignment、teacher routing、FP32 reconstruction、完整词表、mask/objective 和梯度验收。当前200次成功更新与最终验证仍未完成。
