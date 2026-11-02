# roofline-lint 预注册（PREREG）

> 本文档是研究规格（registered-report 风格），不是结果报告。它声明本仓库**计划检验的假设、评测方案、成功标准与明确边界**，全部基于 `README.md`、`roofline_lint.py`、`test_roofline_lint.py` 中已落盘的真实内容。本文档不虚构任何未执行的实验。

## 1. 研究问题

一个零依赖 CLI 的 LLM 部署可行性检查器能否做到：

- **RQ1**：当配置无法运行（权重装不进显存 / batch 的 KV 超余量）时，是否拒绝打印吞吐数字，而非给出虚构的 tokens/s？
- **RQ2**：当分析公式本身写错时，工具能否证明自己的测试会失败（`--prove-red`），而非自证成同义反复（tautology）？
- **RQ3**：批量是否提高算术强度（权重每步只读一次）——即工具是否正确区分 memory-bound 与 compute-bound？

## 2. 预注册假设

| ID | 假设 | 检验方式 | 成功标准 |
|---|---|---|---|
| H1 | 不可行配置**拒绝打印**吞吐 | `--gpu rtx-3060m-6g --seq 4096 --batch 1`（权重 7.49 GiB > 6 GiB） | 输出「判定: 跑不起来」且**不印** tokens/s |
| H2 | 自测能失败（`--prove-red` 注入两个已知错误公式 → 套件转红） | `roofline_lint.py --prove-red` | 注入 batching bug（AI 停在 1.21 而非升至 12.2）与 textbook-PP bug（声称 1.94× 而真值 0.19×）均被断言捕获 |
| H3 | 批量提升算术强度（权重每步只读一次） | 默认 qwen3-4b @ a100-40g，batch 1→N | AI 从 1.21 上升（权重读不随 batch 增长，KV 读接管前 AI 上升） |
| H4 | 预设参数 byte-exact 校验 | `test_roofline_lint.py` 43 个单测 | 模型参数经 `params_of` 推导，与 HF safetensors `total_size` 逐字节一致 |

## 3. 评测方案

- **单测**：`python3 -m unittest test_roofline_lint -v`（43 个测试，含 byte-exact 预设校验 + `--prove-red` 钩子）。
- **自测**：`python3 roofline_lint.py --selftest`（17 个手算工作点）。
- **反证**：`python3 roofline_lint.py --prove-red`（注入两个已知错误公式，断言套件捕获两者）。
- **手动校验**：`--list` 列出模型/硬件预设与来源（`README.md` 的 Data provenance 表）。

## 4. 已声明边界（证明什么 / 不证明什么）

**证明什么（已实测）**：
- 不可行配置拒绝打印吞吐（权重 7.49 GiB > 6 GiB → 判定跑不起来；batch=128 的 KV 72.00 GiB > 余量 32.51 GiB → 不印）。
- 自测能失败（`--prove-red` 注入 batching bug 与 textbook-PP bug，断言捕获两者，AI 1.21 vs 12.2、PP 1.94× vs 0.19×）。
- 预设参数 byte-exact 校验（43 个单测，参数与 HF safetensors `total_size` 一致）。

**不证明什么（诚实边界）**：
- 分析下界，不是 benchmark：峰值规格 × 算子数；真实内核只达峰值的一部分，调度/启动开销/内存系统争用未建模。
- 只聚焦 decode；prefill 只作为 regime 指示器（权重主导近似）。
- `--tp` 是理想化分割；AllReduce 界是教科书理想环，未折入 step-time 下限（真实总线带宽与重叠超出范围）。
- `--pp` 是字节记账模型，不是调度器；未建模 stage 间通信、交错 1F1B 调度、层数需整除（不整除时报告警告）。
- `--kv-block` 只建模碎片；未建模 block table、copy-on-write 共享、prefix caching、外部碎片。
- MoE 读取是近似（`min(batch·top_k, E)` 专家、平衡路由；真实 router 不平衡、专家权重不可任意跨 TP 分割）。
- 密集注意力（标准 GQA MHA）；MLA 与 sliding-window 有效 KV 未建模。
- 显存 = 权重 + KV 仅此；activations、CUDA context、碎片、prefill 瞬态尖峰不在拟合检查内——真实 serving 需高于此下限的余量。

## 5. 成功标准（预先声明）

H1 与 H2 的**联合**成功 = 不可行配置拒绝打印 + 自测能失败（反证非装饰）。H3 成功 = 工具正确区分 memory-bound/compute-bound（批量提升 AI）。本文档声明的成功**不**包括「预测真实 GPU 利用率」——那明确排除（H4 只证预设参数 byte-exact，不证实测性能）。

## 6. 预注册时间戳与状态

- 本文档为**回顾性预注册**：反映仓库既有已落盘内容（`README.md`、`roofline_lint.py`、`test_roofline_lint.py`），不做前瞻性承诺。
- 目的：把「假设 + 评测方案 + 成功标准 + 边界」写清楚，供评审判断本仓库到底证明/不证明什么。
