# roofline-lint：LLM 部署可行性的诚实下界检查器

## 摘要（Abstract）

**背景**：把 LLM 部署到 GPU 前，需要回答「这个配置能不能跑起来、瓶颈在哪、批量是否值得」。现有 roofline 与 KV-cache 计算器（如 `LLM-Viewer`、`pochenai/llm-inference-calculator`）大多聚焦「算出一个漂亮的 tokens/s 数字」，但当配置不可行时，打印一个虚构的吞吐数字会误导决策。

**方法**：`roofline-lint` 是一个零依赖 CLI，围绕两条规则构建：(1) 如果配置无法运行，唯一诚实的答案是「跑不起来」——拒绝打印任何吞吐数字；(2) 一个无法失败的测试证明不了什么——`--prove-red` 注入两个已知错误公式（「每请求都读一遍权重」的 batching bug、对 memory-bound decode 误用 compute-bound pipeline 公式的 textbook-PP bug），断言套件必须转红。它回答三个问题：是否装得下（权重 ≤ 显存、batch×KV/请求 ≤ 余量）；装得下则瓶颈在 memory-bandwidth 还是 compute（roofline 算术强度）；若 bandwidth-bound，提高 batch 是否有帮助（batch=1 时权重主导，收益递减）。

**结果**：在默认配置（qwen3-4b @ a100-40g, fp16, seq=4096, batch=1）上，工具正确判定 memory-bound（AI=1.21 FLOP/byte < ridge=200.6），纸面单步下限 ≈5.6ms（聚合 ≈180 tok/s）。对不可行配置（权重 7.49 GiB > 6 GiB；batch=128 的 KV 72.00 GiB > 余量 32.51 GiB），工具拒绝打印吞吐。`--prove-red` 注入 batching bug 时 AI 停在 1.21（正确 12.2）、注入 textbook-PP bug 时声称 1.94×（正确 0.19×，10.1× 差距且符号相反）——两者均被 43 个单测的断言捕获。

**边界**：本工具给出的是**解析下界**（峰值规格 × 算子数），不是实测 benchmark；只聚焦 decode（prefill 仅作 regime 指示）；`--tp` 为理想化分割、`--pp` 为字节记账模型、`--kv-block` 只建模碎片、MoE 读取为近似、密集注意力（MLA/sliding-window 未建模）、显存只计权重+KV。它回答「什么**不可能**」的成本极低，让 GPU 小时只花在**可能**的配置上。

**关键词**：LLM 部署可行性；roofline 分析；memory-bound；KV cache；fail-closed；反证测试
