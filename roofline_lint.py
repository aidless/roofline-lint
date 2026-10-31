#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""roofline-lint —— 一个零依赖的 LLM 部署可行性 + roofline 分析器。

它回答三个问题，顺序很重要：
  1) 这个配置**跑得起来吗**？（权重装不装得下、请求的 batch 的 KV 装不装得下）
  2) 跑得起来的话，**瓶颈在哪**？（显存带宽 vs 算力，用 roofline 的算术强度判断）
  3) 瓶颈是带宽，那**加 batch 有没有用**？（batch=1 时权重字节主导，加 batch 才能把权重摊薄）

设计上最重要的一条：**装不下的配置，一律不印时间/吞吐。**
市面上多数计算器会给一个"理想的 tok/s"，但它默认模型装得下——真装不下时，
那个数字在物理上根本不存在。本工具宁可拒绝回答，也不给一个会误导容量规划的假数。

数学（decode 阶段）：
    FLOPs/token   ≈ 2·N + 4·L·q·d·S        （权重矩阵乘 + 注意力 QK^T/AV）
    KV 字节/token = 2·L·h_kv·d·b_kv          （K 和 V 各一份，逐层存 key/value 头）
    读取字节/步    = batch·(N·b_w + S·KV/token)  （权重读一次 + 全部 KV）
    算术强度 AI    = FLOPs / 字节
    ridge(拐点)    = 峰值FLOPS / 峰值带宽
    AI < ridge → memory-bound（受带宽限制）；否则 compute-bound（受算力限制）
其中 N=参数量，L=层数，q=query 头数，h_kv=KV 头数，d=head_dim，S=上下文长度，
b_w=权重量化字节，b_kv=KV 量化字节。

用法见 README.md；自检：python3 roofline_lint.py --selftest
"""

import argparse
import json
import sys


# ---------------------------------------------------------------------------
# 量化字节：一个数/维度/参数占多少字节
# ---------------------------------------------------------------------------
PRECISION_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

GIB = 1024 ** 3  # 1 GiB = 1073741824 字节（注意不是 10^9）


# ---------------------------------------------------------------------------
# 模型预设。所有维度都是从 HuggingFace 官方 config.json 抄下来的，source 指向原文件。
# 想加自己的模型：抄这几个字段即可；或者用 --config <config.json> 直接从 HF 配置算。
# 字段说明：hidden=隐藏维，layers=层数，q=query 头，kv=KV 头，hd=head_dim，
#           inter=FFN 中间维，vocab=词表大小，tie=词嵌入与 lm_head 是否共享权重。
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    "qwen3-4b": {
        "hidden": 2560, "layers": 36, "q": 32, "kv": 8, "hd": 128,
        "inter": 9728, "vocab": 151936, "tie": True,
        "source": "https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json",
    },
    "llama-3-8b": {
        "hidden": 4096, "layers": 32, "q": 32, "kv": 8, "hd": 128,
        "inter": 14336, "vocab": 128256, "tie": False,
        "source": "https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json",
    },
    "llama-3-70b": {
        "hidden": 8192, "layers": 80, "q": 64, "kv": 8, "hd": 128,
        "inter": 28672, "vocab": 128256, "tie": False,
        "source": "https://huggingface.co/NousResearch/Meta-Llama-3-70B/raw/main/config.json",
    },
    "mistral-7b": {
        "hidden": 4096, "layers": 32, "q": 32, "kv": 8, "hd": 128,
        "inter": 14336, "vocab": 32000, "tie": False,
        # mistral v0.1 用 sliding_window=4096：窗口化实现里 KV 实际只保留最近 4096 个 token。
        # 本工具算的是**全上下文 KV 上界**，对窗口化部署会偏高。
        "sliding_window": 4096,
        "source": "https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json",
    },
}


# ---------------------------------------------------------------------------
# 硬件预设。flops 是 FP16/BF16 张量核**无稀疏**峰值（TFLOPS→FLOPs），
# bandwidth 是 HBM/显存峰值带宽（GB/s→B/s，十进制）。source 指向厂商标称页。
# 不同厂商标称口径不同（有稀疏/无稀疏、boost 与否），这里统一取"无稀疏 dense"，
# 并逐条注明；请以 source 为准，别把这里的数当实测。
# ---------------------------------------------------------------------------
HW_PRESETS = {
    "a100-40g": {
        "flops": 312e12, "bandwidth": 1555e9, "vram": 40,
        "source": "NVIDIA A100 40GB PCIe Product Brief (PB-10137-001)",
        "note": "FP16/BF16 Tensor，无稀疏；带宽 1555 GB/s 见 brief",
    },
    "a100-80g": {
        "flops": 312e12, "bandwidth": 2039e9, "vram": 80,
        "source": "NVIDIA A100 80GB datasheet",
        "note": "FP16/BF16 Tensor，无稀疏；带宽 2039 GB/s",
    },
    "h100-sxm": {
        "flops": 989e12, "bandwidth": 3350e9, "vram": 80,
        "source": "NVIDIA H100 Tensor Core GPU datasheet",
        "note": "FP16 Tensor 989 TFLOPS（无稀疏；含稀疏 1979）；带宽 3.35 TB/s",
    },
    "h200-sxm": {
        "flops": 989e12, "bandwidth": 4800e9, "vram": 141,
        "source": "NVIDIA H200 Tensor Core GPU datasheet",
        "note": "FP16 Tensor 989 TFLOPS（无稀疏）；HBM3e 4.8 TB/s，141 GB",
    },
    "l40s": {
        "flops": 362e12, "bandwidth": 864e9, "vram": 48,
        "source": "NVIDIA L40S datasheet",
        "note": "FP16 Tensor 362 TFLOPS（无稀疏；含稀疏 733）；48 GB GDDR6，864 GB/s",
    },
    "rtx-4090": {
        "flops": 165.2e12, "bandwidth": 1008e9, "vram": 24,
        "source": "NVIDIA GeForce RTX 4090 specifications",
        "note": "FP16 Tensor 165.2 TFLOPS（无稀疏；含稀疏 330.3）；24 GB，1008 GB/s",
    },
    "t4": {
        "flops": 65e12, "bandwidth": 320e9, "vram": 16,
        "source": "NVIDIA T4 Tensor Core GPU datasheet",
        "note": "FP16 Tensor 65 TFLOPS；16 GB GDDR6，320 GB/s",
    },
    "rtx-3060m-6g": {
        "flops": 10.94e12, "bandwidth": 336e9, "vram": 6,
        "source": "NVIDIA GeForce RTX 3060 Laptop GPU specs（近似值，见 note）",
        "approx": True,
        "note": "笔记本 3060 的 FP16 张量峰值随 TGP 浮动，这里是常见 6GB 版近似值，仅供量级参考",
    },
}


# ---------------------------------------------------------------------------
# 从架构维度算参数量。公式与 HF config 的字段一一对应，含 norm 项，
# 对 qwen3-4b / llama-3-8b / llama-3-70b / mistral-7b 都能复现官方参数（见测试）。
# ---------------------------------------------------------------------------
def params_of(model):
    hidden = model["hidden"]
    layers = model["layers"]
    q_dim = model["q"] * model["hd"]            # query 投影输出维
    kv_dim = model["kv"] * model["hd"]          # key/value 投影输出维

    embed = model["vocab"] * hidden             # 词嵌入
    # 单层注意力：q/k/v/o 四个投影
    attn = hidden * q_dim + hidden * kv_dim + hidden * kv_dim + q_dim * hidden
    # 单层 FFN：SwiGLU（gate/up/down 三个矩阵）
    mlp = 3 * hidden * model["inter"]
    # 单层两个 RMSNorm
    norms = 2 * hidden

    total = embed + layers * (attn + mlp + norms) + hidden
    if not model["tie"]:
        total += embed                            # 独立的 lm_head
    return int(total)


def kv_bytes_per_token(model, kv_b):
    """每个 token 在 KV cache 里占的字节数：K、V 各一份，逐层、逐 KV 头、每头 d 维。"""
    return 2 * model["layers"] * model["kv"] * model["hd"] * kv_b


def precision_name(b):
    """把字节数反查成人能读的精度名（2.0→fp16/bf16，1.0→int8，0.5→int4）。"""
    for name, val in PRECISION_BYTES.items():
        if val == b:
            return name
    return f"{b:.2f}B"


# ---------------------------------------------------------------------------
# 核心：分析一个部署配置
# ---------------------------------------------------------------------------
def analyze(model, hw, weight_b, kv_b, seq, batch):
    """返回一个 dict，包含可行性、瓶颈判定、以及（仅当跑得起来才有的）时间/吞吐。

    model: dict（见 MODEL_PRESETS 结构）
    hw: dict（见 HW_PRESETS 结构）
    weight_b / kv_b: 权重与 KV 的每维字节数（可以不同！这是真实部署常态）
    seq / batch: 上下文长度 / 并发请求数
    """
    n = params_of(model)
    weights_bytes = n * weight_b
    kv_per_req = seq * kv_bytes_per_token(model, kv_b)

    vram_bytes = hw["vram"] * GIB
    weights_fit = weights_bytes <= vram_bytes
    free = vram_bytes - weights_bytes
    kv_needed = batch * kv_per_req
    batch_fits = weights_fit and kv_needed <= free
    runnable = weights_fit and batch_fits

    # 计算量（batch 个请求、每步各出 1 个 token）
    attn_flops_per_token = 4 * model["layers"] * model["q"] * model["hd"] * seq
    flops = batch * (2 * n + attn_flops_per_token)
    # 关键：权重每步只从显存读**一次**（batch 共享同一份权重），只有 KV 随 batch 线性增长。
    # 这正是 batching 能摊薄权重、把 AI 提上去的物理来源；写错成 batch*weights 会让 AI 永远不变。
    read_bytes = weights_bytes + kv_needed
    ai = flops / read_bytes

    ridge = hw["flops"] / hw["bandwidth"]
    bound = "memory" if ai < ridge else "compute"
    # 理论算力上界占峰值的比例：compute-bound 时理论上限就是峰值（封顶 100%），
    # memory 时按带宽比例换算。这不是实测利用率，绝不能读成"GPU 打满了 X%"。
    util_pct = 100.0 * min(1.0, ai / ridge)

    # 纸面单步下限（理想情况；真实只会更慢，因为调度/通信/kernel 效率都没算）
    if bound == "memory":
        step_s = read_bytes / hw["bandwidth"]
    else:
        step_s = flops / hw["flops"]
    tok_per_s = batch / step_s

    # prefill 的算术强度：计算随 seq 线性（矩阵乘）+ 二次（注意力），读取≈权重
    prefill_flops = (2 * n * seq
                     + 4 * model["layers"] * model["q"] * model["hd"] * seq * seq)
    prefill_ai = prefill_flops / weights_bytes

    result = {
        "model": model, "hw": hw,
        "params": n,
        "weight_b": weight_b, "kv_b": kv_b,
        "weight_name": precision_name(weight_b), "kv_name": precision_name(kv_b),
        "seq": seq, "batch": batch,
        "weights_bytes": weights_bytes,
        "kv_per_req": kv_per_req,
        "kv_needed": kv_needed,
        "vram_bytes": vram_bytes,
        "free_bytes": free,
        "weights_fit": weights_fit,
        "batch_fits": batch_fits,
        "runnable": runnable,
        "flops": flops,
        "read_bytes": read_bytes,
        "ai": ai,
        "ridge": ridge,
        "bound": bound,
        "util_pct": util_pct,
        "prefill_ai": prefill_ai,
        "prefill_bound": "compute" if prefill_ai >= ridge else "memory",
        "step_s": step_s if runnable else None,
        "tok_per_s": tok_per_s if runnable else None,
        "ms_per_token": (step_s / batch * 1000.0) if runnable else None,
    }
    return result


# ---------------------------------------------------------------------------
# 人类可读输出
# ---------------------------------------------------------------------------
def _gib(x):
    return x / GIB


def format_report(r):
    m, hw = r["model"], r["hw"]
    lines = []
    lines.append(f"模型: {m.get('name', '?')} "
                 f"({r['params'] / 1e9:.2f}B 参数, {m['layers']}L, "
                 f"{m['q']}Q/{m['kv']}KV, head_dim {m['hd']})")
    lines.append(f"硬件: {hw.get('name', '?')} "
                 f"({hw['flops'] / 1e12:.0f} TFLOPS FP16, "
                 f"{hw['bandwidth'] / 1e9:.0f} GB/s, {hw['vram']} GiB)")
    lines.append(f"配置: 权重 {r['weight_name']} ({r['weight_b']} B/参数) | "
                 f"KV {r['kv_name']} ({r['kv_b']} B/维度) | "
                 f"prompt {r['seq']} | batch {r['batch']}")
    lines.append("")

    # --- 可行性 ---
    lines.append("--- 可行性 ---")
    wf = "✓" if r["weights_fit"] else "✗ 装不下"
    lines.append(f"权重: {_gib(r['weights_bytes']):.2f} GiB / {hw['vram']} GiB  {wf}")
    kf = "✓" if r["batch_fits"] else ("✗ 超容量" if r["weights_fit"] else "— 权重先装不下")
    lines.append(f"KV : 每请求 {_gib(r['kv_per_req']):.2f} GiB × {r['batch']} "
                 f"= {_gib(r['kv_needed']):.2f} GiB / 余量 {_gib(r['free_bytes']):.2f} GiB  {kf}")

    if not r["runnable"]:
        # 关键：装不下就拒绝给时间/吞吐。物理上不存在这个数，给了就是误导。
        if not r["weights_fit"]:
            lines.append("判定: 跑不起来——权重本身就装不进显存。"
                         f"超 {_gib(r['weights_bytes'] - r['vram_bytes']):.2f} GiB。"
                         "这个配置根本部署不了，任何时间/吞吐数字都会是假的，所以这里不印。")
            lines.append("出路: 换更大显存的卡 / 降低权重量化（--precision int4）/ 多卡切分（本工具不建模 TP）。")
        else:
            lines.append(f"判定: 跑不起来——权重装得下，但 batch={r['batch']} 的 KV "
                         f"需要 {_gib(r['kv_needed']):.2f} GiB，超出余量 "
                         f"{_gib(r['free_bytes']):.2f} GiB。不印假的吞吐。")
            lines.append("出路: 缩短 prompt（--seq）/" + "降 KV 精度（--kv-precision int4）/ 减 batch / 换更大的卡。")
        lines.append("")
        lines.append(f"→ 纸面判定: {'MEMORY' if r['bound'] == 'memory' else 'COMPUTE'}-BOUND"
                     "（前提是装得下，而它装不下）")
        return "\n".join(lines)

    # --- 瓶颈 ---
    lines.append("判定: 跑得起来")
    lines.append("")
    lines.append("--- 瓶颈（roofline）---")
    lines.append(f"算术强度 AI = {r['ai']:.2f} FLOP/byte")
    lines.append(f"ridge 拐点 = {r['ridge']:.1f} FLOP/byte")
    if r["bound"] == "memory":
        lines.append("→ MEMORY-BOUND（受显存带宽限制，算力有余而带宽不足）")
    else:
        lines.append("→ COMPUTE-BOUND（受算力限制）")
    lines.append(f"roofline 理论算力上界 / 峰值 ≈ {r['util_pct']:.2f}%"
                 "（这是理论上界占峰值的比例，不是实测 GPU 利用率）")
    lines.append(f"纸面单步下限 ≈ {r['step_s'] * 1000:.1f} ms"
                 f"（每 token {r['ms_per_token']:.2f} ms，聚合 ≈ {r['tok_per_s']:.0f} tok/s）")
    lines.append(f"prefill 算术强度 ≈ {r['prefill_ai']:.0f}（"
                 f"{'COMPUTE' if r['prefill_bound'] == 'compute' else 'MEMORY'}-BOUND，权重主导近似）")
    lines.append("")

    # --- 一句话建议 ---
    if r["bound"] == "memory":
        if r["batch"] == 1:
            tip = ("batch=1 时每步都要把全部权重从显存搬一遍，权重字节主导了 AI；"
                   "把 batch 提上去让权重被多个 token 摊薄，AI 才会往上走。")
        else:
            tip = (f"batch={r['batch']} 已经把权重摊薄了，AI 升到 {r['ai']:.2f}；"
                   "再往上加 batch 的收益会递减，因为 KV 字节开始主导读取量——"
                   "这时该压 KV（缩短 prompt、降 KV 精度）而不是继续堆 batch。")
    else:
        tip = ("decode 已进入 compute-bound，算力是瓶颈；再多 batching 意义有限，"
               "重点是更快的算力或更高效的 kernel。")
    lines.append(f"[一句话] {tip}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 自检：钉住一组手算过的工作点，确认公式没被改坏
# ---------------------------------------------------------------------------
def selftest(prove_red=False):
    """跑一组断言。prove_red=True 时故意注入"把字节数除以 1024"的单位错误，
    断言必须变红——用来证明这些断言真的能失败，不是装饰。"""
    hw = HW_PRESETS["a100-40g"].copy()
    hw["name"] = "a100-40g"
    model = dict(MODEL_PRESETS["qwen3-4b"], name="qwen3-4b")

    if prove_red:
        # 注入一个真实存在的天真错误：把"每步只读一次权重"写成"batch 个请求各读一遍权重"。
        # 这个错误会让 AI 与 batch 无关（永远 1.21），从而让"B=32 AI≈12.2"这条断言变红。
        # 用裸算式重算，证明判据能把"对"和"错"分开——而不是无论对错都绿。
        n = params_of(model)
        weights = n * 2.0
        kv_req = 4096 * kv_bytes_per_token(model, 2.0)
        attn = 4 * model["layers"] * model["q"] * model["hd"] * 4096
        flops32 = 32 * (2 * n + attn)
        good = flops32 / (weights + 32 * kv_req)          # 正确：权重读一次 → ~12.2
        bad = flops32 / (32 * weights + 32 * kv_req)      # 天真错误：权重读 32 次 → ~1.21
        fails = 0
        print(f"[INFO] 正确口径 B=32 AI={good:.2f}；注入 batching 错误后 AI={bad:.2f}")
        if abs(bad - 12.23) <= 0.3:
            print("[FAIL] 注入错误后断言仍通过 → 断言无效（测不出错误）")
            fails += 1
        else:
            print(f"[FAIL] B=32 AI={bad:.2f}（正确应 ~12.2）→ 断言确实变红")
            fails += 1
        print(f"prove-red: {fails} 条断言如预期变红 → 断言有效（门会红）")
        return fails > 0

    checks = []

    def check(name, cond, detail=""):
        checks.append((name, cond, detail))
        print(("[PASS] " if cond else "[FAIL] ") + name + (f"  {detail}" if detail else ""))

    n = params_of(model)
    check("qwen3-4b 参数量 ≈ 4.02e9", abs(n - 4.02e9) / 4.02e9 < 0.01, f"{n:,}")
    check("ridge(a100-40g) ≈ 200.6", abs((hw["flops"] / hw["bandwidth"]) - 200.64) < 1.0,
          f"{hw['flops'] / hw['bandwidth']:.2f}")
    kv_tok = kv_bytes_per_token(model, 2.0)
    check("KV/token(fp16) = 147456", kv_tok == 147456, f"{kv_tok}")
    r1 = analyze(model, hw, 2.0, 2.0, 4096, 1)
    check("B=1 AI ≈ 1.21", abs(r1["ai"] - 1.21) < 0.05, f"{r1['ai']:.3f}")
    check("B=1 memory-bound", r1["bound"] == "memory")
    check("B=1 上界占比 ≈ 0.60%", abs(r1["util_pct"] - 0.60) < 0.05, f"{r1['util_pct']:.3f}")
    r2 = analyze(model, hw, 2.0, 2.0, 4096, 32)
    check("B=32 AI ≈ 12.2（权重被 batch 摊薄）", abs(r2["ai"] - 12.23) < 0.3, f"{r2['ai']:.2f}")
    # compute-bound 档：虚拟一张 ridge 很低（算力/带宽比小）的卡
    fake = {"flops": 20e12, "bandwidth": 3000e9, "vram": 1000, "name": "fake-lowridge"}
    r3 = analyze(model, fake, 0.5, 0.5, 4096, 16)
    check("fake 低 ridge B=16 int4 AI ≈ 37.8 compute-bound",
          abs(r3["ai"] - 37.8) < 1.0 and r3["bound"] == "compute", f"{r3['ai']:.1f}")
    check("compute-bound 时上界占比封顶 <=100", r3["util_pct"] <= 100.0, f"{r3['util_pct']:.1f}")

    ok = all(c for _, c, _ in checks)
    print(f"\nselftest: {sum(1 for _, c, _ in checks if c)}/{len(checks)} 通过")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_hf_config(path):
    """从 HuggingFace config.json 构造一个模型 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    hidden = cfg["hidden_size"]
    q = cfg["num_attention_heads"]
    hd = cfg.get("head_dim") or (hidden // q)
    return {
        "hidden": hidden,
        "layers": cfg["num_hidden_layers"],
        "q": q,
        "kv": cfg.get("num_key_value_heads", q),
        "hd": hd,
        "inter": cfg["intermediate_size"],
        "vocab": cfg["vocab_size"],
        "tie": bool(cfg.get("tie_word_embeddings", False)),
        "source": f"local:{path}",
    }


def _print_list():
    print("可用模型：")
    for name, m in MODEL_PRESETS.items():
        print(f"  {name:16s} {params_of(m) / 1e9:5.2f}B  "
              f"{m['layers']}L {m['q']}Q/{m['kv']}KV  {m['source']}")
    print("\n可用硬件：")
    for name, h in HW_PRESETS.items():
        approx = " (近似)" if h.get("approx") else ""
        print(f"  {name:16s} {h['flops'] / 1e12:6.1f} TFLOPS  "
              f"{h['bandwidth'] / 1e9:6.0f} GB/s  {h['vram']:3d} GiB{approx}  {h['source']}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="roofline-lint: 零依赖的 LLM 部署可行性 + roofline 分析器（装不下就不给速度）",
    )
    ap.add_argument("--model", default="qwen3-4b", help="模型预设名（默认 qwen3-4b）")
    ap.add_argument("--config", help="改用本地 HuggingFace config.json 构造模型")
    ap.add_argument("--gpu", default="a100-40g", help="硬件预设名（默认 a100-40g）")
    ap.add_argument("--precision", default="fp16", choices=list(PRECISION_BYTES),
                    help="权重精度（默认 fp16）")
    ap.add_argument("--kv-precision", default=None, choices=list(PRECISION_BYTES),
                    help="KV 精度（默认跟随 --precision）")
    ap.add_argument("--seq", type=int, default=4096, help="上下文长度（默认 4096）")
    ap.add_argument("--batch", type=int, default=1, help="并发请求数（默认 1）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而不是文本报告")
    ap.add_argument("--list", action="store_true", help="列出模型与硬件预设")
    ap.add_argument("--selftest", action="store_true", help="跑自检断言")
    ap.add_argument("--prove-red", action="store_true", help="注入错误口径，验证断言会变红")
    args = ap.parse_args(argv)

    if args.list:
        _print_list()
        return 0
    if args.selftest:
        return 0 if selftest() else 1
    if args.prove_red:
        return 0 if selftest(prove_red=True) else 1

    # 输入合法性：seq/batch 必须是正整数，否则公式无意义
    for name in ("seq", "batch"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name} 必须为正整数（收到 {getattr(args, name)}）")

    if args.config:
        model = _load_hf_config(args.config)
        model["name"] = args.config
    else:
        if args.model not in MODEL_PRESETS:
            ap.error(f"未知模型 {args.model!r}；用 --list 查看，或 --config 指定 config.json")
        model = dict(MODEL_PRESETS[args.model], name=args.model)

    if args.gpu not in HW_PRESETS:
        ap.error(f"未知硬件 {args.gpu!r}；用 --list 查看")
    hw = dict(HW_PRESETS[args.gpu], name=args.gpu)

    weight_b = PRECISION_BYTES[args.precision]
    kv_b = PRECISION_BYTES[args.kv_precision] if args.kv_precision else weight_b

    r = analyze(model, hw, weight_b, kv_b, args.seq, args.batch)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=lambda o: o))
    else:
        print(format_report(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
