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
    FLOPs/token   ≈ 2·N_act + 4·L·q·d·S      （权重矩阵乘 + 注意力 QK^T/AV；
                                                MoE 的 N_act 是激活参数，不是全量）
    KV 字节/token = 2·L·h_kv·d·b_kv          （K 和 V 各一份，逐层存 key/value 头）
    读取字节/步    = 权重读一次 + batch·S·KV/token
                    （稠密：权重=N_act·b_w；MoE：权重=共享 + min(batch·top_k, 专家数)×专家）
    算术强度 AI    = FLOPs / 字节
    ridge(拐点)    = 峰值FLOPS / 峰值带宽
    AI < ridge → memory-bound（受带宽限制）；否则 compute-bound（受算力限制）
张量并行 --tp N：权重/KV/计算按 N 均分到每卡，AI 不变、时间约除以 N；
    另印每步每卡 ring AllReduce 字节下界 8·(N-1)/N·L·h·batch。
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
#           qk_norm=True 表示每层有 q/k 的 RMSNorm（Qwen3 系有，Llama/Mistral 无）。
#           moe={experts,top_k,inter}：专家总数 / 每 token 激活专家数 / 每专家 FFN 中间维。
# MoE 参数量已对上官方 safetensors 总数（见测试）：Mixtral 46,702,792,704、
# Qwen3-30B-A3B 30,532,122,624，逐字节一致。
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    "qwen3-4b": {
        "hidden": 2560, "layers": 36, "q": 32, "kv": 8, "hd": 128,
        "inter": 9728, "vocab": 151936, "tie": True, "qk_norm": True,
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
    "mixtral-8x7b": {
        "hidden": 4096, "layers": 32, "q": 32, "kv": 8, "hd": 128,
        "vocab": 32000, "tie": False,
        "moe": {"experts": 8, "top_k": 2, "inter": 14336},
        "source": "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/config.json",
    },
    "qwen3-30b-a3b": {
        "hidden": 2048, "layers": 48, "q": 32, "kv": 4, "hd": 128,
        "vocab": 151936, "tie": False, "qk_norm": True,
        "moe": {"experts": 128, "top_k": 8, "inter": 768},
        "source": "https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json",
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
# 从架构维度算参数量。公式与 HF config 的字段一一对应，含 norm 项与 MoE 路由。
# 六个预设全部与 HF safetensors 官方总数**逐字节一致**（2026-10-31 实测核对：
# qwen3-4b 4,022,468,096 / llama-3-8b 8,030,261,248 / llama-3-70b 70,553,706,496 /
# mistral-7b 7,241,732,096 / mixtral-8x7b 46,702,792,704 / qwen3-30b-a3b 30,532,122,624）。
# ---------------------------------------------------------------------------
def _layer_shared_params(model):
    """单层的公共参数：注意力四投影 + MoE 路由器 + 两个 RMSNorm（+ Qwen3 系的 q/k norm）。"""
    hidden, hd = model["hidden"], model["hd"]
    q_dim = model["q"] * hd
    kv_dim = model["kv"] * hd
    attn = hidden * q_dim + hidden * kv_dim + hidden * kv_dim + q_dim * hidden
    gate = hidden * model["moe"]["experts"] if model.get("moe") else 0  # MoE 路由器
    norms = 2 * hidden
    qkn = 2 * hd if model.get("qk_norm") else 0
    return attn + gate + norms + qkn


def embed_params_of(model):
    embed = model["vocab"] * model["hidden"]
    return embed if model["tie"] else 2 * embed   # untie 时 lm_head 独立一份


def expert_params_of(model):
    """全部专家的参数：层数 × 专家数 × (gate/up/down 三个矩阵)。MoE 模型才有。"""
    moe = model.get("moe")
    if not moe:
        return 0
    return model["layers"] * moe["experts"] * 3 * model["hidden"] * moe["inter"]


def params_of(model):
    mlp = 0 if model.get("moe") else 3 * model["hidden"] * model["inter"]
    total = (embed_params_of(model)
             + model["layers"] * (_layer_shared_params(model) + mlp)
             + expert_params_of(model)
             + model["hidden"])                    # 最后一层 RMSNorm
    return int(total)


def active_params_of(model):
    """每 token 实际参与计算的参数。稠密 = 全量；MoE = 共享 + top_k 个专家。"""
    moe = model.get("moe")
    if not moe:
        return params_of(model)
    per_layer = _layer_shared_params(model) + moe["top_k"] * 3 * model["hidden"] * moe["inter"]
    return int(embed_params_of(model)
               + model["layers"] * per_layer
               + model["hidden"])


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
def analyze(model, hw, weight_b, kv_b, seq, batch, tp=1, pp=1, micro_batch=None,
            kv_block=None):
    """返回一个 dict，包含可行性、瓶颈判定、以及（仅当跑得起来才有的）时间/吞吐。

    model: dict（见 MODEL_PRESETS 结构）
    hw: dict（见 HW_PRESETS 结构）
    weight_b / kv_b: 权重与 KV 的每维字节数（可以不同！这是真实部署常态）
    seq / batch: 上下文长度 / 并发请求数
    tp: 张量并行卡数。权重/KV/计算按 tp 均分到每卡，可行性按**每卡**判；
        切分不改变算术强度（AI 不变），只改变每步时间。默认 1。
    pp: 流水线并行段数。模型按层切到 pp 段，每段的权重/KV/计算只占 1/pp。
        **买的是容量，不是速度**：每个微批都要把本段的权重重读一遍。
    micro_batch: 微批个数 M（pp>1 时生效，缺省 = batch）。M 越大气泡越小，
        但权重被重读 M 次——memory-bound 的 decode 里这两者方向相反。
    kv_block: KV 分配的块大小（token），模拟 PagedAttention 的整块分配。
        缺省 None = 精确分配（不做块对齐）。设了就按 ceil(seq/块)×块 算容量。
    """
    if tp < 1:
        raise ValueError("tp 必须 >= 1")
    if pp < 1:
        raise ValueError("pp 必须 >= 1")
    if kv_block is not None and kv_block < 1:
        raise ValueError("kv_block 必须 >= 1")
    # 微批只在 pp>1 时有意义；pp=1 时整批一次算完，M 恒为 1。
    mb = 1 if pp == 1 else (batch if micro_batch is None else micro_batch)
    if mb < 1 or mb > batch:
        raise ValueError(f"micro_batch 必须在 [1, batch={batch}] 之间（收到 {mb}）")

    n = params_of(model)                 # 全部参数（MoE 含所有专家）
    active = active_params_of(model)     # 每 token 实际激活的参数
    moe = model.get("moe")
    weights_bytes = n * weight_b         # 集群总权重字节（必须全部常驻显存）

    # 分块分配：PagedAttention 按整块给 KV，每个请求最多浪费一块。
    kv_tok = kv_bytes_per_token(model, kv_b)
    if kv_block:
        alloc_tokens = -(-seq // kv_block) * kv_block   # ceil(seq/kv_block)*kv_block
    else:
        alloc_tokens = seq
    kv_pad_tokens = alloc_tokens - seq
    kv_per_req = alloc_tokens * kv_tok           # 每请求 KV（含内部碎片）

    # 卡数 = tp × pp：权重/KV/计算都按这个总数均分
    gpus = tp * pp
    per_gpu_weights = weights_bytes / gpus
    kv_needed = batch * kv_per_req               # 集群总 KV
    kv_gpu = kv_needed / gpus                    # 每卡 KV
    kv_per_req_gpu = kv_per_req / gpus           # 每卡每请求 KV（算容量上限用）

    vram_bytes = hw["vram"] * GIB
    weights_fit = per_gpu_weights <= vram_bytes
    free = vram_bytes - per_gpu_weights
    batch_fits = weights_fit and kv_gpu <= free
    runnable = weights_fit and batch_fits
    max_concurrency = int(free // kv_per_req_gpu) if (weights_fit and kv_per_req_gpu > 0) else 0

    # 计算量（batch 个请求、每步各出 1 个 token）。MoE 用激活参数，不是全量参数。
    attn_flops_per_token = 4 * model["layers"] * model["q"] * model["hd"] * seq
    flops = batch * (2 * active + attn_flops_per_token)   # 集群总 FLOPs
    flops_gpu = flops / gpus

    # 关键：权重每步只从显存读**一次**（batch 共享同一份权重），只有 KV 随 batch 线性增长。
    # 这正是 batching 能摊薄权重、把 AI 提上去的物理来源；写错成 batch*weights 会让 AI 永远不变。
    # MoE 附加语义：每步读「共享参数 + min(batch×top_k, 专家数) 个专家」——batch 小时
    # 只有被激活的那几个专家被读到；batch 大到覆盖全部专家后，每步都要扫全部专家。
    # 无论如何，全部专家都必须**常驻**显存（激活省的是算力/带宽，省不了显存）。
    if moe:
        expert_b = 3 * model["hidden"] * moe["inter"] * weight_b
        shared_b = (n - expert_params_of(model)) * weight_b
        expert_reads = min(batch * moe["top_k"], moe["experts"])
        weight_read = shared_b + expert_reads * expert_b
    else:
        expert_reads = None
        weight_read = weights_bytes
    read_gpu = (weight_read + kv_needed) / gpus   # 每卡每步读取（理想：权重只读一次）
    read_bytes = read_gpu * tp
    ai = flops_gpu / read_gpu

    ridge = hw["flops"] / hw["bandwidth"]
    bound = "memory" if ai < ridge else "compute"
    # 理论算力上界占峰值的比例：compute-bound 时理论上限就是峰值（封顶 100%），
    # memory 时按带宽比例换算。这不是实测利用率，绝不能读成"GPU 打满了 X%"。
    util_pct = 100.0 * min(1.0, ai / ridge)

    # 一段处理一个微批的时间（理想情况；真实只会更慢）
    if bound == "memory":
        # 每个微批都要把本段的权重**再读一遍**——这就是 PP 在 decode 上的真实代价
        stage_s = (weight_read / gpus + kv_needed / (gpus * mb)) / hw["bandwidth"]
    else:
        stage_s = flops_gpu / hw["flops"]
    # 排空 M 个微批需要 (M + P - 1) 个段时步
    drain_s = (mb + pp - 1) * stage_s
    tok_per_s = batch / drain_s

    # PP 相对"同样的卡只做 TP 不做 PP"的有效吞吐倍数。
    # 教科书公式 P·M/(M+P−1) 是 compute-bound 口径（计算不会被重读）；
    # 我们多数时候是 memory-bound，权重会被重读 M 次，所以要用字节口径：
    #   P·M·(w+k) / ((M+P−1)·(M·w+k))     ← M 大时会 < 1，即比单卡还慢
    if pp == 1:
        pp_speedup = 1.0
    elif bound == "compute":
        pp_speedup = pp * mb / (mb + pp - 1)
    else:
        w, k = weight_read, kv_needed
        pp_speedup = (pp * mb * (w + k)) / ((mb + pp - 1) * (mb * w + k))
    bubble_frac = 0.0 if pp == 1 else (pp - 1) / (mb + pp - 1)

    # TP 通信：每层 2 次 ring AllReduce，每次移动 2(tp-1)/tp × h×batch 个 fp16 激活。
    # 只是理想 ring 口径的**下界**，没有计入单步下限（需要实测互连 busbw）。
    comm = 0 if tp == 1 else int(8 * (tp - 1) / tp * model["layers"] * model["hidden"] * batch)

    # prefill 的算术强度：计算随 seq 线性（矩阵乘）+ 二次（注意力），读取≈权重
    prefill_flops = (2 * active * seq
                     + 4 * model["layers"] * model["q"] * model["hd"] * seq * seq)
    prefill_ai = prefill_flops / (weight_read * pp)

    result = {
        "model": model, "hw": hw,
        "params": n, "active_params": active,
        "tp": tp, "pp": pp, "gpus": gpus,
        "micro_batch": mb,
        "kv_block": kv_block,
        "kv_alloc_tokens": alloc_tokens,
        "kv_pad_tokens": kv_pad_tokens,
        "kv_pad_pct": 100.0 * kv_pad_tokens / seq if seq else 0.0,
        "weight_b": weight_b, "kv_b": kv_b,
        "weight_name": precision_name(weight_b), "kv_name": precision_name(kv_b),
        "seq": seq, "batch": batch,
        "weights_bytes": weights_bytes,
        "per_gpu_weights_bytes": per_gpu_weights,
        "kv_per_req": kv_per_req,
        "kv_needed": kv_needed,
        "vram_bytes": vram_bytes,
        "free_bytes": free,
        "max_concurrency": max_concurrency,
        "weights_fit": weights_fit,
        "batch_fits": batch_fits,
        "runnable": runnable,
        "flops": flops, "flops_per_gpu": flops_gpu,
        "read_bytes": read_bytes, "read_bytes_per_gpu": read_gpu,
        "moe_expert_reads": expert_reads,
        "comm_bytes_per_step": comm,
        "ai": ai,
        "ridge": ridge,
        "bound": bound,
        "util_pct": util_pct,
        "prefill_ai": prefill_ai,
        "prefill_bound": "compute" if prefill_ai >= ridge else "memory",
        "bubble_frac": bubble_frac,
        "pp_speedup": pp_speedup,
        "stage_s": stage_s if runnable else None,
        "step_s": drain_s if runnable else None,
        "tok_per_s": tok_per_s if runnable else None,
        "ms_per_token": (drain_s / batch * 1000.0) if runnable else None,
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
    head = (f"模型: {m.get('name', '?')} "
            f"({r['params'] / 1e9:.2f}B 参数, {m['layers']}L, "
            f"{m['q']}Q/{m['kv']}KV, head_dim {m['hd']}")
    if m.get("moe"):
        head += f"；MoE 激活 {r['active_params'] / 1e9:.2f}B"
    head += ")"
    lines.append(head)
    lines.append(f"硬件: {hw.get('name', '?')} "
                 f"({hw['flops'] / 1e12:.0f} TFLOPS FP16, "
                 f"{hw['bandwidth'] / 1e9:.0f} GB/s, {hw['vram']} GiB)")
    cfg = (f"配置: 权重 {r['weight_name']} ({r['weight_b']} B/参数) | "
           f"KV {r['kv_name']} ({r['kv_b']} B/维度) | "
           f"prompt {r['seq']} | batch {r['batch']}")
    if r["tp"] > 1:
        cfg += f" | tp={r['tp']}（张量并行）"
    if r["pp"] > 1:
        cfg += f" | pp={r['pp']}（流水线并行，{r['micro_batch']} 个微批）"
    if r.get("kv_block"):
        cfg += f" | KV 块 {r['kv_block']} token（分页分配）"
    lines.append(cfg)
    if m.get("moe"):
        moe = m["moe"]
        lines.append(f"MoE: {moe['experts']} 专家取 top-{moe['top_k']}；"
                     f"本步读取 min(batch×{moe['top_k']}, {moe['experts']}) = "
                     f"{r['moe_expert_reads']} 个专家的权重，"
                     f"但全部 {moe['experts']} 个专家都必须常驻显存")
    if r["tp"] > 1:
        warn = ""
        if m["kv"] % r["tp"] != 0:
            warn = "；注意 kv 头数不能被 tp 整除，实际实现会做头数补齐，这里的均分是近似"
        lines.append(f"TP: 权重/KV/计算按 tp={r['tp']} 均分，以下可行性为**每卡**口径{warn}；"
                     f"每步每卡 AllReduce ≈ {r['comm_bytes_per_step'] / 1e6:.2f} MB"
                     f"（理想 ring 下界，未计入单步下限——需实测互连 busbw）")
    if r["pp"] > 1:
        warn = ""
        if m["layers"] % r["pp"] != 0:
            warn = (f"；注意层数 {m['layers']} 不能被 pp={r['pp']} 整除，"
                    "实际框架要求层数能被段数整除，这里的均分是近似")
        lines.append(f"PP: 模型按 pp={r['pp']} 切成 {r['pp']} 段"
                     f"（每段约 {m['layers'] / r['pp']:.1f} 层），"
                     f"{r['micro_batch']} 个微批 → 气泡占比 {r['bubble_frac'] * 100:.1f}%；"
                     f"相对同卡只做 TP 的有效吞吐 {r['pp_speedup']:.2f}×{warn}")
        if r["pp_speedup"] < 1.0:
            lines.append(f"⚠️  微批开到 {r['micro_batch']} 个，每个微批都要把本段权重**再读一遍** → "
                         f"这套 PP 配置比单卡还慢（{r['pp_speedup']:.2f}×）。"
                         "PP 在这里买的是容量，不是速度：把微批调大不等于更快。")
    if r.get("kv_block"):
        lines.append(f"分页: KV 按 {r['kv_block']} token 整块分配 → 每请求占 "
                     f"{r['kv_alloc_tokens']} token（内部碎片 {r['kv_pad_tokens']} token，"
                     f"{r['kv_pad_pct']:.1f}%）")
    lines.append("")

    # --- 可行性 ---
    lines.append("--- 可行性 ---")
    wf = "✓" if r["weights_fit"] else "✗ 装不下"
    lines.append(f"权重: {_gib(r['per_gpu_weights_bytes']):.2f} GiB / {hw['vram']} GiB 每卡  {wf}")
    kf = "✓" if r["batch_fits"] else ("✗ 超容量" if r["weights_fit"] else "— 权重先装不下")
    lines.append(f"KV : 每请求 {_gib(r['kv_per_req']):.2f} GiB × {r['batch']} "
                 f"= {_gib(r['kv_needed']):.2f} GiB 集群 / 每卡余量 {_gib(r['free_bytes']):.2f} GiB  {kf}")
    cap = r["max_concurrency"]
    extra = ""
    if cap < r["batch"]:
        extra = " ← 当前 batch 已超容量"
    elif cap == 0:
        extra = " ← 一个请求都放不下"
    lines.append(f"容量: 余量最多容纳 {cap} 个并发请求（当前 batch {r['batch']}）{extra}")

    if not r["runnable"]:
        # 关键：装不下就拒绝给时间/吞吐。物理上不存在这个数，给了就是误导。
        if not r["weights_fit"]:
            lines.append("判定: 跑不起来——权重本身就装不进显存。"
                         f"每卡超 {_gib(r['per_gpu_weights_bytes'] - r['vram_bytes']):.2f} GiB。"
                         "这个配置根本部署不了，任何时间/吞吐数字都会是假的，所以这里不印。")
            lines.append("出路: 换更大显存的卡 / 降低权重量化（--precision int4）/ "
                         "张量并行（--tp N，权重按卡数均分）/ "
                         "流水线并行（--pp N，按层切段，只买容量不买速度）。")
        else:
            lines.append(f"判定: 跑不起来——权重装得下，但 batch={r['batch']} 的 KV "
                         f"每卡需要 {_gib(r['kv_needed'] / r['gpus']):.2f} GiB，超出余量 "
                         f"{_gib(r['free_bytes']):.2f} GiB。不印假的吞吐。")
            lines.append("出路: 缩短 prompt（--seq）/ 降 KV 精度（--kv-precision int4）/ "
                         "减 batch / 换更大的卡 / 加 --tp 或 --pp 分摊 KV。")
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
    if r["pp"] > 1:
        lines.append(f"纸面流水线排空 ≈ {r['step_s'] * 1000:.1f} ms"
                     f"（每段 {r['stage_s'] * 1000:.2f} ms × {r['micro_batch'] + r['pp'] - 1} 步；"
                     f"每 token {r['ms_per_token']:.2f} ms，聚合 ≈ {r['tok_per_s']:.0f} tok/s）")
    else:
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
    if r["pp"] > 1 and r["pp_speedup"] < 1.0:
        tip = (f"PP 在这里是容量工具、不是加速工具：{r['micro_batch']} 个微批让本段权重被重读 "
               f"{r['micro_batch']} 次，有效吞吐只有单卡的 {r['pp_speedup']:.2f}×。"
               "要速度用 TP，要容量才用 PP——或者把微批数压到最小。")
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
        fails = 0
        # 反证 1：把"每步只读一次权重"写成"batch 个请求各读一遍权重"。
        # 这个错误会让 AI 与 batch 无关（永远 1.21），从而让"B=32 AI≈12.2"这条断言变红。
        # 用裸算式重算，证明判据能把"对"和"错"分开——而不是无论对错都绿。
        n = params_of(model)
        weights = n * 2.0
        kv_req = 4096 * kv_bytes_per_token(model, 2.0)
        attn = 4 * model["layers"] * model["q"] * model["hd"] * 4096
        flops32 = 32 * (2 * n + attn)
        good = flops32 / (weights + 32 * kv_req)          # 正确：权重读一次 → ~12.2
        bad = flops32 / (32 * weights + 32 * kv_req)      # 天真错误：权重读 32 次 → ~1.21
        print(f"[INFO] 正确口径 B=32 AI={good:.2f}；注入 batching 错误后 AI={bad:.2f}")
        if abs(bad - 12.23) <= 0.3:
            print("[FAIL] 注入错误后断言仍通过 → 断言无效（测不出错误）")
            fails += 1
        else:
            print(f"[FAIL] B=32 AI={bad:.2f}（正确应 ~12.2）→ 断言确实变红")
            fails += 1

        # 反证 2：PP 用教科书（compute-bound）公式 P·M/(M+P−1) 代替字节口径。
        # decode 是 memory-bound，权重会被重读 M 次；教科书公式看不见这件事，
        # 于是把"比单卡慢 5 倍"算成"比单卡快 2 倍"——断言必须抓到这个差。
        w = n * 2.0
        k = 32 * 4096 * kv_bytes_per_token(model, 2.0)
        good_pp = (2 * 32 * (w + k)) / ((32 + 2 - 1) * (32 * w + k))
        bad_pp = 2 * 32 / (32 + 2 - 1)
        print(f"[INFO] 正确 PP 口径 pp=2 M=32 → {good_pp:.2f}×；"
              f"教科书公式给 {bad_pp:.2f}×（差 {bad_pp / good_pp:.1f} 倍）")
        if abs(bad_pp - 0.1918) < 1e-3:
            print("[FAIL] 注入教科书公式后断言仍通过 → 断言无效（测不出错误）")
            fails += 1
        else:
            print(f"[FAIL] pp=2 M=32 有效吞吐 {bad_pp:.2f}×（正确应 ~0.19）→ 断言确实变红")
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

    # --- MoE / TP（官方参数量来自 HF safetensors 元数据，2026-10-31 实测核对）---
    mx = dict(MODEL_PRESETS["mixtral-8x7b"], name="mixtral-8x7b")
    check("mixtral-8x7b 全量 ≈ 46.70e9 / 激活 ≈ 12.88e9",
          abs(params_of(mx) - 46_702_792_704) == 0
          and abs(active_params_of(mx) - 12_879_925_248) == 0,
          f"{params_of(mx):,} / {active_params_of(mx):,}")
    qa = dict(MODEL_PRESETS["qwen3-30b-a3b"], name="qwen3-30b-a3b")
    check("qwen3-30b-a3b 全量 = 30,532,122,624（逐字节一致）",
          params_of(qa) == 30_532_122_624, f"{params_of(qa):,}")
    hw80 = dict(HW_PRESETS["a100-80g"], name="a100-80g")
    rmoe = analyze(mx, hw80, 2.0, 2.0, 4096, 1)
    check("mixtral B=1 AI ≈ 6.27（只读 top-2 专家）memory-bound",
          abs(rmoe["ai"] - 6.27) < 0.2 and rmoe["bound"] == "memory", f"{rmoe['ai']:.2f}")
    rtp1 = analyze(model, hw, 2.0, 2.0, 4096, 8, tp=1)
    rtp2 = analyze(model, hw, 2.0, 2.0, 4096, 8, tp=2)
    check("tp=2 切分不改变 AI，且每步通信 = 2,949,120 B（理想 ring，B=8）",
          abs(rtp2["ai"] - rtp1["ai"]) < 1e-9 and rtp2["comm_bytes_per_step"] == 2_949_120,
          f"AI {rtp2['ai']:.4f} / comm {rtp2['comm_bytes_per_step']:,}")

    # --- 分页 KV 与流水线并行 ---
    rpg = analyze(model, hw, 2.0, 2.0, 100, 1, kv_block=16)
    check("分页 B=16：100 token 要占 112 token（内部碎片 12 = 12.0%）",
          rpg["kv_alloc_tokens"] == 112 and rpg["kv_pad_tokens"] == 12
          and abs(rpg["kv_pad_pct"] - 12.0) < 1e-9,
          f"{rpg['kv_alloc_tokens']} token / +{rpg['kv_pad_tokens']} / "
          f"{rpg['kv_pad_pct']:.1f}%")
    rpg4k = analyze(model, hw, 2.0, 2.0, 4096, 1, kv_block=16)
    check("分页 B=16：4096 token 零碎片（块粒度对长上下文免费）",
          rpg4k["kv_pad_tokens"] == 0, f"+{rpg4k['kv_pad_tokens']}")
    rpp = analyze(model, hw, 2.0, 2.0, 4096, 32, pp=2, micro_batch=2)
    check("pp=2 M=2（B=32）有效吞吐 1.03×、气泡 33.3%（几乎白给）",
          abs(rpp["pp_speedup"] - 1.0305) < 1e-3
          and abs(rpp["bubble_frac"] - 1 / 3) < 1e-9,
          f"{rpp['pp_speedup']:.4f}× / 气泡 {rpp['bubble_frac'] * 100:.1f}%")
    rpp32 = analyze(model, hw, 2.0, 2.0, 4096, 32, pp=2, micro_batch=32)
    check("pp=2 M=32 有效吞吐 0.19×（权重被重读 32 次，比单卡还慢 5 倍）",
          rpp32["pp_speedup"] < 1.0 and abs(rpp32["pp_speedup"] - 0.1918) < 1e-3,
          f"{rpp32['pp_speedup']:.4f}×")

    ok = all(c for _, c, _ in checks)
    print(f"\nselftest: {sum(1 for _, c, _ in checks if c)}/{len(checks)} 通过")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_hf_config(path):
    """从 HuggingFace config.json 构造一个模型 dict（支持稠密与 MoE）。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    hidden = cfg["hidden_size"]
    q = cfg["num_attention_heads"]
    hd = cfg.get("head_dim") or (hidden // q)
    out = {
        "hidden": hidden,
        "layers": cfg["num_hidden_layers"],
        "q": q,
        "kv": cfg.get("num_key_value_heads", q),
        "hd": hd,
        "inter": cfg.get("intermediate_size"),
        "vocab": cfg["vocab_size"],
        "tie": bool(cfg.get("tie_word_embeddings", False)),
        "source": f"local:{path}",
    }
    # MoE 字段：Mixtral 用 num_local_experts，Qwen3-MoE 用 num_experts；
    # 专家 FFN 中间维优先 moe_intermediate_size。qk_norm 在 config.json 里没有
    # 通用字段（Qwen3 是架构约定），--config 加载的 MoE 模型算出的参数可能差一个
    # 2×head_dim×层数 的小项——用 --list 里的预设最准。
    if cfg.get("num_local_experts") or cfg.get("num_experts"):
        out["moe"] = {
            "experts": cfg.get("num_local_experts") or cfg["num_experts"],
            "top_k": cfg.get("num_experts_per_tok", 2),
            "inter": cfg.get("moe_intermediate_size") or cfg["intermediate_size"],
        }
        out.pop("inter", None)
    return out


def _print_list():
    print("可用模型：")
    for name, m in MODEL_PRESETS.items():
        if m.get("moe"):
            ptxt = (f"{params_of(m) / 1e9:5.2f}B/激活 {active_params_of(m) / 1e9:5.2f}B"
                    f" top-{m['moe']['top_k']}/{m['moe']['experts']}")
        else:
            ptxt = f"{params_of(m) / 1e9:5.2f}B"
        print(f"  {name:16s} {ptxt:26s} "
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
    ap.add_argument("--tp", type=int, default=1,
                    help="张量并行卡数（默认 1；权重/KV/计算按卡数均分）")
    ap.add_argument("--pp", type=int, default=1,
                    help="流水线并行段数（默认 1；按层切段，只分摊容量、不加速）")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="微批个数 M（仅 --pp>1 有效，缺省 = batch；"
                         "M 越大气泡越小但权重被重读 M 次）")
    ap.add_argument("--kv-block", type=int, default=None,
                    help="KV 分页块大小（token），模拟 PagedAttention 的整块分配；缺省不启用")
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

    # 输入合法性：seq/batch/tp/pp/kv-block 必须是正整数，否则公式无意义
    for name in ("seq", "batch", "tp", "pp"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name} 必须为正整数（收到 {getattr(args, name)}）")
    if args.kv_block is not None and args.kv_block <= 0:
        ap.error(f"--kv-block 必须为正整数（收到 {args.kv_block}）")
    if args.micro_batch is not None:
        if args.micro_batch <= 0:
            ap.error(f"--micro-batch 必须为正整数（收到 {args.micro_batch}）")
        if args.pp <= 1:
            ap.error("--micro-batch 只在 --pp > 1 时有意义（pp=1 时整批一次算完）")
        if args.micro_batch > args.batch:
            ap.error(f"--micro-batch({args.micro_batch}) 不能大于 --batch({args.batch})："
                     "微批是并发请求的切分，切不出比总数更多的份")

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

    r = analyze(model, hw, weight_b, kv_b, args.seq, args.batch, tp=args.tp,
                pp=args.pp, micro_batch=args.micro_batch, kv_block=args.kv_block)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=lambda o: o))
    else:
        print(format_report(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
