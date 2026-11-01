# roofline-lint

**A zero-dependency CLI that tells you whether an LLM deployment actually fits — and refuses to print a throughput number when it doesn't.**

[![CI](https://github.com/aidless/roofline-lint/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/roofline-lint/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![deps](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](#)

---

## Why another roofline tool

Good roofline and KV-cache calculators already exist (see [comparison](#vs-existing-tools)), and the
strong ones are not naive about feasibility: `LLM-Viewer` reports peak memory next to inference
time, and `pochenai/llm-inference-calculator` reports OOM and back-solves the maximum batch size
when a config does not fit. Feasibility checking is *not* a gap in this ecosystem, and this tool
does not claim to have invented it.

The gap I could not find filled is narrower, and it is about trusting the tool rather than adding
features: **what happens when the analysis itself is wrong?**

So `roofline-lint` is built around two rules.

> **1. If the config cannot run, the only honest answer is "it can't run" — not a fantasy number.**

That is a design choice, not a novelty claim: back-solving a maximum batch (what `pochenai` does)
and refusing outright (what this tool does) are both defensible; silently printing tokens/s for an
impossible config is not.

> **2. A test that cannot fail proves nothing — so the tool must be able to show its own tests failing.**

`--prove-red` injects two known-wrong formulas (weights read once per request; the compute-bound
pipeline formula applied to a memory-bound decode) and asserts the suite turns red. That is the
one thing I could not find in any of the roofline tools surveyed on 2026-11-01.

It answers three questions, in this order:

1. **Does it fit?** weights ≤ VRAM, and `batch × KV_per_request` ≤ free VRAM.
2. **If it fits, where is the bottleneck?** memory-bandwidth vs compute, via roofline arithmetic intensity.
3. **If it's bandwidth-bound, does raising the batch help?** (yes at batch=1 — weights dominate the read; the returns taper as KV starts to dominate).

Plus four things worth checking for in any calculator:

- **Weight precision and KV precision are independent** (`--precision int4 --kv-precision fp16`). "Quantize the weights to fit" and "quantize the KV cache to fit more sequences" are different levers with different costs — the tool keeps them apart, and shows that int4 weights + fp16 KV can *cut your max concurrent requests by 4×*.
- **Tensor parallel *and* pipeline parallel, and they are not interchangeable** (`--tp`, `--pp`). TP splits weights/KV/FLOPs and raises throughput; PP splits layers and buys capacity. On the same two A100s, llama-3-70B does **215 tok/s with `--tp 2`** and **25 tok/s with `--pp 2 --micro-batch 8`** — because every micro-batch re-reads the stage's weights. `pochenai`'s model explicitly assumes `bubble ≈ 0` and so cannot produce this; the report says "slower than one card" out loud instead of printing a happy number.
- **Paged KV shows its own fragmentation** (`--kv-block`). A 100-token request with 16-token blocks really occupies 112 tokens — 12% wasted, and it's exactly the short-request traffic where paging is supposed to pay off.
- **The self-test can go red** (rule 2 above). Pin 17 hand-computed working points, then prove the pins are wired to a real mechanism rather than to themselves.

---

## The math

Everything is a closed-form estimate of the **decode** phase (generating one token). Let
`N` = params, `L` = layers, `q` = query heads, `h_kv` = KV heads, `d` = head_dim,
`S` = context length, `b_w` = weight bytes/param, `b_kv` = KV bytes/dim.

```
FLOPs / token     ≈ 2·N_act  +  4·L·q·d·S        # weight matmuls + attention (QKᵀ, AV)
KV bytes / token  = 2·L·h_kv·d·b_kv              # one K and one V per layer, per KV head
read bytes / step = weights-read + batch·S·(KV/token)  # weights read ONCE per step
arithmetic inten. = FLOPs / read_bytes
ridge point       = peak FLOPS / peak bandwidth
AI < ridge  → memory-bound ;  AI ≥ ridge → compute-bound
step time (floor) = read_bytes / bandwidth   (memory)  |  FLOPs / peak_flops  (compute)
```

- **Dense**: `N_act = N`, `weights-read = N·b_w`.
- **MoE** (`--model mixtral-8x7b`, `--model qwen3-30b-a3b`): `N_act = shared + top_k·expert`,
  and `weights-read = shared + min(batch·top_k, E)·expert` — a small batch touches only the
  activated experts; a batch ≥ E/top_k sweeps them all. **Capacity still uses the full N**:
  activation saves FLOPs and bandwidth, it does not save VRAM.
- **Tensor parallel** (`--tp N`): weights, KV, and FLOPs are split N ways per GPU, so the
  arithmetic intensity is **unchanged** and the per-GPU step time drops ≈ N×. The report also
  prints the ideal-ring AllReduce lower bound `8·(N_tp−1)/N_tp · L · hidden · batch`
  bytes/step/GPU — explicitly *not* folded into the step-time floor, since that needs a
  measured bus bandwidth.

The "weights read **once** per step" is the whole reason batching works: matmul FLOPs scale
with the batch, but the weight read does not, so AI rises with batch until the KV read takes
over. (A tool that writes `read = batch × weights` will report an AI that never moves — that's
exactly the bug `--prove-red` is built to catch.)

Prefill is reported separately: it is compute-bound and its FLOPs grow with `S²` for attention,
so its arithmetic intensity is orders of magnitude higher. The prefill number here is a
*weight-dominated approximation* — it ignores activations and kernel efficiency, so read it as
"which regime", not "exact milliseconds".

---

## Quick start

```bash
git clone https://github.com/aidless/roofline-lint.git
cd roofline-lint
python3 roofline_lint.py            # defaults: qwen3-4b @ a100-40g, fp16, seq=4096, batch=1
```

No install, no dependencies — a single standard-library Python file.

### It fits, and decode is memory-bound

```text
$ python3 roofline_lint.py --model qwen3-4b --gpu a100-40g --seq 4096 --batch 1
模型: qwen3-4b (4.02B 参数, 36L, 32Q/8KV, head_dim 128)
硬件: a100-40g (312 TFLOPS FP16, 1555 GB/s, 40 GiB)
配置: 权重 fp16 (2.0 B/参数) | KV fp16 (2.0 B/维度) | prompt 4096 | batch 1

--- 可行性 ---
权重: 7.49 GiB / 40 GiB  ✓
KV : 每请求 0.56 GiB × 1 = 0.56 GiB / 余量 32.51 GiB  ✓
判定: 跑得起来

--- 瓶颈（roofline）---
算术强度 AI = 1.21 FLOP/byte
ridge 拐点 = 200.6 FLOP/byte
→ MEMORY-BOUND（受显存带宽限制，算力有余而带宽不足）
roofline 理论算力上界 / 峰值 ≈ 0.60%（这是理论上界占峰值的比例，不是实测 GPU 利用率）
纸面单步下限 ≈ 5.6 ms（每 token 5.56 ms，聚合 ≈ 180 tok/s）
prefill 算术强度 ≈ 5326（COMPUTE-BOUND，权重主导近似）

[一句话] batch=1 时每步都要把全部权重从显存搬一遍，权重字节主导了 AI；把 batch 提上去让权重被多个 token 摊薄，AI 才会往上走。
```

### It does not fit — so no speed is printed

```text
$ python3 roofline_lint.py --gpu rtx-3060m-6g --seq 4096 --batch 1
...
权重: 7.49 GiB / 6 GiB  ✗ 装不下
判定: 跑不起来——权重本身就装不进显存。超 1.49 GiB。这个配置根本部署不了，任何时间/吞吐数字都会是假的，所以这里不印。
出路: 换更大显存的卡 / 降低权重量化（--precision int4）/ 张量并行（--tp N，权重按卡数均分）。
```

### The weights fit but the batch's KV doesn't

```text
$ python3 roofline_lint.py --gpu a100-40g --seq 4096 --batch 128
...
KV : 每请求 0.56 GiB × 128 = 72.00 GiB / 余量 32.51 GiB  ✗ 超容量
判定: 跑不起来——权重装得下，但 batch=128 的 KV 需要 72.00 GiB，超出余量 32.51 GiB。不印假的吞吐。
```

### The int4-weights / fp16-KV trap

```bash
# int4 weights + fp16 KV: fits on 6 GB, but only 7 concurrent requests
python3 roofline_lint.py --gpu rtx-3060m-6g --precision int4 --kv-precision fp16 --batch 7

# int4 weights + int4 KV: 29 concurrent requests — the KV precision, not the weights, is what caps you
python3 roofline_lint.py --gpu rtx-3060m-6g --precision int4 --kv-precision int4 --batch 29
```

### MoE and tensor parallel

```bash
# Mixtral 8x7B fp16 = 86.99 GiB of weights: single A100-80G cannot hold it — refused, no numbers
$ python3 roofline_lint.py --model mixtral-8x7b --gpu a100-80g
MoE: 8 专家取 top-2；本步读取 min(batch×2, 8) = 2 个专家的权重，但全部 8 个专家都必须常驻显存
权重: 86.99 GiB / 80 GiB 每卡  ✗ 装不下
判定: 跑不起来——权重本身就装不进显存。每卡超 6.99 GiB。…所以这里不印。

# tp=2: weights/KV/FLOPs split per GPU → it fits, and you get the per-GPU floor + ring-AllReduce lower bound
$ python3 roofline_lint.py --model mixtral-8x7b --gpu a100-80g --tp 2
→ MEMORY-BOUND … roofline 理论算力上界 / 峰值 ≈ 4.10%
纸面单步下限 ≈ 1.1 ms（每 token 1.09 ms，聚合 ≈ 916 tok/s）

# A3B MoE: 30.5B total but only 3.35B active — cheap FLOPs, full 56.87 GiB still resident
$ python3 roofline_lint.py --model qwen3-30b-a3b --gpu a100-80g
模型: qwen3-30b-a3b (30.53B 参数 …；MoE 激活 3.35B)
权重: 56.87 GiB / 80 GiB 每卡  ✓
```

### Paged KV: block-level allocation is not free for short requests

PagedAttention allocates KV in fixed-size blocks, so a request shorter than a block still pays
for the whole block. `--kv-block` makes that padding visible:

```bash
# 100-token request with 16-token blocks: you pay for 112 tokens, 12.0% wasted
$ python3 roofline_lint.py --gpu a100-40g --seq 100 --batch 1 --kv-block 16
配置: … prompt 100 | batch 1 | KV 块 16 token（分页分配）
分页: KV 按 16 token 整块分配 → 每请求占 112 token（内部碎片 12 token，12.0%）

# 4096 tokens with the same 16-token blocks: zero padding — block granularity is free
$ python3 roofline_lint.py --gpu a100-40g --seq 4096 --batch 1 --kv-block 16
分页: KV 按 16 token 整块分配 → 每请求占 4096 token（内部碎片 0 token，0.0%）
```

The waste is a *fraction of one block per request*, so it only matters when requests are short.
That is precisely the traffic shape where paging is supposed to help (many small chats), which
is worth knowing before you attribute a capacity win to the scheduler alone.

### Pipeline parallelism buys capacity, not speed

This is the section worth reading twice. `--pp N` splits the model into N layer stages. Same
GPU count, same model: compare pipeline parallel against tensor parallel on 2×A100-80G.

```bash
# tp=2: shard weights/KV/FLOPs across 2 GPUs
$ python3 roofline_lint.py --model llama-3-70b --gpu a100-80g --batch 8 --tp 2
纸面单步下限 ≈ 37.2 ms（每 token 4.65 ms，聚合 ≈ 215 tok/s）

# pp=2 with 8 micro-batches: exactly the same 2 GPUs
$ python3 roofline_lint.py --model llama-3-70b --gpu a100-80g --batch 8 --pp 2 --micro-batch 8
PP: 模型按 pp=2 切成 2 段（每段约 40.0 层），8 个微批 → 气泡占比 11.1%；相对同卡只做 TP 的有效吞吐 0.24×
⚠️  微批开到 8 个，每个微批都要把本段权重**再读一遍** → 这套 PP 配置比单卡还慢（0.24×）。
纸面流水线排空 ≈ 314.4 ms（每段 34.93 ms × 9 步；每 token 39.30 ms，聚合 ≈ 25 tok/s）
```

**215 tok/s vs 25 tok/s on the same hardware — TP is 8.4× faster here.** The reason is not the
bubble (only 11.1%). It is that decode is memory-bound, and every micro-batch re-reads the whole
stage's weights from HBM. Eight micro-batches means eight weight reads per step. Tensor parallel
does the opposite: it divides both the weights *and* the KV bytes, so its memory traffic per GPU
falls.

Shrink the micro-batch count and the penalty shrinks with it — but it never turns into a speedup:

```bash
# M=1: exactly throughput-neutral (1.00×). Each GPU reads its own half of the weights once,
# so the total weight traffic per step equals the single-card case. Bubble 50.0%.
$ python3 roofline_lint.py --model llama-3-70b --gpu a100-80g --batch 32 --pp 2 --micro-batch 1
PP: 模型按 pp=2 切成 2 段（每段约 40.0 层），1 个微批 → 气泡占比 50.0%；相对同卡只做 TP 的有效吞吐 1.00×

# M=2: still below 1.0 for a 70B model — weights dominate the read, so re-reading them hurts
$ python3 roofline_lint.py --model llama-3-70b --gpu a100-80g --batch 32 --pp 2 --micro-batch 2
PP: … 2 个微批 → 气泡占比 33.3%；相对同卡只做 TP 的有效吞吐 0.75×

# The one case where PP is a net win: when KV reads dominate weights (small model, long
# context, big batch). qwen3-4b at B=32: KV is 19.3 GB/step vs 8.0 GB of weights.
$ python3 roofline_lint.py --model qwen3-4b --gpu a100-80g --batch 32 --pp 2 --micro-batch 2
PP: … 气泡占比 33.3%；相对同卡只做 TP 的有效吞吐 1.03×
```

So the rule the tool encodes: **PP for capacity, TP for throughput.** PP with M=1 costs you
nothing in throughput and halves the per-GPU weight footprint — that is the honest pitch. Any
M > 1 re-reads weights and loses throughput on a memory-bound decode, and the report says so
explicitly instead of printing a happy number.

---

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--model NAME` | `qwen3-4b` | model preset (see `--list`) |
| `--config PATH` | — | build the model from a local HuggingFace `config.json` instead |
| `--gpu NAME` | `a100-40g` | hardware preset (see `--list`) |
| `--precision P` | `fp16` | weight precision: `fp32 fp16 bf16 int8 int4` |
| `--kv-precision P` | follows `--precision` | KV-cache precision (independent of weights) |
| `--seq N` | `4096` | context length (prompt + generated tokens) |
| `--batch N` | `1` | concurrent requests |
| `--tp N` | `1` | tensor parallel: split weights/KV/FLOPs across N GPUs (per-GPU view + ideal-ring comm bound) |
| `--pp N` | `1` | pipeline parallel: split the model into N layer stages (capacity, not throughput) |
| `--micro-batch M` | `= --batch` when `--pp>1` | micro-batches per step; only valid with `--pp>1`, and `M ≤ --batch`. More micro-batches = smaller bubble, but the stage's weights get re-read M times |
| `--kv-block N` | off | page the KV cache in N-token blocks (PagedAttention); shows internal fragmentation |
| `--json` | off | machine-readable output |
| `--list` | — | list model & hardware presets, with sources |
| `--selftest` | — | run the built-in assertions |
| `--prove-red` | — | inject two known wrong formulas; prove the assertions go red |

Exit codes: `0` ok, `1` self-test/prove-red failed, `2` bad arguments.

---

## Data provenance

Every preset carries a `source`. Numbers are vendor **dense (no-sparsity) FP16/BF16 tensor**
figures — not measured here, and not `--sparsity` marketing numbers. Please check the source;
don't treat them as guarantees.

**Models** (dims copied from each repo's `config.json`; params derived by `params_of` and
verified **byte-exact** against HF safetensors `total_size` / API `safetensors.total` in the
tests, 2026-10-31):

| preset | params (total / active) | layers | heads | config.json |
|---|---|---|---|---|
| qwen3-4b | 4.02 B (dense) | 36 | 32Q / 8KV | `Qwen/Qwen3-4B` |
| llama-3-8b | 8.03 B (dense) | 32 | 32Q / 8KV | `NousResearch/Meta-Llama-3-8B` |
| llama-3-70b | 70.55 B (dense) | 80 | 64Q / 8KV | `NousResearch/Meta-Llama-3-70B` |
| mistral-7b | 7.24 B (dense) | 32 | 32Q / 8KV | `mistralai/Mistral-7B-v0.1` ⚠️ sliding-window 4096 |
| mixtral-8x7b | 46.70 B / 12.88 B active, top-2 of 8 | 32 | 32Q / 8KV | `mistralai/Mixtral-8x7B-v0.1` |
| qwen3-30b-a3b | 30.53 B / 3.35 B active, top-8 of 128 | 48 | 32Q / 4KV | `Qwen/Qwen3-30B-A3B` |

**Hardware** (peak FLOPs / bandwidth / VRAM):

| preset | FP16 TFLOPS | GB/s | VRAM | source |
|---|---|---|---|---|
| a100-40g | 312 | 1555 | 40 GiB | A100 40GB PCIe Product Brief |
| a100-80g | 312 | 2039 | 80 GiB | A100 80GB datasheet |
| h100-sxm | 989 | 3350 | 80 GiB | H100 datasheet (no-sparsity) |
| h200-sxm | 989 | 4800 | 141 GiB | H200 datasheet |
| l40s | 362 | 864 | 48 GiB | L40S datasheet |
| rtx-4090 | 165.2 | 1008 | 24 GiB | RTX 4090 specs |
| t4 | 65 | 320 | 16 GiB | T4 datasheet |
| rtx-3060m-6g | ~10.9 | 336 | 6 GiB | RTX 3060 Laptop (approximate) |

To add a model, copy a `MODEL_PRESETS` entry (or just use `--config path/to/config.json`).
To add a GPU, copy a `HW_PRESETS` entry and cite your source.

---

## Tests

```bash
python3 -m unittest test_roofline_lint -v   # 43 tests
python3 roofline_lint.py --selftest         # 17 pinned arithmetic points
python3 roofline_lint.py --prove-red        # injects two bugs, asserts the suite catches both
```

`--prove-red` injects two wrong formulas and prints the numbers they *would* produce:

1. **Batching bug** — treating "read the weights once per step" as "each of the batch requests
   reads the weights". AI stays 1.21 instead of rising to 12.2.
2. **Textbook-PP bug** — using the compute-bound pipeline formula `P·M/(M+P−1)` instead of the
   byte-aware one. For `pp=2 M=32` it claims **1.94×** when the real answer is **0.19×** (a 10.1×
   gap, and the wrong sign: it says "faster than one card" when the truth is "5× slower").

If a future change ever made either formula wrong, `--selftest` would fail too — the red proof
shows the assertions are wired to a real mechanism, not a tautology.

---

## Limitations (read these before quoting a number)

- **Analytical lower bound, not a benchmark.** Peak specs × op counts. Real kernels reach a
  fraction of peak; scheduling, launch overhead, and memory-system contention are not modelled.
- **Decode focus.** Prefill is reported only as a regime indicator (weight-dominated approx).
- **Tensor parallelism is an idealized split.** `--tp N` divides weights/KV/FLOPs by N; the
  AllReduce bound printed is the textbook ideal ring and is *not* folded into the step-time
  floor — real bus bandwidth and overlap are out of scope.
- **Pipeline parallelism is a byte-accounting model, not a scheduler.** `--pp N` assumes uniform
  layers, one forward stage per GPU, and a stage reading its share of the weights once per
  micro-batch. It does not model communication between stages (activation/P2P), interleaved
  1F1B schedules, or the fact that real frameworks require the layer count to divide evenly
  (the report warns when it does not). Its speedup ceiling is therefore a *lower* bound on how
  badly naive PP loses, not an upper bound on what a good schedule achieves.
- **Paged KV models fragmentation only.** `--kv-block N` rounds each request up to a whole
  number of blocks and reports the padding. It does not model block tables, copy-on-write
  sharing, prefix caching, or the allocator's external fragmentation.
- **MoE reads are approximated.** The weight-read model uses `min(batch·top_k, E)` experts per
  step with balanced routing; real routers are not balanced, and expert weights cannot be split
  arbitrarily across TP ranks in practice.
- **Dense attention.** Standard GQA MHA math. MLA and sliding-window effective KV are not
  modelled (mistral-7b's window is flagged, but the KV number is a full-context upper bound).
- **TP needs divisibility.** Heads and KV heads must divide evenly across ranks; the report
  flags a warning when the KV head count is not divisible by `--tp`.
- **VRAM = weights + KV only.** Activations, CUDA context, fragmentation, and the prefill
  transient spike are not in the fit check — real serving needs headroom above this floor.

If any of those matter for your decision, measure. This tool exists to tell you what is
*impossible* cheaply, so you only spend GPU hours on what is *possible*.

---

## Vs existing tools

| Tool | Focus | Infeasible configs | Weight/KV precision separated | Can prove its own tests fail? | Deps |
|---|---|---|---|---|---|
| **roofline-lint** (this) | fit + bottleneck lint | **refuses outright, prints no number** | **yes** | **yes — `--prove-red`** | **0** |
| `hahnyuan/LLM-Viewer` (689★) | layer-wise analysis, peak memory + time | reports peak memory, no dedicated gate | no | no test suite | many |
| `feifeibear/LLMRoofline` (126★) | hardware-comparison grid | n/a | no | no test suite | deps |
| `pochenai/llm-inference-calculator` (22★) | full static perf model: 403 models, 110 GPUs, TP/PP/EP/DP, web UI | **yes — reports OOM and back-solves `B_max`** | **yes** | guard tests (contradictory *inputs* throw), but no formula-injection proof | npm |
| `Pluenet-Killian/llm-roofline` | decode tok/s floor + GPU ranking | no (it is a floor) | no | node:test | 0 |
| `maverickanos` simulator | latency/throughput/energy/cost sim | no | partial | yes | numpy/matplotlib |
| `monamishra95` | teaching + interactive calc | no | no | — | JS |
| KV-cache calculators (many) | KV/VRAM sizing | sizing only | yes (KV) | varies | JS/browser |

**How this table was built (2026-11-01):** each row comes from reading that project's README *and*
its repo tree. The "no test suite" cells are literal — `git/trees?recursive=1` returns zero
test/spec files for `LLM-Viewer` and `LLMRoofline` on that date. Star counts are as of that date.

If a row is wrong, please open an issue. I would rather correct a flattering table than keep one:
an earlier version of this table omitted `LLM-Viewer` and `pochenai` entirely and claimed other
tools "still hand you a tokens/s number" for impossible configs, which was simply false.

The niche is narrow and deliberate: **a tiny, honest feasibility gate you can drop in front of
a real benchmark run, with a switch that proves the gate can fail.** It covers dense and MoE
models, tensor and pipeline parallelism, and paged KV — but only at the level of *accounting*,
never scheduling. If you want a rich simulator with 403 models, use `pochenai`'s; if you want a
fast "will this even launch, and where will it hurt, and can I trust the answer", use this.

---

## License

MIT — see [LICENSE](LICENSE).
