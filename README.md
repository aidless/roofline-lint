# roofline-lint

**A zero-dependency CLI that tells you whether an LLM deployment actually fits — and refuses to print a throughput number when it doesn't.**

[![CI](https://github.com/aidless/roofline-lint/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/roofline-lint/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![deps](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](#)

---

## Why another roofline tool

There are already good roofline and KV-cache calculators (see [comparison](#vs-existing-tools)). Almost all of them answer **"how fast could it go?"** and quietly assume the model fits. When it doesn't fit — weights larger than VRAM, or the requested batch's KV cache larger than the free memory — they still hand you a tokens/s number. That number is physically impossible, yet it gets copied straight into a capacity-planning spreadsheet.

`roofline-lint` is built around one rule:

> **If the config cannot run, the only honest answer is "it can't run" — not a fantasy number.**

So it answers three questions, in this order:

1. **Does it fit?** weights ≤ VRAM, and `batch × KV_per_request` ≤ free VRAM.
2. **If it fits, where is the bottleneck?** memory-bandwidth vs compute, via roofline arithmetic intensity.
3. **If it's bandwidth-bound, does raising the batch help?** (yes at batch=1 — weights dominate the read; the returns taper as KV starts to dominate).

And two extras that most one-shot calculators skip:

- **Weight precision and KV precision are independent** (`--precision int4 --kv-precision fp16`). "Quantize the weights to fit" and "quantize the KV cache to fit more sequences" are different levers with different costs — the tool keeps them apart, and shows that int4 weights + fp16 KV can *cut your max concurrent requests by 4×*.
- **The self-test can go red.** `--prove-red` injects a classic bug (reading weights once per request instead of once per step) and asserts the test catches it. A test that cannot fail proves nothing.

---

## The math

Everything is a closed-form estimate of the **decode** phase (generating one token). Let
`N` = params, `L` = layers, `q` = query heads, `h_kv` = KV heads, `d` = head_dim,
`S` = context length, `b_w` = weight bytes/param, `b_kv` = KV bytes/dim.

```
FLOPs / token     ≈ 2·N  +  4·L·q·d·S            # weight matmuls + attention (QKᵀ, AV)
KV bytes / token  = 2·L·h_kv·d·b_kv              # one K and one V per layer, per KV head
read bytes / step = N·b_w  +  batch·S·(KV/token) # weights read ONCE per step + all KV
arithmetic inten. = FLOPs / read_bytes
ridge point       = peak FLOPS / peak bandwidth
AI < ridge  → memory-bound ;  AI ≥ ridge → compute-bound
step time (floor) = read_bytes / bandwidth   (memory)  |  FLOPs / peak_flops  (compute)
```

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
出路: 换更大显存的卡 / 降低权重量化（--precision int4）/ 多卡切分（本工具不建模 TP）。
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
| `--json` | off | machine-readable output |
| `--list` | — | list model & hardware presets, with sources |
| `--selftest` | — | run the built-in assertions |
| `--prove-red` | — | inject the batching bug; prove the assertions go red |

Exit codes: `0` ok, `1` self-test/prove-red failed, `2` bad arguments.

---

## Data provenance

Every preset carries a `source`. Numbers are vendor **dense (no-sparsity) FP16/BF16 tensor**
figures — not measured here, and not `--sparsity` marketing numbers. Please check the source;
don't treat them as guarantees.

**Models** (dims copied from each repo's `config.json`; params derived by `params_of` and
cross-checked against the published counts in the tests):

| preset | params | layers | heads | config.json |
|---|---|---|---|---|
| qwen3-4b | 4.02 B | 36 | 32Q / 8KV | `Qwen/Qwen3-4B` |
| llama-3-8b | 8.03 B | 32 | 32Q / 8KV | `NousResearch/Meta-Llama-3-8B` |
| llama-3-70b | 70.6 B | 80 | 64Q / 8KV | `NousResearch/Meta-Llama-3-70B` |
| mistral-7b | 7.24 B | 32 | 32Q / 8KV | `mistralai/Mistral-7B-v0.1` ⚠️ sliding-window 4096 |

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
python3 -m unittest test_roofline_lint -v   # 18 tests
python3 roofline_lint.py --selftest         # 9 pinned arithmetic points
python3 roofline_lint.py --prove-red        # injects a bug, asserts the suite catches it
```

`--prove-red` prints the number the bug *would* produce (AI stays 1.21 instead of rising to
12.2), then flags the assertion red. If a future change ever made the batching math wrong, the
ordinary `--selftest` would fail too — the red proof shows the assertions are wired to a real
mechanism, not a tautology.

---

## Limitations (read these before quoting a number)

- **Analytical lower bound, not a benchmark.** Peak specs × op counts. Real kernels reach a
  fraction of peak; scheduling, launch overhead, and memory-system contention are not modelled.
- **Decode focus.** Prefill is reported only as a regime indicator (weight-dominated approx).
- **No tensor parallelism.** Multi-GPU weight sharding and collective-communication cost are out
  of scope; the tool will tell you a model doesn't fit rather than pretend TP fixes it for free.
- **Dense attention.** Standard GQA MHA math. MoE routing, MLA, sliding-window effective KV, and
  PagedAttention's block-level effects are not modelled (mistral-7b's window is flagged, but the
  KV number is a full-context upper bound).
- **VRAM = weights + KV only.** Activations, CUDA context, fragmentation, and the prefill
  transient spike are not in the fit check — real serving needs headroom above this floor.

If any of those matter for your decision, measure. This tool exists to tell you what is
*impossible* cheaply, so you only spend GPU hours on what is *possible*.

---

## Vs existing tools

| Tool | Focus | Refuses infeasible configs? | Decouples weight/KV precision? | Provable tests? | Deps |
|---|---|---|---|---|---|
| **roofline-lint** (this) | fit + bottleneck lint | **yes** | **yes** | **yes (`--prove-red`)** | **0** |
| Pluenet-Killian/llm-roofline | decode tok/s floor, GPU ranking | no (it's a floor) | no | node:test | 0 |
| azrabano23/llm-roofline → maverickanos simulator | full latency/throughput/energy/cost sim, TP | no | partial | yes | numpy/matplotlib |
| monamishra95/roofline-model-llm-inference | teaching + interactive calc | no | no | — | JS |
| KV-cache calculators (many) | KV/VRAM sizing | sizing only | yes (KV) | varies | JS/browser |

The niche is narrow and deliberate: **a tiny, honest feasibility gate you can drop in front of
a real benchmark run.** If you want a rich simulator, use one of the others; if you want a fast
"will this even launch, and where will it hurt", use this.

---

## License

MIT — see [LICENSE](LICENSE).
