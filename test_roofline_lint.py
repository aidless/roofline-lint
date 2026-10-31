#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""roofline_lint 的单元测试。运行：python3 -m unittest test_roofline_lint -v

测试分三类：
  1) 正确性：参数量/算术强度/瓶颈判定与手算对得上；
  2) 诚实性（本工具的核心卖点）：装不下/超容量时**必须**拒绝给速度；
  3) 数据与 CLI：每个预设都有可溯源 source；非法输入要被拒。
"""

import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr

import roofline_lint as R


def model(name):
    return dict(R.MODEL_PRESETS[name], name=name)


def hw(name):
    return dict(R.HW_PRESETS[name], name=name)


class TestParams(unittest.TestCase):
    """参数量公式必须能复现官方公布的参数（含 norm、tie/untie 差异）。
    期望值全部来自 HF API 的 safetensors.total（2026-10-31 实测核对；
    qwen3-4b 的 4,022,468,096 含 qk_norm 2×128×36=9216，旧口径 4,022,458,880 是漏了它）。"""

    def test_known_param_counts(self):
        expect = {
            "qwen3-4b": 4_022_468_096,
            "llama-3-8b": 8_030_261_248,
            "llama-3-70b": 70_553_706_496,
            "mistral-7b": 7_241_732_096,
        }
        for name, want in expect.items():
            got = R.params_of(model(name))
            self.assertEqual(got, want, f"{name}: got {got:,} want {want:,}")


class TestRoofline(unittest.TestCase):
    def test_b1_is_memory_bound_at_a100(self):
        r = R.analyze(model("qwen3-4b"), hw("a100-40g"), 2.0, 2.0, 4096, 1)
        self.assertAlmostEqual(r["ai"], 1.21, delta=0.05)
        self.assertEqual(r["bound"], "memory")
        self.assertAlmostEqual(r["util_pct"], 0.60, delta=0.05)

    def test_batching_amortizes_weights(self):
        # 权重每步只读一次 → batch 上去 AI 应该上升，而不是不变。
        r1 = R.analyze(model("qwen3-4b"), hw("a100-40g"), 2.0, 2.0, 4096, 1)
        r32 = R.analyze(model("qwen3-4b"), hw("a100-40g"), 2.0, 2.0, 4096, 32)
        self.assertGreater(r32["ai"], r1["ai"] * 5)  # 1.21 → 12.23
        self.assertAlmostEqual(r32["ai"], 12.23, delta=0.3)

    def test_util_capped_at_100(self):
        fake = {"flops": 20e12, "bandwidth": 3000e9, "vram": 1000, "name": "fake"}
        r = R.analyze(model("qwen3-4b"), fake, 0.5, 0.5, 4096, 16)
        self.assertEqual(r["bound"], "compute")
        self.assertLessEqual(r["util_pct"], 100.0)

    def test_prefill_reported(self):
        r = R.analyze(model("qwen3-4b"), hw("a100-40g"), 2.0, 2.0, 4096, 1)
        self.assertGreater(r["prefill_ai"], r["ai"])  # prefill 更偏 compute
        self.assertEqual(r["prefill_bound"], "compute")


class TestPrecisionDecoupling(unittest.TestCase):
    def test_kv_precision_independent_of_weight(self):
        m = model("qwen3-4b")
        a = R.analyze(m, hw("a100-40g"), 0.5, 2.0, 4096, 1)   # int4 权重 + fp16 KV
        b = R.analyze(m, hw("a100-40g"), 0.5, 0.5, 4096, 1)   # int4 + int4
        self.assertGreater(a["kv_per_req"], b["kv_per_req"] * 3)  # fp16 KV 是 int4 的 4 倍
        self.assertLess(a["weights_bytes"], 3e9)                  # int4 权重 ~2 GiB

    def test_kv_bytes_per_token(self):
        self.assertEqual(R.kv_bytes_per_token(model("qwen3-4b"), 2.0), 147456)
        self.assertEqual(R.kv_bytes_per_token(model("qwen3-4b"), 0.5), 147456 / 4)


class TestHonesty(unittest.TestCase):
    """核心卖点：物理上跑不起来的配置，绝不能印时间/吞吐。"""

    def _report(self, r):
        return R.format_report(r)

    def test_weights_dont_fit_no_speed(self):
        r = R.analyze(model("qwen3-4b"), hw("rtx-3060m-6g"), 2.0, 2.0, 4096, 1)
        self.assertFalse(r["runnable"])
        self.assertIsNone(r["step_s"])
        self.assertIsNone(r["tok_per_s"])
        out = self._report(r)
        self.assertIn("装不下", out)
        self.assertNotIn("纸面单步下限", out)   # 不允许出现速度估计
        self.assertNotIn("tok/s", out)

    def test_batch_exceeds_kv_capacity_no_speed(self):
        r = R.analyze(model("qwen3-4b"), hw("a100-40g"), 2.0, 2.0, 4096, 128)
        self.assertFalse(r["runnable"])
        self.assertTrue(r["weights_fit"])
        self.assertFalse(r["batch_fits"])
        out = self._report(r)
        self.assertIn("超容量", out)
        self.assertNotIn("纸面单步下限", out)

    def test_deployable_does_report_speed(self):
        # 守卫：能跑的配置必须给出速度，别误杀。
        r = R.analyze(model("qwen3-4b"), hw("a100-40g"), 2.0, 2.0, 4096, 8)
        self.assertTrue(r["runnable"])
        self.assertIsNotNone(r["step_s"])
        self.assertIn("纸面单步下限", self._report(r))


class TestMoE(unittest.TestCase):
    """MoE：官方参数量来自 HF safetensors 元数据（2026-10-31 实测抓取：
    Mixtral-8x7B total_size 93,405,584,608 B / bf16 → 46,702,792,704 参数；
    Qwen3-30B-A3B API safetensors.total = 30,532,122,624）。"""

    def test_moe_params_match_official(self):
        self.assertAlmostEqual(
            R.params_of(model("mixtral-8x7b")) / 46_702_792_704, 1.0, delta=0.002)
        self.assertAlmostEqual(
            R.active_params_of(model("mixtral-8x7b")) / 12_879_925_248, 1.0, delta=0.002)
        self.assertAlmostEqual(
            R.params_of(model("qwen3-30b-a3b")) / 30_532_122_624, 1.0, delta=0.002)
        self.assertAlmostEqual(
            R.active_params_of(model("qwen3-30b-a3b")) / 3_353_032_704, 1.0, delta=0.002)

    def test_dense_params_unchanged_by_moe_support(self):
        # 守卫：支持 MoE 不能动稠密模型的参数（官方含 qk_norm 的总数，见 TestParams 注）
        self.assertEqual(R.params_of(model("qwen3-4b")), 4_022_468_096)
        self.assertEqual(R.active_params_of(model("qwen3-4b")), 4_022_468_096)

    def test_moe_b1_reads_only_topk_experts(self):
        m = model("mixtral-8x7b")
        r = R.analyze(m, hw("a100-80g"), 2.0, 2.0, 4096, 1)
        shared = R.params_of(m) - R.expert_params_of(m)
        expert_b = 3 * m["hidden"] * m["moe"]["inter"] * 2.0
        want = (shared * 2.0 + 2 * expert_b        # 共享参数 + top-2 专家
                + 4096 * R.kv_bytes_per_token(m, 2.0))
        self.assertAlmostEqual(r["read_bytes"], want, delta=want * 1e-9)
        # 稀疏可见：读取明显小于全部权重
        self.assertLess(r["read_bytes"], 0.2 * R.params_of(m) * 2.0)

    def test_moe_batch_covers_all_experts(self):
        m = model("mixtral-8x7b")
        r = R.analyze(m, hw("a100-80g"), 2.0, 2.0, 1024, 16)   # 16×top2=32 ≥ 8 个专家
        shared = R.params_of(m) - R.expert_params_of(m)
        expert_b = 3 * m["hidden"] * m["moe"]["inter"] * 2.0
        want = (shared * 2.0 + 8 * expert_b                     # 全部 8 个专家都被读到
                + 16 * 1024 * R.kv_bytes_per_token(m, 2.0))
        self.assertAlmostEqual(r["read_bytes"], want, delta=want * 1e-9)

    def test_moe_flops_use_active_params(self):
        m = model("mixtral-8x7b")
        r = R.analyze(m, hw("a100-80g"), 2.0, 2.0, 4096, 1)
        attn = 4 * m["layers"] * m["q"] * m["hd"] * 4096
        want = 2 * R.active_params_of(m) + attn
        self.assertAlmostEqual(r["flops"], want, delta=want * 1e-9)

    def test_moe_capacity_requires_all_experts(self):
        # Mixtral fp16 权重 86.99 GiB：单卡 A100-80 装不下——哪怕每 token 只激活 12.9B。
        # 激活省的是算力和带宽，省不了显存：全部专家都必须常驻。
        r = R.analyze(model("mixtral-8x7b"), hw("a100-80g"), 2.0, 2.0, 4096, 1)
        self.assertFalse(r["weights_fit"])
        self.assertFalse(r["runnable"])
        self.assertIsNone(r["step_s"])
        out = R.format_report(r)
        self.assertIn("装不下", out)
        self.assertNotIn("纸面单步下限", out)

    def test_moe_report_shows_active_and_experts(self):
        r = R.analyze(model("qwen3-30b-a3b"), hw("a100-80g"), 2.0, 2.0, 4096, 1)
        out = R.format_report(r)
        self.assertIn("激活", out)
        self.assertIn("专家", out)


class TestTensorParallel(unittest.TestCase):
    def test_tp_splits_weights_kv_flops_and_keeps_ai(self):
        m = model("qwen3-4b")
        r1 = R.analyze(m, hw("a100-40g"), 2.0, 2.0, 4096, 8, tp=1)
        r2 = R.analyze(m, hw("a100-40g"), 2.0, 2.0, 4096, 8, tp=2)
        # 权重、KV、FLOPs 都被切成两半 → AI 不变（切分不改变算术强度，只改变时间）
        self.assertAlmostEqual(r2["ai"], r1["ai"], delta=1e-9)
        self.assertAlmostEqual(r2["per_gpu_weights_bytes"], r1["weights_bytes"] / 2, delta=1)
        self.assertAlmostEqual(r2["flops_per_gpu"], r1["flops"] / 2, delta=1)
        self.assertAlmostEqual(r2["step_s"], r1["step_s"] / 2, delta=1e-6)

    def test_tp_comm_bytes_reported(self):
        m = model("qwen3-4b")
        r1 = R.analyze(m, hw("a100-40g"), 2.0, 2.0, 4096, 1, tp=1)
        r2 = R.analyze(m, hw("a100-40g"), 2.0, 2.0, 4096, 1, tp=2)
        r4 = R.analyze(m, hw("a100-40g"), 2.0, 2.0, 4096, 1, tp=4)
        # 每层 2 次 ring AllReduce，每次移动 2(tp-1)/tp × h×batch × 2B
        self.assertEqual(r1["comm_bytes_per_step"], 0)
        self.assertEqual(r2["comm_bytes_per_step"], 368_640)
        self.assertGreater(r4["comm_bytes_per_step"], r2["comm_bytes_per_step"])

    def test_tp_makes_model_fit(self):
        # Mixtral fp16 86.99 GiB：tp=1 装不下 → tp=2 每卡 43.5 GiB 装得下，给速度
        m = model("mixtral-8x7b")
        r1 = R.analyze(m, hw("a100-80g"), 2.0, 2.0, 4096, 1, tp=1)
        r2 = R.analyze(m, hw("a100-80g"), 2.0, 2.0, 4096, 1, tp=2)
        self.assertFalse(r1["runnable"])
        self.assertTrue(r2["runnable"])
        out = R.format_report(r2)
        self.assertIn("tp=2", out)
        self.assertIn("纸面单步下限", out)

    def test_cli_accepts_tp(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = R.main(["--model", "mixtral-8x7b", "--gpu", "a100-80g",
                         "--tp", "2", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["tp"], 2)
        self.assertTrue(payload["runnable"])

    def test_cli_tp_must_be_positive(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                R.main(["--tp", "0"])
        self.assertEqual(cm.exception.code, 2)


class TestDataProvenance(unittest.TestCase):
    """每个预设都必须带 source，否则等于不可溯源的黑数。"""

    def test_models_have_source(self):
        for name, m in R.MODEL_PRESETS.items():
            self.assertTrue(m.get("source"), f"模型 {name} 缺 source")

    def test_hardware_have_source(self):
        for name, h in R.HW_PRESETS.items():
            self.assertTrue(h.get("source"), f"硬件 {name} 缺 source")
            self.assertGreater(h["flops"], 0)
            self.assertGreater(h["bandwidth"], 0)
            self.assertGreater(h["vram"], 0)


class TestCLI(unittest.TestCase):
    def test_kv_precision_flag(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = R.main(["--precision", "int4", "--kv-precision", "fp16",
                         "--gpu", "rtx-3060m-6g", "--seq", "4096", "--batch", "1"])
        self.assertEqual(rc, 0)
        self.assertIn("KV fp16", buf.getvalue())

    def test_nonpositive_inputs_rejected(self):
        for bad in ("--seq", "--batch"):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    R.main([bad, "0"])
            self.assertEqual(cm.exception.code, 2)

    def test_json_output_is_valid(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = R.main(["--gpu", "a100-40g", "--batch", "4", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertIn("ai", payload)
        self.assertTrue(payload["runnable"])

    def test_hf_config_loader(self):
        # 真写一个临时 config.json，走 _load_hf_config，验证字段映射与参数复现。
        import os
        import tempfile
        cfg = {"hidden_size": 2560, "num_hidden_layers": 36, "num_attention_heads": 32,
               "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9728,
               "vocab_size": 151936, "tie_word_embeddings": True}
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            m = R._load_hf_config(path)
        finally:
            os.unlink(path)
        self.assertAlmostEqual(R.params_of(m) / 4.02e9, 1.0, delta=0.01)


class TestSelfTestItself(unittest.TestCase):
    """自检和反证本身要能用：selftest 通过；prove-red 必须能变红（返回 True）。"""

    def test_selftest_passes(self):
        with redirect_stdout(io.StringIO()):
            self.assertTrue(R.selftest())

    def test_prove_red_goes_red(self):
        with redirect_stdout(io.StringIO()):
            self.assertTrue(R.selftest(prove_red=True))


if __name__ == "__main__":
    unittest.main()
