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
    """参数量公式必须能复现官方公布的参数（含 norm、tie/untie 差异）。"""

    def test_known_param_counts(self):
        expect = {
            "qwen3-4b": 4.022458880e9,
            "llama-3-8b": 8.030261248e9,
            "llama-3-70b": 70.633706496e9,
            "mistral-7b": 7.241732096e9,
        }
        for name, want in expect.items():
            got = R.params_of(model(name))
            self.assertLess(abs(got - want) / want, 0.01,
                            f"{name}: got {got:,} want {want:,.0f}")


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
