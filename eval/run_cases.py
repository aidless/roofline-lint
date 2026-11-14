#!/usr/bin/env python3
"""eval/run_cases.py — 把 eval/cases.jsonl 当作可执行验收题库跑一遍。

设计口径（写在这里，避免下一个人猜）：

- 每条 case 有 `mode`：
  * `expr`   —— `verify` 是一段 Python 代码（可多行），执行后读变量 `result`，与 `expected` 比较；
  * `raises` —— `verify` 会抛异常，断言异常类型名 == `expected`；
  * `doc`    —— `verify` 形如 `相对路径::正则`，断言该正则在文件里出现。
- `expr` 题的期望值必须是**真跑出来的**，不是抄文档：改一行实现就可能让它变红，
  这正是这套题的意义。
- `--selftest` 把第一条 case 的 expected 篡改成不可能的值，必须变红 —— 否则这套题是装饰
  （对应 AGENTS.md 第 8 条 / 第 21 条的 RED→GREEN 两段断言）。
- 退出码：0 全过；1 有失败；2 题库本身有问题（读不了/字段缺/重复 id）。

用法：
    python3 eval/run_cases.py            # 跑全部
    python3 eval/run_cases.py --json     # 机器可读
    python3 eval/run_cases.py --selftest # 证明它会红
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES = REPO_ROOT / "eval" / "cases.jsonl"
REQUIRED = ("id", "axis", "question", "expected", "mode", "verify", "source")

# 裸克隆（未 pip install -e .）也要能跑：把仓库根放到 sys.path 最前。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def die(msg: str) -> "NoReturn":
    """题库自身有问题 → exit 2（与文件头文档一致；exit 1 专留给 case 失败）。"""
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_cases(path: Path) -> list[dict]:
    if not path.is_file():
        die(f"找不到题库：{path}")
    rows, seen = [], set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            die(f"第 {lineno} 行不是合法 JSON：{exc}")
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            die(f"第 {lineno} 行缺字段 {missing}")
        if row["id"] in seen:
            die(f"重复 id: {row['id']}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        die("题库为空")
    return rows


def cmp_value(actual, expected: str) -> tuple[bool, str]:
    if isinstance(actual, bool):
        return str(actual).lower() == str(expected).strip().lower(), repr(actual)
    if isinstance(actual, (int, float)):
        try:
            exp = float(expected)
        except ValueError:
            return False, repr(actual)
        if isinstance(actual, int) and exp.is_integer():
            return actual == int(exp), repr(actual)
        tol = max(1e-9, 1e-9 * abs(exp))
        return math.isclose(float(actual), exp, rel_tol=0.0, abs_tol=tol), repr(actual)
    return str(actual).strip() == str(expected).strip(), str(actual)


def run_case(row: dict) -> tuple[bool, str]:
    mode, verify = row["mode"], row["verify"]
    if mode in ("expr", "raises"):
        ns: dict = {"__name__": "__case__"}
        try:
            exec(compile(verify, f"<{row['id']}>", "exec"), ns)  # noqa: S102 — 题库是我们自己的文件
            actual = ns.get("result")
            if mode == "raises":
                return False, f"未抛异常，result={actual!r}"
        except Exception as exc:  # noqa: BLE001
            if mode == "raises":
                return type(exc).__name__ == row["expected"].strip(), f"raised {type(exc).__name__}"
            return False, f"raised {type(exc).__name__}: {exc}"
        return cmp_value(actual, row["expected"])
    if mode == "doc":
        target, _, pattern = verify.partition("::")
        path = REPO_ROOT / target
        if not path.is_file():
            return False, f"文件不存在 {target}"
        hit = re.search(pattern, path.read_text(encoding="utf-8", errors="ignore"))
        return bool(hit), (hit.group(0)[:40] if hit else "未命中")
    return False, f"未知 mode {mode!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="篡改一条期望值，必须变红")
    ap.add_argument("--save", action="store_true",
                    help="把本次结果写到 eval/run-report.json（默认不写，保持工作树干净）")
    args = ap.parse_args()

    rows = load_cases(CASES)
    if args.selftest:
        rows[0] = dict(rows[0], expected="__selftest_impossible_value__")

    results, passed = [], 0
    for row in rows:
        ok, detail = run_case(row)
        passed += ok
        results.append({"id": row["id"], "axis": row["axis"], "ok": ok, "detail": detail})

    # 失败归因：按 axis 汇总，而不是只报一个 pass_rate。
    failed_by_axis: dict[str, list[str]] = {}
    for r in results:
        if not r["ok"]:
            failed_by_axis.setdefault(r["axis"], []).append(r["id"])

    payload = {"total": len(rows), "passed": passed, "failed": len(rows) - passed,
               "selftest": args.selftest, "failed_by_axis": failed_by_axis,
               "results": results}
    if args.save:
        out = REPO_ROOT / "eval" / "run-report.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[saved] {out.relative_to(REPO_ROOT)}", file=sys.stderr)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        for r in results:
            print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['id']} {r['axis']}: {r['detail']}")
        print(f"\n{passed}/{len(rows)} cases passed")
        if failed_by_axis:
            print("失败归因（按 axis）：")
            for axis, ids in sorted(failed_by_axis.items()):
                print(f"  {axis}: {len(ids)} 条 —— {', '.join(ids)}")
        if args.selftest:
            print("SELFTEST: 期望变红；若上面仍全绿则这套题无效")

    if args.selftest:
        return 0 if passed < len(rows) else 3
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
