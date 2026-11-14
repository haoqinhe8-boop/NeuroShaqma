# -*- coding: utf-8 -*-
"""experiments/ 目录健康检查（纳入 pytest，防止实验脚本腐化）。

为什么需要它：
  `experiments/` 里是 9 个可复现实验脚本，它们不在主 MODULES 里，
  以前只能靠临时 ad-hoc 脚本抽查 —— 一旦某个脚本缺 target_tol / 缺 __main__ 保护 /
  sys.path 没修，要等真跑起来才发现。本模块把这些检查固化下来。

检查项：
  1. 全部脚本语法合法（ast.parse）
  2. 全部可 import 且**无副作用**（import 不该跑实验/打日志）
  3. 可执行脚本都有 __main__ 保护
  4. 失效结论有显式标注（如 exp_multihead.py v1 的 INVALID RESULT）
"""
import ast
import glob
import importlib
import subprocess
import os
import sys

__version__ = "1.0.0"

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, "experiments")


def selftest():
    print("  experiments/ 健康检查（语法 / import / 副作用 / main 保护）")
    ok = 0
    total = 0
    files = sorted(glob.glob(os.path.join(EXP, "*.py")))
    names = [os.path.basename(f) for f in files]

    # 1) 语法
    total += 1
    bad = []
    for f in files:
        try:
            ast.parse(open(f, encoding="utf-8").read())
        except SyntaxError as e:
            bad.append("%s:%s" % (os.path.basename(f), e.lineno))
    print("  [1] 语法检查 %d 个脚本%s" % (len(files), "" if not bad else " —— 失败: " + ", ".join(bad)))
    ok += int(not bad)

    # 2) 可 import 且无副作用（用 stdout 捕获判断）
    total += 1
    sys.path.insert(0, EXP)
    import io
    from contextlib import redirect_stdout
    bad2 = []
    for f in files:
        name = os.path.basename(f)[:-3]
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                if name in sys.modules:
                    del sys.modules[name]
                importlib.import_module(name)
        except Exception as e:
            bad2.append("%s(%s)" % (name, type(e).__name__))
            continue
        if buf.getvalue().strip():
            bad2.append("%s(有顶层输出)" % name)
    print("  [2] import 无副作用 %d 个%s" % (len(files), "" if not bad2 else " —— 失败: " + ", ".join(bad2)))
    ok += int(not bad2)

    # 3) 可执行脚本都有 __main__ 保护
    total += 1
    no_guard = []
    for f in files:
        src = open(f, encoding="utf-8").read()
        has_main_call = "\nmain()" in src
        if has_main_call and '__name__ == "__main__"' not in src:
            no_guard.append(os.path.basename(f))
    print("  [3] __main__ 保护%s" % ("" if not no_guard else " —— 缺失: " + ", ".join(no_guard)))
    ok += int(not no_guard)

    # 4) 失效结论有显式标注
    total += 1
    v1 = os.path.join(EXP, "exp_multihead.py")
    marked = False
    if os.path.exists(v1):
        marked = "INVALID RESULT" in ast.get_docstring(ast.parse(
            open(v1, encoding="utf-8").read())) or ""
    print("  [4] 失效结论已标注 INVALID%s" % ("" if marked else " —— 缺少警告"))
    ok += int(bool(marked))

    # 5) 边界冒烟：带 ROUNDS 环境变量的脚本在极小轮数下不能崩
    #    （教训：exp_structure_vs_random.py 曾在 ROUNDS<EVAL_EVERY 时 IndexError）
    total += 1
    edge_bad = []
    for f in files:
        src = open(f, encoding="utf-8").read()
        if "os.environ.get('ROUNDS'" not in src:
            continue
        env = dict(os.environ)
        env["ROUNDS"] = "30"
        try:
            r = subprocess.run([sys.executable, "-X", "utf8", f],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", cwd=EXP, env=env, timeout=180)
            if r.returncode != 0:
                edge_bad.append("%s(exit=%d)" % (os.path.basename(f), r.returncode))
            elif "Traceback" in (r.stderr or ""):
                edge_bad.append("%s(traceback)" % os.path.basename(f))
        except subprocess.TimeoutExpired:
            edge_bad.append("%s(超时)" % os.path.basename(f))
    print("  [5] 边界冒烟(ROUNDS=30)%s" % ("" if not edge_bad else " —— 失败: " + ", ".join(edge_bad)))
    ok += int(not edge_bad)

    print("  [信息] 共 %d 个实验脚本: %s" % (len(files), ", ".join(names)))
    print("  结果：%d/%d 项通过" % (ok, total))
    return ok == total


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
