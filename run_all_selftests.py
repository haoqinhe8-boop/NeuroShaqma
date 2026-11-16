# -*- coding: utf-8 -*-
"""逐个跑所有 selftest（避免 pytest 卡住/内存累积），汇总结果。"""
import os
import re
import subprocess
import sys
import time

ROOT = r"D:\Apps\MyModel\chat\workspace\xiaosha"
PY = r"D:\Apps\MyModel\train\venv\Scripts\python.exe"

t = open(os.path.join(ROOT, "tests", "test_all.py"), encoding="utf-8").read()
mods = re.findall(r'\("([a-z0-9_]+)",\s*\d+', t)
print("共 %d 个模块\n" % len(mods))

passed, failed = [], []
t_all = time.time()
for m in mods:
    t0 = time.time()
    try:
        r = subprocess.run([PY, "-X", "utf8", m + ".py"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=900)
        code = r.returncode
        out = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        code = -1
        out = "TIMEOUT 900s"
    el = time.time() - t0
    res = ""
    for line in out.splitlines():
        if "结果：" in line or "MemoryError" in line or "Traceback" in line:
            res = line.strip()
            break
    ok = (code == 0)
    (passed if ok else failed).append(m)
    print("  %-26s %s  %6.1fs  %s" % (m, "PASS" if ok else "FAIL", el, res))

print("\n汇总: %d/%d 通过 | 总耗时 %.0fs" % (len(passed), len(mods), time.time() - t_all))
if failed:
    print("失败: " + ", ".join(failed))
sys.exit(0 if not failed else 1)
