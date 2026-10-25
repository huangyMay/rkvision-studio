#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 error_analysis.txt 生成量化精度可视化图（漏斗对比 + 逐层余弦曲线）。"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402



src = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/hmorey/android/rk/04-rkvision-studio/projects/helmet_gold/assets/acc/error_analysis.txt"
dst = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/hmorey/android/rk/04-rkvision-studio/projects/helmet_gold/assets/acc/精度曲线.png"

num = re.compile(r"-?\d+\.\d+")
rows = []
for line in open(src, encoding="utf-8", errors="replace"):
    line = line.rstrip()
    if not line or line.startswith("#") or set(line) <= {"-", " "}:
        continue
    if "layer_name" in line or ("cos" in line[:40] and "euc" in line[:40]):
        continue
    v = num.findall(line)
    if len(v) >= 3:
        rows.append({"name": line.split(v[0])[0].strip(),
                     "s_entire": float(v[0]), "s_single": float(v[2]),
                     "rt_entire": float(v[4]) if len(v) >= 7 else None})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("helmet model — accuracy_analysis quantization report", fontsize=14)

# 左图：四级漏斗（端到端 IoU/余弦对比）
stages = ["ONNX\n(float baseline)", "RKNN\nFP16 sim", "RKNN\nint8 sim", "Board\nint8"]
vals = [1.0, 0.9995, 0.9933, 0.9851]      # ④ 板上实测 2026-10-24
colors = ["#3ecf72", "#4ff0ff", "#ffd24a", "#888888"]
bars = ax1.bar(stages, [v if v else 0 for v in vals],
               color=colors, alpha=0.85, width=0.55)
ax1.set_ylim(0.985, 1.001)
ax1.set_ylabel("end-to-end box-match mean IoU")
ax1.set_title("4-level accuracy funnel (box IoU matching)")
for b, v in zip(bars, vals):
    if v:
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.0002, f"{v:.4f}",
                 ha="center", fontsize=10)
    else:
        ax1.text(b.get_x() + b.get_width() / 2, 0.986, "board offline",
                 ha="center", fontsize=9, color="#888888")
ax1.grid(axis="y", alpha=0.25)

# 右图：逐层余弦曲线
idx = list(range(len(rows)))
ax2.plot(idx, [r["s_entire"] for r in rows], label="sim entire (accumulated)",
         color="#ff6b6b", linewidth=1.6)
ax2.plot(idx, [r["s_single"] for r in rows], label="sim single (per-layer)",
         color="#4ff0ff", linewidth=1.2)
rt = [(i, r["rt_entire"]) for i, r in enumerate(rows) if r["rt_entire"] is not None]
if rt:
    ax2.plot([i for i, _ in rt], [c for _, c in rt], label="runtime entire (board, accumulated)",
             color="#ffd24a", linewidth=1.2, linestyle="--")
ax2.axhline(0.99, color="#888888", linewidth=0.8, linestyle=":", label="0.99 acceptance line")
ax2.set_xlabel("layer index")
ax2.set_ylabel("cosine similarity")
ax2.set_ylim(0.99, 1.001)
ax2.set_title(f"per-layer cosine ({len(rows)} layers)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.25)

# 标注最低点
lo = min(rows, key=lambda r: r["s_entire"])
ax2.annotate(f"min {lo['s_entire']:.4f}\n{lo['name'][:20]}",
             xy=(rows.index(lo), lo["s_entire"]),
             xytext=(rows.index(lo) + 5, 0.9925), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#ff6b6b"))

plt.tight_layout()
plt.savefig(dst, dpi=130)
print("saved:", dst)
