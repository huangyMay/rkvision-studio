#!/usr/bin/env python3
# 用引擎真实布局（plane=7a+c, 索引 i*W+j）在三个输出里找最高分框
import json, numpy as np

G = "projects/helmet_gold/assets/gold"  # 金张量目录（相对仓库根运行）
attrs = json.load(open(f"{G}/attrs.json"))
ANCH = [[(10,13),(16,30),(33,23)], [(30,61),(62,45),(59,119)], [(116,90),(156,198),(373,326)]]
STRIDES = [8, 16, 32]
TARGET = (369.0, 188.0, 438.0, 273.0)   # 结果 JSON 里的最高分框（可改）
SC = 640.0 / 1024.0   # letterbox scale（原图 1024x800）
PAD_T = 70

results = []
for o in range(3):
    ghw = {8: 80, 16: 40, 32: 20}[STRIDES[o]]
    zp, sc = attrs["zps"][o], attrs["scales"][o]
    raw = np.frombuffer(open(f"{G}/out{o}.bin", "rb").read(), dtype=np.int8)
    deq = ((raw.astype(np.float32) - zp) * sc)    # 扁平 21*W*W，索引=plane*W*W + i*W + j
    W = ghw
    for a in range(3):
        bw, bh = ANCH[o][a]
        for i in range(ghw):                     # i = y-grid
            for j in range(ghw):                 # j = x-grid
                base = (7 * a) * W * W + i * W + j
                v = [deq[base + c * W * W] for c in range(7)]
                dx = (v[0] * 2 - 0.5 + j) * STRIDES[o]
                dy = (v[1] * 2 - 0.5 + i) * STRIDES[o]
                dw = (v[2] * 2) ** 2 * bw
                dh = (v[3] * 2) ** 2 * bh
                conf = v[4] * max(v[5], v[6])
                if conf < 0.92: continue
                box = ((dx-dw/2)/SC, (dy-dh/2-PAD_T)/SC, (dx+dw/2)/SC, (dy+dh/2-PAD_T)/SC)
                err = sum(abs(p-q) for p, q in zip(box, TARGET))
                results.append((err, o, a, i, j, conf, v, box))

results.sort(key=lambda r: r[0])
err, o, a, i, j, conf, v, box = results[0]
zp, sc = attrs["zps"][o], attrs["scales"][o]
ghw = {8: 80, 16: 40, 32: 20}[STRIDES[o]]
raw = np.frombuffer(open(f"{G}/out{o}.bin", "rb").read(), dtype=np.int8)
base = (7 * a) * ghw * ghw + i * ghw + j
print(f"命中: out{o} (stride {STRIDES[o]}) anchor={a} (w,h={ANCH[o][a]}) grid=(x={j}, y={i}) conf={conf:.4f}")
print(f"四坐标误差 {err:.2f}px, 解码框={tuple(round(x,1) for x in box)} vs {TARGET}")
print(f"\n7 通道原始 int8 与反量化 ((int8)-({zp}))*{sc:.9f}:")
for c in range(7):
    q = raw[base + c * ghw * ghw]
    d = (q - zp) * sc
    names = ["box_x", "box_y", "box_w", "box_h", "objectness", "cls_head", "cls_helmet"]
    print(f"  plane {7*a+c:2d} [{names[c]:10s}]: int8={q:4d} -> {d:.4f}")
