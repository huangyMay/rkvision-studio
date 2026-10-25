#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 逐层体检探针：图手术暴露全部中间输出，单次推理收集逐层张量统计。

评估量化友好度——outlier 比（absmax/p99.9）大的层 = int8 量化敏感层候选。
纯 PC 运行：无需板子、无需 toolkit2，任何装了 onnxruntime 的机器都能跑。

用法: onnx_probe.py <model.onnx> <image> [out_dir]
输出 JSON: {"ok":true,"model":..,"input_hw":[h,w],"total_ms":..,"n_layers":N,
            "sensitive":[{name,outlier}..],"report":报告路径}
"""
import json
import os
import sys
import time

import numpy as np
import cv2
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_infer import letterbox  # noqa: E402


def patch_graph_outputs(model):
    """把每个节点的输出都挂成图输出 → 一次 run 拿到全部中间张量。"""
    existing = {o.name for o in model.graph.output}
    for node in model.graph.node:
        for o in node.output:
            if o and o not in existing:
                model.graph.output.append(onnx.helper.make_empty_tensor_value_info(o))
                existing.add(o)
    return model


def probe(onnx_path, image_path):
    """核心体检：图手术暴露全部中间张量 → 单次推理 → 逐层统计。
    返回 (layers, meta)；layers 每项含 name/type/shape/min/max/mean/std/zero_pct/absmax/p999/outlier。"""
    model = onnx.load(onnx_path)
    patch_graph_outputs(model)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])

    inp = sess.get_inputs()[0]
    dims = list(inp.shape)
    ih = dims[2] if isinstance(dims[2], int) else 640
    iw = dims[3] if isinstance(dims[3], int) else 640

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"图片解码失败: {image_path}")
    lb, _, _, _ = letterbox(img, max(ih, iw))
    blob = cv2.dnn.blobFromImage(lb, 1.0 / 255.0, swapRB=True)

    names = [o.name for o in sess.get_outputs()]
    t0 = time.time()
    results = sess.run(None, {inp.name: blob})
    total_ms = (time.time() - t0) * 1000

    ntype = {}
    for n in model.graph.node:
        for o in n.output:
            ntype[o] = n.op_type

    layers = []
    for name, t in zip(names, results):
        a = np.asarray(t)
        af = a.astype(np.float32).ravel()
        if af.size == 0:
            continue
        absmax = float(np.abs(af).max())
        p999 = float(np.percentile(np.abs(af), 99.9))
        outlier = round(absmax / p999, 1) if p999 > 1e-9 else None
        layers.append({
            "name": name, "type": ntype.get(name, "?"),
            "shape": list(a.shape),
            "min": round(float(af.min()), 5), "max": round(float(af.max()), 5),
            "mean": round(float(af.mean()), 5), "std": round(float(af.std()), 5),
            "zero_pct": round(float((af == 0).mean() * 100), 2),
            "absmax": round(absmax, 5), "p999": round(p999, 5),
            "outlier": outlier,
        })
    return layers, {"ih": ih, "iw": iw, "total_ms": round(total_ms, 1)}


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"ok": False,
                          "error": "usage: onnx_probe.py <model.onnx> <image> [out_dir]"}))
        return
    onnx_path = os.path.abspath(sys.argv[1])
    image_path = os.path.abspath(sys.argv[2])
    out_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else "/tmp/onnx_probe_out"
    os.makedirs(out_dir, exist_ok=True)

    for p, what in ((onnx_path, "模型"), (image_path, "图片")):
        if not os.path.exists(p):
            print(json.dumps({"ok": False, "error": f"{what}不存在: {p}"}))
            return

    try:
        layers, meta = probe(onnx_path, image_path)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(e)[:200]}))
        return
    ih, iw, total_ms = meta["ih"], meta["iw"], meta["total_ms"]

    # 量化敏感层：outlier 比（absmax / p99.9）越大，int8 按 max 定 scale 时
    # 其余值的有效分辨率被压得越狠——这是转换前就能看到的"量化伤"信号
    sens = [l for l in layers if l.get("outlier") and l["outlier"] > 10]
    sens.sort(key=lambda l: -l["outlier"])

    report = os.path.join(out_dir, "onnx_probe_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"# ONNX 逐层体检报告\n\n模型: `{os.path.basename(onnx_path)}`  "
                f"输入: {ih}x{iw}  推理: {total_ms:.0f}ms  层数: {len(layers)}\n\n")
        f.write("| 层 | 类型 | shape | min | max | mean | std | 零% | absmax | p99.9 | outlier比 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for l in layers:
            f.write(f"| {l['name']} | {l['type']} | {l['shape']} | {l['min']} | {l['max']} | "
                    f"{l['mean']} | {l['std']} | {l['zero_pct']} | {l['absmax']} | "
                    f"{l['p999']} | {l['outlier']} |\n")
        f.write("\n> outlier比 = absmax / p99.9(|x|)。比值越大说明该层存在极端离群激活，"
                "int8 量化按 max 定 scale 会压缩其余值的分辨率——**量化敏感层候选**。\n")

    table_lines = ["层                          类型        shape                 min        max      mean       std   零%   absmax    p99.9  outlier"]
    for l in layers:
        sh = "x".join(str(s) for s in l["shape"][1:]) or "scalar"
        table_lines.append(
            f"{l['name'][:26]:26s} {l['type']:10s} {sh:20s} {l['min']:10.4f} {l['max']:10.4f} "
            f"{l['mean']:10.4f} {l['std']:9.4f} {l['zero_pct']:5.1f} {l['absmax']:9.4f} "
            f"{l['p999']:9.4f} {l['outlier']}")
    table = "\n".join(table_lines)

    print(json.dumps({"ok": True, "model": os.path.basename(onnx_path),
                      "input_hw": [ih, iw], "total_ms": round(total_ms, 1),
                      "n_layers": len(layers),
                      "sensitive": [{"name": l["name"], "outlier": l["outlier"]}
                                    for l in sens[:10]],
                      "report": report, "table": table}, ensure_ascii=False))


if __name__ == "__main__":
    main()
