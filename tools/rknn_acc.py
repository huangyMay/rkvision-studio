#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RKNN 量化精度分析（toolkit2 accuracy_analysis，真板执行——模拟器跑不了）。

前提（一次性）:
  1. 板上 /data/local/tmp/{rknn_server, librknnrt.so}（2.3.2 配套版）已推、rknn_server 已启动
  2. adb 可达：USB（~/bin/adb 共享 Windows server）或网络（IP:5555，adb tcpip 5555）
  3. 模型须是 int8 量化 .rknn（fp16 模型无量化误差可分析）

用法:
  python3 rknn_acc.py <model.rknn> <device> [image] [out_dir]

输出 JSON: {"ok":true,"out_dir":..,"layers":[{"name":..,"cosine":..}...],"raw":..}
accuracy_analysis 逐层对比 模拟器 vs 真板 的每层输出（余弦相似度），定位量化敏感层。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_infer import norm_path  # noqa: E402

HOME_BIN = os.environ.get("RVS_BOARD_BIN") or os.path.expanduser("~/bin")


def parse_error_analysis(path):
    """error_analysis.txt → [{name, cosine}]（兼容 2.3.x 的表格排版）。"""
    layers = []
    if not os.path.exists(path):
        return layers
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^(\S+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line.strip())
        if not m:
            m2 = re.search(r"(cosine|余弦)[^\d]*([0-9.]+)", line, re.IGNORECASE)
            if m2 and len(line) > 10:
                layers.append({"name": line.split()[0][:60], "cosine": float(m2.group(2))})
            continue
        layers.append({"name": m.group(1), "cosine": float(m.group(3))})
    return layers


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"ok": False, "error": "usage: rknn_acc.py <model.rknn> <device> [image] [out_dir]"}))
        return
    model = norm_path(sys.argv[1])
    device = sys.argv[2]
    image = norm_path(sys.argv[3]) if len(sys.argv) > 3 else None
    out_dir = norm_path(sys.argv[4]) if len(sys.argv) > 4 else "/tmp/rknn_acc_out"

    os.environ["PATH"] = HOME_BIN + os.pathsep + os.environ.get("PATH", "")

    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    if rknn.load_rknn(model) != 0:
        print(json.dumps({"ok": False, "error": f"load_rknn failed: {model}"}))
        return
    try:
        if rknn.init_runtime(target="rk3576", device_id=device) != 0:
            print(json.dumps({"ok": False,
                              "error": f"init_runtime failed (device={device})——查 rknn_server 已启动、adb 可达"}))
            return
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"init_runtime raised (device={device}): {e}"}))
        return

    os.makedirs(out_dir, exist_ok=True)
    inputs = [image] if image else [os.path.join(os.path.dirname(model), "..",
                                                 "assets/test_images/test.jpg")]
    inputs = [p for p in inputs if os.path.exists(p)]
    if not inputs:
        print(json.dumps({"ok": False, "error": "no input image found"}))
        return

    try:
        rknn.accuracy_analysis(inputs=inputs, output_dir=out_dir,
                               target="rk3576", device_id=device)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"accuracy_analysis raised: {e}"}))
        return

    layers = parse_error_analysis(os.path.join(out_dir, "error_analysis.txt"))
    print(json.dumps({"ok": True, "out_dir": out_dir, "layers": layers,
                      "n_layers": len(layers)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
