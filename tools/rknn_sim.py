#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RKNN 模拟推理（rknn-toolkit2 simulator，PC 上无板执行）。

关键事实：toolkit2 不支持 load_rknn 后模拟——模拟必须走
  load_onnx(mean/std) → build(do_quantization=False) → init_runtime(target=None) → inference
因此本脚本的输入是 ONNX+转换配置，副产品是 .rknn（已存盘，可直接上板）。

用法:
  python3 rknn_sim.py <image> <model.onnx> [conf] [iou] [meanstd] [save_rknn]

  meanstd = "0,0,0|255,255,255"（转换期 mean_values/std_values，图内归一化，喂原始 0~255）
          或 "none|none"（无归一化，喂 /255——与 ONNX 推理同口径）

stdout 最后一行 JSON（与 onnx_infer.py 同构）；timings.infer_ms 为 x86 模拟耗时，
不代表板端 NPU 性能。build 编译耗时计入 build_ms，单图首次 ~数十秒属正常。
"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_infer import decode, letterbox, norm_path  # noqa: E402

INPUT_SIZE_FALLBACK = 640


def parse_meanstd(s):
    """'0,0,0/255,255,255' -> ([[0,0,0]],[[255,255,255]])；'none/none' -> (None,None)
    分隔符用 /（QProcess→wsl→bash 链路中 | ; & 等会被 shell 解析）。"""
    if not s or s.lower().startswith("none"):
        return None, None
    s = s.replace("|", "/")                              # 兼容旧写法
    mean_s, std_s = s.split("/")
    mean = [[float(v) for v in mean_s.split(",")]]
    std = [[float(v) for v in std_s.split(",")]]
    return mean, std


def main():
    t_total = time.time()
    if len(sys.argv) < 3:
        print(json.dumps({"ok": False, "error": "usage: rknn_sim.py <image> <model.onnx> [conf] [iou] [meanstd] [save_rknn]"}))
        return
    image_path = norm_path(sys.argv[1])
    onnx_path = norm_path(sys.argv[2])
    conf_th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    iou_th = float(sys.argv[4]) if len(sys.argv) > 4 else 0.45
    meanstd = sys.argv[5] if len(sys.argv) > 5 else "0,0,0|255,255,255"
    save_rknn = norm_path(sys.argv[6]) if len(sys.argv) > 6 else None

    t0 = time.time()
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(json.dumps({"ok": False, "error": "cannot read image: " + image_path}))
        return
    t_img = (time.time() - t0) * 1000

    lb, scale, left, top = letterbox(img, INPUT_SIZE_FALLBACK)
    blob_nchw = cv2.dnn.blobFromImage(lb, 1.0 / 255.0, swapRB=True)  # (1,3,H,W) RGB 0~1
    mean, std = parse_meanstd(meanstd)
    nchw = blob_nchw if mean is None else (blob_nchw * 255.0)        # 图内归一化则喂原值

    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    try:
        t0 = time.time()
        if mean is not None:                                 # 2.x：mean/std 在 config 里
            if rknn.config(mean_values=mean, std_values=std, target_platform="rk3576") != 0:
                raise RuntimeError("config failed")
        if rknn.load_onnx(model=onnx_path) != 0:
            raise RuntimeError("load_onnx failed")
        if rknn.build(do_quantization=False) != 0:                   # fp16 模拟；量化属二期
            raise RuntimeError("build failed")
        t_build = (time.time() - t0) * 1000
        if save_rknn:
            rknn.export_rknn(save_rknn)

        if rknn.init_runtime(target=None) != 0:                      # None = PC 模拟器
            raise RuntimeError("init_runtime(target=None) failed")

        nhwc = np.ascontiguousarray(nchw.transpose(0, 2, 3, 1))      # toolkit2 期望 NHWC
        t0 = time.time()
        outs = rknn.inference(inputs=[nhwc])
        t_inf = (time.time() - t0) * 1000
        if outs is None:
            raise RuntimeError("inference returned None")
    except Exception as e:                                           # noqa: BLE001
        print(json.dumps({"ok": False, "error": "simulator: %s" % e}))
        return
    finally:
        try:
            rknn.release()
        except Exception:                                            # noqa: BLE001
            pass

    t0 = time.time()
    dets, num_classes = decode(outs, INPUT_SIZE_FALLBACK, scale, left, top, conf_th, iou_th)
    if dets is None:
        shapes = [list(a.shape) for a in outs]
        print(json.dumps({"ok": False,
                          "error": "非标准输出头 %s——通用脚本不解码，需模型家族插件" % (shapes,)}))
        return
    t_post = (time.time() - t0) * 1000

    print(json.dumps({
        "ok": True,
        "model": os.path.basename(onnx_path),
        "image": os.path.basename(image_path),
        "image_w": int(img.shape[1]), "image_h": int(img.shape[0]),
        "input_size": int(INPUT_SIZE_FALLBACK),
        "num_classes": num_classes,
        "simulator": True,
        "meanstd": meanstd,
        "rknn_saved": bool(save_rknn and os.path.exists(save_rknn)),
        "timings": {
            "image_read_ms": round(t_img, 2),
            "build_ms": round(t_build, 2),
            "preprocess_ms": 0.0,
            "infer_ms": round(t_inf, 2),
            "postprocess_ms": round(t_post, 2),
            "total_ms": round((time.time() - t_total) * 1000, 2),
        },
        "detections": [{"x1": round(d[0], 1), "y1": round(d[1], 1),
                        "x2": round(d[2], 1), "y2": round(d[3], 1),
                        "conf": round(d[4], 4), "cls": d[5]} for d in dets],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
