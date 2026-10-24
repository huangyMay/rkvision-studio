#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""精度对比台：同一张图喂四级推理，量化/转换损失一表看清。

四级（User Guide 第 7 章精度排查漏斗的 Studio 化）:
  ① ONNX float      onnxruntime 直推              —— 金标准
  ② RKNN FP16 模拟  build(do_quantization=False)  —— 转换损失
  ③ RKNN int8 模拟  build(do_quantization=True)   —— 量化损失
  ④ 板上 int8       init_runtime(target,device)   —— 板端实现（需网络/USB adb + rknn_server）

对比方法: 以 ① 的检测框为基准，②③④ 各做 IoU 贪心匹配；
          每个匹配框报告 conf 差与中心点漂移；另给最终输出张量的余弦。

用法:
  python3 acc_compare.py <model.onnx> <image> <device> [conf] [iou] [meanstd] [dataset.txt]
    device 为空字符串则跳过 ④；meanstd 默认 "none|none"（与 ONNX 同口径 /255）
"""
import json
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_infer import decode, letterbox, norm_path, iou_matrix  # noqa: E402
from rknn_sim import parse_meanstd  # noqa: E402


def run_onnx(onnx_path, blob, input_size, scale, left, top, conf_th, iou_th):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    outs = sess.run(None, {inp.name: blob})
    return outs


def build_rknn(onnx_path, quant, dataset, mean, std):
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    kw = {"target_platform": "rk3576"}
    if mean is not None:
        kw["mean_values"] = mean
        kw["std_values"] = std
    if rknn.config(**kw) != 0:
        raise RuntimeError("config failed")
    if rknn.load_onnx(model=onnx_path) != 0:
        raise RuntimeError("load_onnx failed")
    if rknn.build(do_quantization=quant, dataset=dataset if quant else None) != 0:
        raise RuntimeError(f"build(do_quantization={quant}) failed")
    return rknn


def rknn_infer_decode(rknn, blob_nhwc, input_size, scale, left, top,
                      conf_th, iou_th, target=None, device=None):
    if rknn.init_runtime(target=target, device_id=device) != 0:
        raise RuntimeError(f"init_runtime failed (target={target}, device={device})")
    outs = rknn.inference(inputs=[blob_nhwc])
    dets, ncls = decode(outs, input_size, scale, left, top, conf_th, iou_th)
    return outs, dets, ncls


def to_boxes(dets):
    return [[d[0], d[1], d[2], d[3]] for d in dets] if dets else []


def match_report(base, other, label, iou_th=0.5):
    """base/other: [[x1,y1,x2,y2,conf,cls],...] → 框级 IoU 贪心匹配报告。"""
    import numpy as np
    if not base:
        return {"label": label, "note": "基准无检出", "matched": 0, "missed": 0,
                "extra": len(other), "conf_drops": [], "shifts": [], "mean_iou": 0.0}
    b = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in base], dtype=np.float32)
    om = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in other], dtype=np.float32) \
        if other else np.zeros((0, 5), dtype=np.float32)
    ious = iou_matrix(b[:, :4], om[:, :4]) if len(om) else np.zeros((len(b), 0))

    matched, used_o = [], set()
    order = np.argsort(-b[:, 4])                       # 按 conf 降序贪心
    for bi in order:
        best_o, best_i = -1, iou_th
        for oi in range(om.shape[0]):
            if oi in used_o:
                continue
            if ious[bi, oi] > best_i:
                best_o, best_i = oi, ious[bi, oi]
        if best_o >= 0:
            used_o.add(best_o)
            dconf = float(om[best_o, 4] - b[bi, 4])
            cx_b = (b[bi, 0] + b[bi, 2]) / 2
            cy_b = (b[bi, 1] + b[bi, 3]) / 2
            cx_o = (om[best_o, 0] + om[best_o, 2]) / 2
            cy_o = (om[best_o, 1] + om[best_o, 3]) / 2
            shift = float(np.hypot(cx_o - cx_b, cy_o - cy_b))
            matched.append({"base_conf": float(b[bi, 4]), "iou": float(best_i),
                            "dconf": dconf, "shift_px": round(shift, 1)})
    missed = len(base) - len(matched)
    extra = len(other) - len(matched)
    di = [m["dconf"] for m in matched]
    return {"label": label, "n_base": len(base), "n_other": len(other),
            "matched": len(matched), "missed": missed, "extra": extra,
            "mean_iou": round(sum(m["iou"] for m in matched) / len(matched), 4) if matched else 0.0,
            "conf_drops": sorted([m["dconf"] for m in matched], reverse=True)[:5],
            "max_shift_px": max((m["shift_px"] for m in matched), default=0.0),
            "detail": sorted(matched, key=lambda m: -m["base_conf"])[:8]}


def main():
    if len(sys.argv) < 4:
        print(json.dumps({"ok": False,
                          "error": "usage: acc_compare.py <model.onnx> <image> <device|''> "
                                   "[conf] [iou] [meanstd] [dataset.txt]"}))
        return
    onnx_path = norm_path(sys.argv[1])
    image_path = norm_path(sys.argv[2])
    device = sys.argv[3]
    conf_th = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    iou_th = float(sys.argv[5]) if len(sys.argv) > 5 else 0.45
    meanstd = sys.argv[6] if len(sys.argv) > 6 else "0,0,0|255,255,255"
    dataset = norm_path(sys.argv[7]) if len(sys.argv) > 7 else None
    dump_dir = norm_path(sys.argv[8]) if len(sys.argv) > 8 else None   # 对比视图：落盘标注图+detections

    mean, std = parse_meanstd(meanstd)          # 已是 config 格式 [[...]]；"none" → None
    t_all = time.time()

    import onnxruntime as ort
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(json.dumps({"ok": False, "error": f"cannot read image: {image_path}"}))
        return

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess0 = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    inp = sess0.get_inputs()[0]
    dims = [d for d in inp.shape if isinstance(d, int) and d > 1]
    input_size = dims[-1] if len(dims) >= 2 and dims[-1] in (320, 416, 512, 640, 960, 1280) else 640

    lb, scale, left, top = letterbox(img, input_size)
    blob = cv2.dnn.blobFromImage(lb, 1.0 / 255.0, swapRB=True)           # ① /255 口径
    nhwc = np.ascontiguousarray(blob.transpose(0, 2, 3, 1))              # RKNN 期望 NHWC
    if mean is not None:                                                 # ②③④ mean/std 进图内：喂原始 0~255
        nhwc = np.ascontiguousarray(nhwc * 255.0)

    report = {"ok": True, "model": os.path.basename(onnx_path),
              "image": os.path.basename(image_path), "input_size": input_size,
              "meanstd": meanstd, "device": device or "(skipped)",
              "tensor_cosine": {}, "match": {}, "stages": {}}

    # 流式输出：每完成一级立刻打印 STAGE 行（Qt 逐行解析、渐进渲染）
    def emit_stage(stage_key, extra=None):
        payload = {"stage": stage_key, "stages": report["stages"],
                   "tensor_cosine": report["tensor_cosine"],
                   "match": report["match"]}
        if extra:
            payload.update(extra)
        print("STAGE " + json.dumps(payload, ensure_ascii=False), flush=True)

    def draw(img_src, dets, color, tag):
        c = img_src.copy()
        for d in (dets or []):
            p1, p2 = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
            cv2.rectangle(c, p1, p2, color, 2)
            cv2.putText(c, f"{tag} {d[4]:.2f}", (p1[0], max(14, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return c

    def stage_image(name, img_annotated):
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            p = os.path.join(dump_dir, name)
            cv2.imwrite(p, img_annotated)
            return p
        return ""

    def tensor_cosine(a_list, b_list):
        import numpy as np
        cos = []
        for x, y in zip(a_list, b_list):
            xf = np.asarray(x, dtype=np.float32).ravel()
            yf = np.asarray(y, dtype=np.float32).ravel()
            if xf.shape != yf.shape or xf.size == 0:
                continue
            denom = np.linalg.norm(xf) * np.linalg.norm(yf)
            cos.append(round(float(np.dot(xf, yf) / denom), 5) if denom > 0 else 1.0)
        return cos

    # ① ONNX 金标准
    t0 = time.time()
    outs1 = run_onnx(onnx_path, blob, input_size, scale, left, top, conf_th, iou_th)
    d1, _ = decode(outs1, input_size, scale, left, top, conf_th, iou_th)
    report["stages"]["1_onnx"] = {"ms": round((time.time() - t0) * 1000, 1),
                                  "n": len(d1) if d1 else 0}
    base = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in (d1 or [])]
    emit_stage("1_onnx", {"image": stage_image("onnx_result.jpg",
                                               draw(img, d1, (0, 255, 0), "onnx"))})

    # ② RKNN FP16 模拟
    t0 = time.time()
    rk_fp16 = build_rknn(onnx_path, False, None, mean, std)
    o2, d2, _ = rknn_infer_decode(rk_fp16, nhwc, input_size, scale, left, top, conf_th, iou_th)
    report["stages"]["2_fp16_sim"] = {"ms": round((time.time() - t0) * 1000, 1),
                                      "n": len(d2) if d2 else 0}
    report["tensor_cosine"]["2_vs_1"] = tensor_cosine(outs1, o2)
    b2 = [[d[0], d[1], d[2], d[3], d[4]] for d in (d2 or [])]
    report["match"]["2_vs_1"] = match_report(base, b2, "FP16模拟 vs ONNX")
    emit_stage("2_fp16_sim")
    try:
        rk_fp16.release()
    except Exception:
        pass

    # ③ RKNN int8 模拟（量化构建）
    if not dataset:
        print(json.dumps({"ok": False,
                          "error": "③ int8 需要校准集 dataset.txt（每行一张图），或用 ③只测 fp16"}))
        return
    t0 = time.time()
    rk_i8 = build_rknn(onnx_path, True, dataset, mean, std)
    report["stages"]["3_int8_sim"] = {"ms": round((time.time() - t0) * 1000, 1)}
    o3, d3, _ = rknn_infer_decode(rk_i8, nhwc, input_size, scale, left, top, conf_th, iou_th)
    report["stages"]["3_int8_sim"]["n"] = len(d3) if d3 else 0
    report["tensor_cosine"]["3_vs_1"] = tensor_cosine(outs1, o3)
    b3 = [[d[0], d[1], d[2], d[3], d[4]] for d in (d3 or [])]
    report["match"]["3_vs_1"] = match_report(base, b3, "int8模拟 vs ONNX")
    emit_stage("3_int8_sim", {"image": stage_image("int8sim_result.jpg",
                                                   draw(img, d3, (255, 128, 0), "i8sim"))})

    # ④ 板上 int8（同一 int8 模型，切真板 runtime）
    d4 = None
    if device:
        t0 = time.time()
        try:
            from rknn_acc import start_keepalive
            start_keepalive(device)
        except Exception:
            pass
        try:
            o4, d4, _ = rknn_infer_decode(rk_i8, nhwc, input_size, scale, left, top,
                                          conf_th, iou_th, target="rk3576", device=device)
            report["stages"]["4_board"] = {"ms": round((time.time() - t0) * 1000, 1),
                                           "n": len(d4) if d4 else 0}
            report["tensor_cosine"]["4_vs_1"] = tensor_cosine(outs1, o4)
            b4 = [[d[0], d[1], d[2], d[3], d[4]] for d in (d4 or [])]
            report["match"]["4_vs_1"] = match_report(base, b4, "板上int8 vs ONNX")
            emit_stage("4_board", {"image": stage_image("board_result.jpg",
                                                        draw(img, d4, (0, 128, 255), "board"))})
        except Exception as e:  # noqa: BLE001
            report["stages"]["4_board"] = {"error": str(e)[:200]}
            report["match"]["4_vs_1"] = {"label": "板上int8", "error": str(e)[:200]}
            emit_stage("4_board")

    # 逐层精度（复用 ③ 的 int8 RKNN 对象直接 accuracy_analysis——不重新构建）
    if dump_dir:
        try:
            from rknn_acc import parse_error_analysis
            acc_dir = os.path.join(dump_dir, "acc")
            inputs = [image_path]
            rk_i8.accuracy_analysis(inputs=inputs, output_dir=acc_dir,
                                    target="rk3576", device_id=device if device else None)
            layers = parse_error_analysis(os.path.join(acc_dir, "error_analysis.txt"))
            if layers:
                report["layers"] = layers
                lows = [l for l in layers if l.get("sim_single_cos", 1) < 0.99]
                report["min_single"] = min((l["sim_single_cos"] for l in layers
                                            if "sim_single_cos" in l), default=1.0)
                report["n_low"] = len(lows)
                print("LAYERS " + json.dumps(layers, ensure_ascii=False), flush=True)
        except Exception as e:  # noqa: BLE001
            report["layers_error"] = str(e)[:200]

    report["total_ms"] = round((time.time() - t_all) * 1000, 1)

    # detections 留档（画面已在各阶段即时落盘）
    if dump_dir:
        json.dump({"1_onnx": [list(map(float, d)) for d in (d1 or [])],
                   "3_int8_sim": [list(map(float, d)) for d in (d3 or [])],
                   "4_board": [list(map(float, d)) for d in (d4 or [])]},
                  open(os.path.join(dump_dir, "detections.json"), "w"))
        report["dump_dir"] = dump_dir

    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
