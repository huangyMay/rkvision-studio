#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量精度对比台：同一张图喂七个推理通道，量化/转换/板端实现损失全量对照。

七通道（模型实验室·全量对比）:
  ① onnx        ONNX float32 基线（onnxruntime，金标准）
  ② sim_fp16    RKNN FP16 模拟（x86 simulator，转换损失参考）
  ③ sim_int8    RKNN int8 模拟（量化损失参考）
  ④ sim_int16   RKNN int16 模拟（w16a16i，参考）
  ⑤ board_fp16  板上 FP16（真 NPU）
  ⑥ board_int8  板上 int8（真 NPU）
  ⑦ board_int16 板上 int16（真 NPU；RK3576 上可能回退 CPU）

逐层数据: int8/int16 各跑 accuracy_analysis（模拟器跑 sim 列；板在线再加跑 board 拿 runtime 列）。
用法:
  python3 acc_compare.py <model.onnx> <image> <device|''> [conf] [iou] [meanstd] [dataset.txt] [dump_dir]
输出: 逐行 STAGE/LAYERS/LOG 流式事件 + 末行 {"ok":true,...} 总报告
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


def build_rknn(onnx_path, quant, dataset, mean, std, quant_dtype=None):
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    kw = {"target_platform": "rk3576"}
    if mean is not None:
        kw["mean_values"] = mean
        kw["std_values"] = std
    if quant and quant_dtype:
        kw["quantized_dtype"] = quant_dtype
    if rknn.config(**kw) != 0:
        raise RuntimeError("config failed")
    if rknn.load_onnx(model=onnx_path) != 0:
        raise RuntimeError("load_onnx failed")
    if rknn.build(do_quantization=quant, dataset=dataset if quant else None) != 0:
        raise RuntimeError(f"build(do_quantization={quant}) failed")
    return rknn


def detect_family(onnx_path):
    import onnx as onnx_lib
    m = onnx_lib.load(onnx_path)
    names = " ".join(n.name for n in m.graph.node)
    if "/model.22/" in names or "model.22" in names:
        return "YOLOv8"
    if "/model.24/" in names or "model.24" in names:
        return "YOLOv5"
    return "通用"


def main():
    if len(sys.argv) < 4:
        print(json.dumps({"ok": False,
                          "error": "usage: acc_compare.py <model.onnx> <image> <device|''> "
                                   "[conf] [iou] [meanstd] [dataset.txt] [dump_dir]"}))
        return
    onnx_path = norm_path(sys.argv[1])
    image_path = norm_path(sys.argv[2])
    device = sys.argv[3]
    conf_th = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    iou_th = float(sys.argv[5]) if len(sys.argv) > 5 else 0.45
    meanstd = sys.argv[6] if len(sys.argv) > 6 else "0,0,0|255,255,255"
    dataset = norm_path(sys.argv[7]) if len(sys.argv) > 7 else None
    dump_dir = norm_path(sys.argv[8]) if len(sys.argv) > 8 else None

    mean, std = parse_meanstd(meanstd)
    t_all = time.time()

    def log(msg):
        print("LOG " + msg, flush=True)

    # 模型家族自动识别
    log(f"识别到 {detect_family(onnx_path)} 模型: {os.path.basename(onnx_path)}")

    # 网络设备：保活从第一刻开始
    if device:
        try:
            from rknn_acc import start_keepalive
            start_keepalive(device)
            log(f"板端连接: {device}（保活已启动）")
        except Exception:
            pass

    # dataset 自愈：缺失时用输入图自动生成
    if dataset and not os.path.exists(dataset):
        log(f"校准集 {dataset} 不存在——用输入图自动生成")
        dataset = None
    if not dataset and dump_dir:
        os.makedirs(dump_dir, exist_ok=True)      # 自愈写文件前先建 dump 目录
        dataset = os.path.join(dump_dir, "dataset_auto.txt")
        with open(dataset, "w", encoding="utf-8") as f:
            f.write(image_path + "\n")
        log(f"已自动生成校准集（1 张输入图）: {dataset}")

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
    nhwc = np.ascontiguousarray(blob.transpose(0, 2, 3, 1))              # RKNN NHWC
    if mean is not None:                                                 # mean/std 进图内：喂原始 0~255
        nhwc = np.ascontiguousarray(nhwc * 255.0)

    report = {"ok": True, "model": os.path.basename(onnx_path),
              "image": os.path.basename(image_path), "input_size": input_size,
              "meanstd": meanstd, "device": device or "(skipped)",
              "tensor_cosine": {}, "match": {}, "stages": {},
              "layers": {}}

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
        cos = []
        for x, y in zip(a_list, b_list):
            xf = np.asarray(x, dtype=np.float32).ravel()
            yf = np.asarray(y, dtype=np.float32).ravel()
            if xf.shape != yf.shape or xf.size == 0:
                continue
            denom = np.linalg.norm(xf) * np.linalg.norm(yf)
            cos.append(round(float(np.dot(xf, yf) / denom), 5) if denom > 0 else 1.0)
        return cos

    # ---------- ① ONNX 基线 ----------
    log("① 开始 ONNX 基线推理（onnxruntime float32）…")
    t0 = time.time()
    outs1 = sess0.run(None, {sess0.get_inputs()[0].name: blob})
    d1, _ = decode(outs1, input_size, scale, left, top, conf_th, iou_th)
    report["stages"]["1_onnx"] = {"ms": round((time.time() - t0) * 1000, 1),
                                  "n": len(d1) if d1 else 0}
    base = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in (d1 or [])]
    stage_dets = {"1_onnx": d1}
    log(f"① ONNX 基线完成: 基准检出 {len(base)} 框")
    emit_stage("1_onnx", {"image": stage_image("onnx_result.jpg",
                                               draw(img, d1, (0, 255, 0), "onnx"))})

    # ---------- ②③④ 模拟三连（x86 simulator） ----------
    sim_specs = [("2_sim_fp16", False, None, (128, 0, 255), "sim_fp16"),
                 ("3_sim_int8", True, None, (255, 128, 0), "sim_int8"),
                 ("4_sim_int16", True, "w16a16i", (128, 128, 255), "sim_int16")]
    for key, quant, dtype, color, tag in sim_specs:
        if quant and not dataset:
            report["stages"][key] = {"error": "无校准集"}
            continue
        label = {"2_sim_fp16": "② FP16 模拟", "3_sim_int8": "③ int8 模拟",
                 "4_sim_int16": "④ int16 模拟"}[key]
        log(f"{label}: 开始转换（build do_quantization={quant}"
            f"{', quantized_dtype=' + dtype if quant and dtype else ''}）…")
        t0 = time.time()
        try:
            rk = build_rknn(onnx_path, quant, dataset, mean, std, quant_dtype=dtype)
            log(f"{label}: 转换完成，开始模拟推理…")
            outs, dets, _ = rknn_infer_sim(rk, nhwc, input_size, scale, left, top,
                                           conf_th, iou_th)
            report["stages"][key] = {"ms": round((time.time() - t0) * 1000, 1),
                                     "n": len(dets) if dets else 0}
            report["tensor_cosine"][key + "_vs_1"] = tensor_cosine(outs1, outs)
            bb = [[d[0], d[1], d[2], d[3], d[4]] for d in (dets or [])]
            report["match"][key + "_vs_1"] = match_report(base, bb, label + " vs ONNX")
            stage_dets[key] = dets
            log(f"{label} 完成: 检出 {len(bb)} 框 · mean IoU "
                f"{report['match'][key + '_vs_1'].get('mean_iou', 0)}")
            emit_stage(key, {"image": stage_image(tag + "_result.jpg",
                                                  draw(img, dets, color, tag))})
            try:
                rk.release()
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001
            report["stages"][key] = {"error": str(e)[:200]}
            report["match"][key + "_vs_1"] = {"label": label, "error": str(e)[:200]}
            log(f"{label} 失败: {str(e)[:120]}")
            emit_stage(key)

    # ---------- ⑤⑥⑦ 板上三连（需板在线） ----------
    board_specs = [("5_board_fp16", False, None, (0, 128, 255), "bd_fp16"),
                   ("6_board_int8", True, None, (0, 128, 255), "bd_int8"),
                   ("7_board_int16", True, "w16a16i", (0, 128, 255), "bd_int16")]
    for key, quant, dtype, color, tag in board_specs:
        if not device:
            report["stages"][key] = {"skipped": True}
            continue
        label = {"5_board_fp16": "⑤ 板上 FP16", "6_board_int8": "⑥ 板上 int8",
                 "7_board_int16": "⑦ 板上 int16"}[key]
        note = "（w16a16i 在 RK3576 可能回退 CPU，较慢）" if key == "7_board_int16" else ""
        log(f"{label}: 开始转换{note}…")
        t0 = time.time()
        try:
            rk = build_rknn(onnx_path, quant, dataset, mean, std, quant_dtype=dtype)
            log(f"{label}: 转换完成，开始板上推理…")
            outs, dets, _ = rknn_infer_board(rk, nhwc, input_size, scale, left, top,
                                             conf_th, iou_th, device)
            report["stages"][key] = {"ms": round((time.time() - t0) * 1000, 1),
                                     "n": len(dets) if dets else 0}
            report["tensor_cosine"][key + "_vs_1"] = tensor_cosine(outs1, outs)
            bb = [[d[0], d[1], d[2], d[3], d[4]] for d in (dets or [])]
            report["match"][key + "_vs_1"] = match_report(base, bb, label + " vs ONNX")
            stage_dets[key] = dets
            log(f"{label} 完成: 检出 {len(bb)} 框 · mean IoU "
                f"{report['match'][key + '_vs_1'].get('mean_iou', 0)}")
            emit_stage(key, {"image": stage_image(tag + "_result.jpg",
                                                  draw(img, dets, color, tag))})
            try:
                rk.release()
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001
            report["stages"][key] = {"error": str(e)[:200]}
            report["match"][key + "_vs_1"] = {"label": label, "error": str(e)[:200]}
            log(f"{label} 失败: {str(e)[:120]}")
            emit_stage(key)

    # ---------- 逐层精度（int8/int16 各跑 accuracy_analysis：模拟器必跑，板在线加跑板端） ----------
    if dump_dir and dataset:
        from rknn_acc import parse_error_analysis
        acc_dir = os.path.join(dump_dir, "acc")
        inputs = [image_path]
        acc_specs = [("int8", None), ("int16", "w16a16i")]
        for qname, dtype in acc_specs:
            layers = []
            # 模拟器列
            try:
                rk = build_rknn(onnx_path, True, dataset, mean, std, quant_dtype=dtype)
                rk.accuracy_analysis(inputs=inputs, output_dir=os.path.join(acc_dir, "sim_" + qname),
                                     target=None)
                layers = parse_error_analysis(os.path.join(acc_dir, "sim_" + qname,
                                                           "error_analysis.txt"))
                try:
                    rk.release()
                except Exception:
                    pass
            except Exception as e:  # noqa: BLE001
                log(f"逐层·模拟{qname} 失败: {str(e)[:100]}")
            # 板端列（追加 runtime 列）
            if device and layers:
                try:
                    rk = build_rknn(onnx_path, True, dataset, mean, std, quant_dtype=dtype)
                    rk.accuracy_analysis(inputs=inputs,
                                         output_dir=os.path.join(acc_dir, "board_" + qname),
                                         target="rk3576", device_id=device)
                    bll = parse_error_analysis(os.path.join(acc_dir, "board_" + qname,
                                                            "error_analysis.txt"))
                    bmap = {l["name"]: l for l in bll}
                    for l in layers:
                        b = bmap.get(l["name"])
                        if b:
                            l["rt_entire_cos"] = b.get("rt_entire_cos", -1)
                            l["rt_single_cos"] = b.get("rt_single_cos", -1)
                    try:
                        rk.release()
                    except Exception:
                        pass
                except Exception as e:  # noqa: BLE001
                    log(f"逐层·板{qname} 失败: {str(e)[:100]}")
            if layers:
                report["layers"][qname] = layers
                lows = [l for l in layers if l.get("sim_single_cos", 1) < 0.99]
                log(f"逐层·{qname}: {len(layers)} 层 · 最低 sim_single "
                    f"{min((l['sim_single_cos'] for l in layers if 'sim_single_cos' in l), default=1.0):.5f}"
                    f" · 低余弦 {len(lows)} 层")
                print("LAYERS " + json.dumps({qname: layers}, ensure_ascii=False), flush=True)

    report["total_ms"] = round((time.time() - t_all) * 1000, 1)

    if dump_dir:
        json.dump({k: [list(map(float, d)) for d in (v or [])]
                   for k, v in stage_dets.items()},
                  open(os.path.join(dump_dir, "detections.json"), "w"))
        report["dump_dir"] = dump_dir

    print(json.dumps(report, ensure_ascii=False))


def rknn_infer_sim(rk, nhwc, input_size, scale, left, top, conf_th, iou_th):
    if rk.init_runtime(target=None) != 0:
        raise RuntimeError("init_runtime(simulator) failed")
    outs = rk.inference(inputs=[nhwc])
    if outs is None:
        raise RuntimeError("sim inference returned None")
    dets, _ = decode(outs, input_size, scale, left, top, conf_th, iou_th)
    return outs, dets, None


def rknn_infer_board(rk, nhwc, input_size, scale, left, top, conf_th, iou_th, device):
    if rk.init_runtime(target="rk3576", device_id=device) != 0:
        raise RuntimeError(f"init_runtime(board {device}) failed")
    outs = rk.inference(inputs=[nhwc])
    if outs is None:
        raise RuntimeError("board inference returned None")
    dets, _ = decode(outs, input_size, scale, left, top, conf_th, iou_th)
    return outs, dets, None


def match_report(base, other, label, iou_th=0.5):
    import numpy as np
    if not base:
        return {"label": label, "note": "基准无检出", "matched": 0, "missed": 0,
                "extra": len(other), "conf_drops": [], "shifts": [], "mean_iou": 0.0}
    b = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in base], dtype=np.float32)
    om = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in other], dtype=np.float32) \
        if other else np.zeros((0, 5), dtype=np.float32)
    ious = iou_matrix(b[:, :4], om[:, :4]) if len(om) else np.zeros((len(b), 0))
    matched, used_o = [], set()
    order = np.argsort(-b[:, 4])
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
            matched.append({"base_conf": float(b[bi, 4]), "iou": float(best_i),
                            "dconf": dconf,
                            "shift_px": round(float(np.hypot(cx_o - cx_b, cy_o - cy_b)), 1)})
    return {"label": label, "n_base": len(base), "n_other": len(other),
            "matched": len(matched), "missed": len(base) - len(matched),
            "extra": len(other) - len(matched),
            "mean_iou": round(sum(m["iou"] for m in matched) / len(matched), 4) if matched else 0.0,
            "conf_drops": sorted([m["dconf"] for m in matched], reverse=True)[:5],
            "max_shift_px": max((m["shift_px"] for m in matched), default=0.0),
            "detail": sorted(matched, key=lambda m: -m["base_conf"])[:8]}


if __name__ == "__main__":
    main()
