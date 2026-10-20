#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单图 ONNX 推理（RKVision Studio 本地验证功能的后端）。

用法:
  python3 onnx_infer.py <image> <model.onnx> [conf_th]

stdout 最后一行输出一个 JSON（其余日志一律走 stderr，便于 Qt 端解析）:
  {"ok":true,"model":..,"image":..,"image_w":..,"image_h":..,"input_size":640,
   "num_classes":N,
   "timings":{"load_ms":..,"preprocess_ms":..,"infer_ms":..,"postprocess_ms":..,"total_ms":..},
   "detections":[{"x1":..,"y1":..,"x2":..,"y2":..,"conf":..,"cls":N}]}
失败时: {"ok":false,"error":"..."}

解码兼容: 标准 ultralytics 展平导出 (ch,N)->(N,ch)，含 obj+cls 与无 obj(8参数) 两种；
非标准头(4D per-level 等)返回 ok=false 的解码说明——特定解码归模型家族插件。
"""
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort


def norm_path(p):
    """容忍 Windows 侧 UNC 路径：//wsl.localhost/<发行版>/home/... -> /home/..."""
    m = re.match(r'^[\\/]{2}wsl(?:\.localhost|\$)[\\/]([^\\/]+)[\\/](.*)$', p)
    if m:
        return '/' + m.group(2).replace('\\', '/')
    return p


def log(msg):
    print(msg, file=sys.stderr)


def letterbox(img, input_size):
    h, w = img.shape[:2]
    scale = min(input_size / h, input_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), 114, np.uint8)
    top, left = (input_size - nh) // 2, (input_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def iou_matrix(a, b):
    """a:[N,4] b:[M,4] xyxy -> IoU [N,M]"""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def nms(boxes, scores, iou_th=0.45):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_th]
    return keep


def _softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def decode(outputs, input_size, scale, left, top, conf_th, iou_th=0.45):
    """通用解码，返回 (dets, num_classes)；不支持返回 (None, None)。

    支持四种标准 ultralytics 形态:
      A. 单输出展平: (N,ch) / (1,N,ch) / (ch,N)
      B. 单输出 4D 图内已解码 (1,ch,H,W)，ch=4+K/5+K
      C. 同 ch 4D 金字塔裸头: [(1,64+K,H,W)...]（DFL+sigmoid 逐层解码）
      D. rknn_model_zoo 分组金字塔: 每层 3 输出 [(1,64,H,W),(1,K,H,W),(1,1,H,W)]
         （DFL + 类别 + objectness；K=nc；conf=sigmoid(obj)*sigmoid(cls).max）
    其余（pose 的 kpt 双头等）不支持——解码归模型家族插件。
    """
    if len(outputs) == 1 and outputs[0].ndim != 4:
        pred = outputs[0]
        if pred.ndim == 3:
            pred = pred[0]
        return decode_flat(pred, scale, left, top, conf_th, iou_th)
    if any(a.ndim != 4 for a in outputs):
        return None, None
    chs = {a.shape[1] for a in outputs}
    if len(chs) == 1:                                   # 全输出同 ch：裸头/金字塔/展平
        ch = chs.pop()
        if ch >= 65:                                    # 裸头: 64 DFL + nc
            return decode_dfl_pyramid(outputs, input_size, scale, left, top, conf_th, iou_th)
        pred = np.concatenate([a.reshape(ch, -1) for a in outputs], axis=1)
        return decode_flat(pred, scale, left, top, conf_th, iou_th)
    # 多 ch：rknn_model_zoo 分组金字塔——每层 [64 DFL, nc cls, 1 obj]
    groups = {}
    for a in outputs:
        groups.setdefault((a.shape[2], a.shape[3]), []).append(a)
    chs = {a.shape[1] for arrs in groups.values() for a in arrs}
    if 64 not in chs or 1 not in chs or len(chs - {64, 1}) != 1:
        return None, None
    return decode_dfl_grouped(groups, input_size, scale, left, top, conf_th, iou_th)


def decode_dfl_grouped(groups, input_size, scale, left, top, conf_th, iou_th):
    """rknn_model_zoo 分组金字塔：每层 [64 DFL, nc cls, 1 obj] 三输出。
    DFL 为原始 logits（需 softmax）；cls/obj 已在图内 sigmoid 激活，
    直接当概率用（黑图输入下恰为 0 可证），再叠 sigmoid 会把 0.87 压到 ~0.5。"""
    chs = {a.shape[1] for arrs in groups.values() for a in arrs}
    nc = next(iter(chs - {64, 1}))
    bins = np.arange(16, dtype=np.float32)
    per_level = []
    for (H, W), arrs in groups.items():
        dfl = next(a for a in arrs if a.shape[1] == 64)
        clsa = next(a for a in arrs if a.shape[1] == nc)
        obja = next(a for a in arrs if a.shape[1] == 1)
        p = dfl[0].astype(np.float32)
        reg = _softmax(p[:64].reshape(4, 16, H, W), axis=1)
        dist = (reg * bins[None, :, None, None]).sum(axis=1)
        cl = np.clip(clsa[0].astype(np.float32), 0.0, 1.0)
        cid = cl.argmax(0)
        conf = np.clip(obja[0].astype(np.float32), 0.0, 1.0)[0] * cl.max(0)
        stride = np.float32(input_size) / np.float32(W)
        xs = (np.arange(W, dtype=np.float32) + 0.5) * stride
        ys = (np.arange(H, dtype=np.float32) + 0.5) * stride
        gx, gy = np.meshgrid(xs, ys)
        l, t, r, b = dist
        m = conf > conf_th
        if not m.any():
            continue
        rows = np.stack([((gx - l * stride - left) / scale)[m],
                         ((gy - t * stride - top) / scale)[m],
                         ((gx + r * stride - left) / scale)[m],
                         ((gy + b * stride - top) / scale)[m],
                         conf[m], cid[m].astype(np.float32)], axis=1)
        per_level.append(rows)
    if not per_level:
        return [], nc
    allr = np.concatenate(per_level)
    boxes, confs, cids = allr[:, :4], allr[:, 4], allr[:, 5].astype(int)
    dets = []
    for c in np.unique(cids):
        mm = cids == c
        idx = nms(boxes[mm], confs[mm], iou_th)
        for i in idx:
            bb, ss = boxes[mm][i], confs[mm][i]
            dets.append((float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]),
                         float(ss), int(c)))
    dets.sort(key=lambda d: -d[4])
    return dets, nc


def decode_dfl_pyramid(levels, input_size, scale, left, top, conf_th, iou_th):
    """裸头逐层解码: ch=64+nc，前 64ch=4x16 DFL 分布，后 nc 需 sigmoid。"""
    nc = levels[0].shape[1] - 64
    bins = np.arange(16, dtype=np.float32)
    per_level = []
    for a in levels:
        _, _, H, W = a.shape
        p = a[0].astype(np.float32)
        reg = _softmax(p[:64].reshape(4, 16, H, W), axis=1)
        dist = (reg * bins[None, :, None, None]).sum(axis=1)   # (4,H,W) l,t,r,b
        cls = 1.0 / (1.0 + np.exp(-p[64:]))                    # (nc,H,W)
        cid = cls.argmax(0)
        conf = cls.max(0)
        stride = np.float32(input_size) / np.float32(W)
        xs = (np.arange(W, dtype=np.float32) + 0.5) * stride
        ys = (np.arange(H, dtype=np.float32) + 0.5) * stride
        gx, gy = np.meshgrid(xs, ys)
        l, t, r, b = dist
        m = conf > conf_th
        if not m.any():
            continue
        rows = np.stack([((gx - l * stride - left) / scale)[m],
                         ((gy - t * stride - top) / scale)[m],
                         ((gx + r * stride - left) / scale)[m],
                         ((gy + b * stride - top) / scale)[m],
                         conf[m], cid[m].astype(np.float32)], axis=1)
        per_level.append(rows)
    if not per_level:
        return [], nc
    allr = np.concatenate(per_level)
    boxes, confs, cids = allr[:, :4], allr[:, 4], allr[:, 5].astype(int)
    dets = []
    for c in np.unique(cids):
        mm = cids == c
        idx = nms(boxes[mm], confs[mm], iou_th)
        for i in idx:
            bb, ss = boxes[mm][i], confs[mm][i]
            dets.append((float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]),
                         float(ss), int(c)))
    dets.sort(key=lambda d: -d[4])
    return dets, nc


def decode_flat(pred, scale, left, top, conf_th, iou_th):
    """(ch,N) 或 (N,ch) 展平解码。"""
    if pred.ndim != 2:
        return None, None
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                          # (ch, N) -> (N, ch)
    ch = pred.shape[1]
    raw = pred[:, :4]
    if ch >= 5 + 1:
        scores_all = pred[:, 4:5] * pred[:, 5:]
        num_classes = ch - 5
    elif ch >= 4 + 1:
        scores_all = pred[:, 4:]
        num_classes = ch - 4
    else:
        return None, None
    cls_ids = scores_all.argmax(axis=1)
    confs = scores_all.max(axis=1)
    keep = confs > conf_th
    if not keep.any():
        return [], num_classes
    raw, confs, cls_ids = raw[keep], confs[keep], cls_ids[keep]
    boxes = np.empty_like(raw)
    boxes[:, 0] = (raw[:, 0] - raw[:, 2] / 2 - left) / scale
    boxes[:, 1] = (raw[:, 1] - raw[:, 3] / 2 - top) / scale
    boxes[:, 2] = (raw[:, 0] + raw[:, 2] / 2 - left) / scale
    boxes[:, 3] = (raw[:, 1] + raw[:, 3] / 2 - top) / scale
    dets = []
    for c in np.unique(cls_ids):
        m = cls_ids == c
        idx = nms(boxes[m], confs[m], iou_th)
        sub_boxes, sub_confs = boxes[m][idx], confs[m][idx]
        for b, s in zip(sub_boxes, sub_confs):
            dets.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                         float(s), int(c)))
    dets.sort(key=lambda d: -d[4])
    return dets, num_classes


def main():
    t_total = time.time()
    if len(sys.argv) < 3:
        print(json.dumps({"ok": False, "error": "usage: onnx_infer.py <image> <model.onnx> [conf_th] [iou_th]"}))
        return
    image_path = norm_path(sys.argv[1])
    model_path = norm_path(sys.argv[2])
    conf_th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    iou_th = float(sys.argv[4]) if len(sys.argv) > 4 else 0.45

    t0 = time.time()
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(json.dumps({"ok": False, "error": "cannot read image: " + image_path}))
        return
    t_img = (time.time() - t0) * 1000

    t0 = time.time()
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(json.dumps({"ok": False, "error": "load model: %s" % e}))
        return
    t_load = (time.time() - t0) * 1000

    inp = sess.get_inputs()[0]
    shape = [d for d in inp.shape if isinstance(d, int) and d > 1]
    input_size = shape[-1] if len(shape) >= 2 else 640
    if input_size not in (320, 416, 512, 640, 960, 1280):
        input_size = 640
    log("model: %s input=%s %s providers=%s" %
        (os.path.basename(model_path), inp.name, inp.shape, sess.get_providers()))

    t0 = time.time()
    lb, scale, left, top = letterbox(img, input_size)
    blob = cv2.dnn.blobFromImage(lb, 1.0 / 255.0, swapRB=True)
    t_pre = (time.time() - t0) * 1000

    t0 = time.time()
    try:
        outs = sess.run(None, {inp.name: blob})
    except Exception as e:
        print(json.dumps({"ok": False, "error": "inference: %s" % e}))
        return
    t_inf = (time.time() - t0) * 1000

    t0 = time.time()
    dets, num_classes = decode(outs, input_size, scale, left, top, conf_th, iou_th)
    if dets is None:
        shapes = [list(a.shape) for a in outs]
        print(json.dumps({"ok": False,
                          "error": "非标准输出头 %s——通用脚本不解码，需模型家族插件" % (shapes,)}))
        return
    t_post = (time.time() - t0) * 1000

    print(json.dumps({
        "ok": True,
        "model": os.path.basename(model_path),
        "image": os.path.basename(image_path),
        "image_w": int(img.shape[1]), "image_h": int(img.shape[0]),
        "input_size": int(input_size),
        "num_classes": num_classes,
        "timings": {
            "image_read_ms": round(t_img, 2),
            "load_ms": round(t_load, 2),
            "preprocess_ms": round(t_pre, 2),
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
