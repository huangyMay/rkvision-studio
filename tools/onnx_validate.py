#!/usr/bin/env python3
"""PC 端 ONNX 验证：opencv 拉 RTSP → letterbox → onnxruntime 推理 → 结果/帧率报告。

用法:
  python3 onnx_validate.py <rtsp_url> <model.onnx> [输出目录] [秒数]

默认: rtsp://127.0.0.1:8554/hdmiin yolov8n-pose.onnx ./onnx_out 12
输出: <输出目录>/frame_NNN.jpg（带检测框）+ 控制台帧率/置信度汇总
"""
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

RTSP = sys.argv[1] if len(sys.argv) > 1 else "rtsp://127.0.0.1:8554/hdmiin"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/d/camera2/yolov8n-pose.onnx"
OUT = sys.argv[3] if len(sys.argv) > 3 else "./onnx_out"
SECONDS = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
INPUT_SIZE = 640
CONF_TH = 0.25

os.makedirs(OUT, exist_ok=True)

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
print(f"model: {os.path.basename(MODEL)}  input={inp_name} "
      f"{sess.get_inputs()[0].shape}  providers={sess.get_providers()}")

cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
if not cap.isOpened():
    print("ERROR: cannot open", RTSP)
    sys.exit(1)


def letterbox(img):
    h, w = img.shape[:2]
    scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
    top, left = (INPUT_SIZE - nh) // 2, (INPUT_SIZE - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def decode(out, scale, left, top, conf_th=CONF_TH):
    """标准 ultralytics 展平导出（ch, N)→(N, ch) 兼容解码。
    非标准头（4D per-level 等）返回 [] —— 特定解码属于模型家族插件。"""
    if out.ndim == 4:
        return []                              # per-level 4D 头：通用脚本不解码
    pred = out[0] if out.ndim == 3 else out
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                          # (ch, N) → (N, ch)
    ch = pred.shape[1]
    boxes = pred[:, :4]
    scores = pred[:, 4] * pred[:, 5:].max(axis=1) if ch > 5 else pred[:, 4]
    keep = scores > conf_th
    boxes, scores = boxes[keep], scores[keep]
    result = []
    for b, s in zip(boxes, scores):
        cx, cy, bw, bh = b
        x1 = (cx - bw / 2 - left) / scale
        y1 = (cy - bh / 2 - top) / scale
        x2 = (cx + bw / 2 - left) / scale
        y2 = (cy + bh / 2 - top) / scale
        result.append((x1, y1, x2, y2, float(s)))
    return result


n_frames = n_inf = 0
t0 = time.time()
t_end = t0 + SECONDS
fps_hist = []

while time.time() < t_end:
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.05)
        continue
    n_frames += 1
    ti = time.time()
    lb, scale, left, top = letterbox(frame)
    blob = cv2.dnn.blobFromImage(lb, 1.0 / 255.0, swapRB=True)
    out = sess.run(None, {inp_name: blob})[0]
    dets = decode(out, scale, left, top)
    dt = time.time() - ti
    n_inf += 1
    fps_hist.append(1.0 / dt if dt > 0 else 0)

    for (x1, y1, x2, y2, s) in dets:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{s:.2f}", (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if n_inf % 10 == 0:
        cv2.imwrite(f"{OUT}/frame_{n_inf:03d}.jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

cap.release()
el = time.time() - t0
print(f"\n== 结果 ==")
print(f"采集帧 {n_frames} · 推理 {n_inf} · 平均推理 {1000*el/max(1,n_inf):.0f}ms/帧"
      f" · 推理FPS {np.mean(fps_hist[-20:]) if fps_hist else 0:.1f}")
print(f"标注帧已存 {OUT}/")
