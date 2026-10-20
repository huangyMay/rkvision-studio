#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PC 实时推理 worker（常驻进程，Studio LIVE 预览叠加检测框用）。

帧协议（stdin/stdout，二进制安全）:
  启动握手: stdout 先发 {"type":"hello","input_size":N}（u32LE 前缀）
  之后每帧: stdin 收 4B 小端长度 + 载荷
           stdout 回 4B 小端长度 + JSON 一行
  载荷两类:
    JPEG 图像帧 → 推理 → {"ok":true,...,"type":"frame",...}
    b'CMD:{...}' 命令 → {"type":"prof",...}     # 目前支持 {"cmd":"profile","runs":20}
消息类型（JSON "type" 字段）:
  frame      每帧结果（含 detections 叠加数据 + pre/infer/post 耗时分解）
  model_info worker 启动时一次（参数量/算子直方图/文件大小）
  stats      每 100 帧滚动统计（平均/最快/最慢/推理FPS）
  prof       逐层剖析 TOP15（独立 profiling session，不影响主推理）
归一化与 onnx_infer.py 同口径（/255）。解码复用其四形态实现。
"""
import json
import os
import struct
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_infer import decode, letterbox  # noqa: E402


def emit(obj):
    payload = json.dumps(obj, ensure_ascii=False).encode('utf-8')
    sys.stdout.buffer.write(struct.pack('<I', len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def read_exact(stdin, n):
    buf = b''
    while len(buf) < n:
        chunk = stdin.buffer.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def model_info(model_path):
    """轻量静态信息（不跑 shape inference，亚秒级）。"""
    import onnx
    m = onnx.load(model_path)
    params = sum(int(np.prod(i.dims)) for i in m.graph.initializer)
    ops = Counter(n.op_type for n in m.graph.node)
    return params, ops.most_common(8)


def main():
    if len(sys.argv) < 2:
        emit({"ok": False, "error": "usage: onnx_stream.py <model.onnx> [conf] [iou]"})
        return
    model = sys.argv[1]
    conf_th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    iou_th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45

    import onnxruntime as ort
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dims = [d for d in inp.shape if isinstance(d, int) and d > 1]
    input_size = dims[-1] if len(dims) >= 2 else 640

    # 模型静态信息（一次）
    try:
        params, ops_list = model_info(model)
        emit({"type": "model_info",
              "file_mb": round(os.path.getsize(model) / 1048576.0, 1),
              "params": params,
              "op_hist": [{"op": k, "count": v} for k, v in ops_list]})
    except Exception as e:                                   # noqa: BLE001
        print("[model_info skipped]", e, file=sys.stderr)

    # 启动握手：与帧同协议（u32LE + JSON），{"type":"hello"} 表示就绪
    emit({"type": "hello", "input_size": input_size})
    sys.stdout.flush()

    last_jpg = None
    frame_idx = 0
    times = []

    while True:
        hdr = read_exact(sys.stdin, 4)
        if hdr is None:
            break
        (n,) = struct.unpack('<I', hdr)
        payload = read_exact(sys.stdin, n)
        if payload is None:
            break

        # ---- 命令帧（CMD: 前缀，旁路推理队列）----
        if payload[:4] == b'CMD:':
            try:
                cmd = json.loads(payload[4:].decode('utf-8'))
                if cmd.get("cmd") == "profile":
                    runs = int(cmd.get("runs", 20))
                    from onnx_analyze import profile as ort_profile
                    avg_total, layers = ort_profile(model, runs)
                    emit({"type": "prof", "runs": runs,
                          "avg_ms": round(avg_total, 1),
                          "layers": layers})
            except Exception as e:                           # noqa: BLE001
                emit({"type": "prof", "error": str(e)})
            continue

        # ---- JPEG 推理帧 ----
        last_jpg = payload
        t0 = time.time()
        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        t_dec = (time.time() - t0) * 1000
        if img is None:
            emit({"ok": False, "type": "frame", "error": "imdecode failed"})
            continue
        t0 = time.time()
        lb, scale, left, top = letterbox(img, input_size)
        blob = cv2.dnn.blobFromImage(lb, 1.0 / 255.0, swapRB=True)
        t_pre = (time.time() - t0) * 1000

        t0 = time.time()
        outs = sess.run(None, {inp.name: blob})
        t_inf = (time.time() - t0) * 1000

        t0 = time.time()
        dets, num_classes = decode(outs, input_size, scale, left, top, conf_th, iou_th)
        t_post = (time.time() - t0) * 1000
        if dets is None:
            emit({"ok": False, "type": "frame", "error": "非标准输出头", "infer_ms": round(t_inf, 1)})
            continue

        frame_idx += 1
        times.append(t_inf)
        top = f"cls{dets[0][5]}:{dets[0][4]:.2f}" if dets else "-"
        emit({
            "ok": True, "type": "frame",
            "w": int(img.shape[1]), "h": int(img.shape[0]),
            "frame": frame_idx,
            "pre_ms": round(t_pre, 1), "infer_ms": round(t_inf, 1),
            "post_ms": round(t_post, 1), "dec_ms": round(t_dec, 1),
            "n": len(dets), "top": top,
            "num_classes": num_classes,
            "detections": [{"x1": round(d[0], 1), "y1": round(d[1], 1),
                            "x2": round(d[2], 1), "y2": round(d[3], 1),
                            "conf": round(d[4], 4), "cls": d[5]} for d in dets],
        })

        # 每 100 帧滚动统计
        if frame_idx % 100 == 0 and times:
            recent = times[-100:]
            recent_sorted = sorted(recent)
            emit({"type": "stats", "frames": frame_idx,
                  "avg_ms": round(sum(recent) / len(recent), 1),
                  "min_ms": round(min(recent), 1),
                  "max_ms": round(max(recent), 1),
                  "p95_ms": round(recent_sorted[int(len(recent_sorted) * 0.95) - 1], 1),
                  "inf_fps": round(1000.0 / (sum(recent) / len(recent)), 1)})


if __name__ == "__main__":
    main()
