#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 模型分析（RKVision Studio 模型分析功能后端）。

用法: python3 onnx_analyze.py <model.onnx> [profile_runs]

stdout 最后一行 JSON（日志走 stderr）:
{"ok":true,"model":..,"file_mb":..,
 "inputs":[{"name":..,"shape":[..],"dtype":..}],
 "outputs":[...],
 "params":31500800, "macs":8.1e9,          // macs=null 表示 shape 推断失败未估算
 "op_hist":[{"op":"Conv","count":53},...],
 "runs":20,"avg_total_ms":5.2,
 "top_layers":[{"name":..,"op":..,"ms":1.23},...]}  // 单次推理平均耗时 TOP15
"""
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import onnxruntime as ort


def log(msg):
    print(msg, file=sys.stderr)


def dims_of(tt):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param or "?"
            for d in tt.shape.dim]


def elem_size(dtype):
    sizes = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8}
    return sizes.get(dtype, 4)


def dtype_name(dtype):
    names = {1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
             6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "double"}
    return names.get(dtype, "type%d" % dtype)


def iolist_of(vals):
    out = []
    for v in vals:
        tt = v.type.tensor_type
        out.append({"name": v.name, "shape": dims_of(tt),
                    "dtype": dtype_name(tt.elem_type)})
    return out


def static_info(model_path):
    """结构/参数量/FLOPs（onnx shape inference）。失败不致命，逐项降级。"""
    import onnx
    from onnx import shape_inference
    m = onnx.load(model_path)
    g = m.graph

    def iolist(vals):
        out = []
        for v in vals:
            tt = v.type.tensor_type
            out.append({"name": v.name, "shape": dims_of(tt),
                        "dtype": tt.elem_type})
        return out

    inputs, outputs = iolist_of(g.input), iolist_of(g.output)
    op_hist = Counter(n.op_type for n in g.node)
    params = sum(int(np.prod(init.dims)) * elem_size(init.data_type)
                 for init in g.initializer)

    macs = 0
    macs_ok = True
    try:
        inferred = shape_inference.infer_shapes(m)
        vi = {v.name: dims_of(v.type.tensor_type) for v in inferred.graph.value_info}
        ini = {k.name: list(k.dims) for k in inferred.graph.initializer}

        def in_dims(name):                       # 输入可能是权重（initializer）或激活
            return vi.get(name) or ini.get(name)

        for n in inferred.graph.node:
            ins = [in_dims(x) for x in n.input]
            outs = [vi.get(x) for x in n.output]
            if n.op_type == "Conv" and len(ins) >= 2 and ins[1] and outs and outs[0]:
                w = ins[1]
                o = outs[0]
                if len(w) == 4 and len(o) == 4 and all(isinstance(d, int) for d in o[2:4]):
                    # MACs = Cout*OH*OW*(Cin/g)*KH*KW；o[1] 已是 Cout，勿再乘 w[0]
                    macs += o[1] * o[2] * o[3] * w[1] * w[2] * w[3]
            elif n.op_type == "Gemm" and len(ins) >= 2 and ins[0] and ins[1]:
                a, b = ins[0], ins[1]
                if len(a) >= 2 and len(b) == 2 and isinstance(a[-1], int) \
                        and isinstance(b[-1], int) and isinstance(b[-2], int):
                    rows = a[-2] if len(a) == 2 else abs(np.prod([d for d in a[:-1] if isinstance(d, int)]))
                    macs += rows * b[-1] * b[-2]
            elif n.op_type == "MatMul" and len(ins) >= 2 and ins[0] and ins[1]:
                a, b = ins[0], ins[1]
                if len(a) >= 2 and len(b) >= 2 and isinstance(a[-1], int) \
                        and isinstance(b[-1], int) and isinstance(b[-2], int):
                    batch = abs(np.prod([d for d in a[:-1] if isinstance(d, int)]) or 1)
                    macs += batch * a[-1] * b[-1] * b[-2]
    except Exception as e:                                   # noqa: BLE001
        log("shape inference failed: %s" % e)
        macs_ok = False

    return inputs, outputs, op_hist, params, (macs if macs_ok else None)


def profile(model_path, runs):
    """onnxruntime 逐层耗时。返回 (avg_total_ms, top_layers)。"""
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = "/tmp/rvs_ort_prof"
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else (1 if i == 0 or len(inp.shape) != 4 else 640)
             for i, d in enumerate(inp.shape)]
    dummy = np.zeros(shape, np.float32)
    feed = {inp.name: dummy}

    for _ in range(5):                                       # warmup（避免首跑冷启动拉高均值）
        sess.run(None, feed)
    t0 = time.time()
    for _ in range(runs):
        sess.run(None, feed)
    avg_total = (time.time() - t0) * 1000 / runs

    prof_file = sess.end_profiling()
    agg = {}                                                 # name -> [op, total_us]
    with open(prof_file, encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents") if isinstance(data, dict) else data
    for ev in events or []:
        cat = str(ev.get("cat", ""))
        if "node" not in cat.lower() or "dur" not in ev:
            continue
        name = ev.get("name", "?")
        op = (ev.get("args") or {}).get("op_name", "?")
        if op.startswith("Reorder"):             # ORT NCHWC 内部布局重排，非模型算子
            continue
        name = name.removesuffix("_kernel_time")
        a = agg.setdefault(name, [op, 0])
        a[1] += ev["dur"]
    os.remove(prof_file)

    top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:15]
    layers = [{"name": k, "op": v[0], "ms": round(v[1] / 1000.0 / runs, 3)}
              for k, v in top]
    return avg_total, layers


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: onnx_analyze.py <model.onnx> [runs]"}))
        return
    model_path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    if not os.path.exists(model_path):
        print(json.dumps({"ok": False, "error": "model not found: " + model_path}))
        return
    file_mb = os.path.getsize(model_path) / 1048576.0

    params = macs = None
    inputs = outputs = []
    op_hist = {}
    try:
        inputs, outputs, op_hist, params, macs = static_info(model_path)
    except Exception as e:                                   # noqa: BLE001
        log("static analysis failed: %s" % e)

    try:
        avg_total, layers = profile(model_path, runs)
    except Exception as e:                                   # noqa: BLE001
        print(json.dumps({"ok": False, "error": "profile: %s" % e}))
        return

    print(json.dumps({
        "ok": True,
        "model": os.path.basename(model_path),
        "file_mb": round(file_mb, 2),
        "inputs": inputs, "outputs": outputs,
        "params": params,
        "macs": macs,
        "op_hist": [{"op": k, "count": v} for k, v in op_hist.most_common()],
        "runs": runs,
        "avg_total_ms": round(avg_total, 2),
        "top_layers": layers,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
