#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 混合量化向导后端：敏感层保 fp16、其余 int8，构建后板上复测对比。

cfg 真实格式（step1 proposal=True 预填，2026-10-25 实测）:
    custom_quantize_layers:                        # 段1：提议层，行格式 "<onnx张量名>: float16  # 相似度分"
        /model.1/conv/Conv_output_0: float16
        ...
    quantize_parameters:                           # 段2：全模型量化参数（勿动）

两段式（Qt 分两次调用，中间给用户勾选层）:
  propose: hybrid_quant.py propose <onnx> <dataset> <dump_dir> <meanstd>
      → step1 → 解析 cfg 提议层（分值升序）→ 末行 JSON {ok, proposal:[{name,score}], files}
  build:   hybrid_quant.py build <cfg> <onnx> <image> <device|''> <meanstd>
                 <dump_dir> <layers 逗号分隔> [ref_iou]
      → 重写 cfg 勾选层 → step2 出混合 .rknn → 板上复测 vs ONNX → 末行 JSON
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_infer import norm_path  # noqa: E402


def log(msg):
    print("LOG " + msg, flush=True)


def tensor_cosine(a_list, b_list):
    """逐输出展平余弦（acc_compare 同款口径）。"""
    import numpy as np
    cos = []
    for x, y in zip(a_list, b_list or []):
        xf = np.asarray(x, dtype=np.float32).ravel()
        yf = np.asarray(y, dtype=np.float32).ravel()
        if xf.size != yf.size:
            continue
        denom = np.linalg.norm(xf) * np.linalg.norm(yf)
        cos.append(round(float(np.dot(xf, yf) / denom), 5) if denom > 0 else 1.0)
    return cos


def parse_cfg_layers(cfg_path):
    """cfg custom_quantize_layers 段 → [{name, score}]，保持文件顺序（=敏感度升序）。"""
    out = []
    in_sec = False
    for ln in open(cfg_path, encoding="utf-8", errors="ignore"):
        s = ln.strip()
        if s.startswith("custom_quantize_layers"):
            in_sec = True
            continue
        if s.startswith("quantize_parameters"):
            in_sec = False
            continue
        if in_sec and s.endswith(": float16"):
            name = s.split(": float16")[0].strip()
            score = None
            if "#" in s:
                try:
                    score = float(s.split("#")[1].strip())
                except ValueError:
                    pass
            out.append({"name": name, "score": score})
    return out


def write_cfg_selection(cfg_path, layers):
    """把勾选层写进 cfg（重写段1，段2 quantize_parameters 原样保留）。"""
    lines = open(cfg_path, encoding="utf-8", errors="ignore").read().splitlines()
    out, in_sec = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("custom_quantize_layers"):
            out.append("custom_quantize_layers:")
            for l in layers:
                out.append("    %s: float16" % l)
            in_sec = True
            continue
        if in_sec:
            if s.startswith("quantize_parameters"):
                in_sec = False
                out.append(ln)
                continue
            continue                    # 旧提议行丢弃
        out.append(ln)
    open(cfg_path, "w", encoding="utf-8").write("\n".join(out) + "\n")


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: hybrid_quant.py propose|build ..."}))
        return
    mode = sys.argv[1]

    # ---------- propose ----------
    if mode == "propose":
        if len(sys.argv) < 6:
            print(json.dumps({"ok": False, "error":
                              "usage: propose <onnx> <dataset> <dump_dir> <meanstd>"}))
            return
        onnx_path = norm_path(sys.argv[2])
        dataset = norm_path(sys.argv[3])
        dump_dir = norm_path(sys.argv[4])
        meanstd = sys.argv[5]
        from rknn_sim import parse_meanstd
        mean, std = parse_meanstd(meanstd.replace("/", "|"))
        from rknn.api import RKNN
        rknn = RKNN(verbose=False)
        kw = {"target_platform": "rk3576"}
        if mean is not None:
            kw["mean_values"] = mean
            kw["std_values"] = std
        if rknn.config(**kw) != 0:
            print(json.dumps({"ok": False, "error": "config failed"})); return
        if rknn.load_onnx(model=onnx_path) != 0:
            print(json.dumps({"ok": False, "error": "load_onnx failed"})); return
        log("混合量化 step1（校准 + toolkit 敏感层提议）…")
        hq = os.path.join(dump_dir, "hq")
        os.makedirs(hq, exist_ok=True)
        os.chdir(hq)
        if rknn.hybrid_quantization_step1(dataset=dataset, proposal=True) != 0:
            print(json.dumps({"ok": False, "error": "step1 failed"})); return
        base = os.path.splitext(os.path.basename(onnx_path))[0]
        files = {"model": os.path.join(hq, base + ".model"),
                 "data": os.path.join(hq, base + ".data"),
                 "cfg": os.path.join(hq, base + ".quantization.cfg")}
        proposal = parse_cfg_layers(files["cfg"])
        log(f"step1 完成：toolkit 提议 {len(proposal)} 层（敏感度升序；默认勾选分值<0.999 的）")
        print(json.dumps({"ok": True, "proposal": proposal, "files": files,
                          "n": len(proposal)}, ensure_ascii=False))
        return

    # ---------- build ----------
    if mode == "build":
        if len(sys.argv) < 9:
            print(json.dumps({"ok": False, "error":
                              "usage: build <cfg> <onnx> <image> <device|''> <meanstd> "
                              "<dump_dir> <layers|-> [ref_iou]"}))
            return
        cfg = norm_path(sys.argv[2])
        onnx_path = norm_path(sys.argv[3])
        image_path = norm_path(sys.argv[4])
        device = sys.argv[5]
        meanstd = sys.argv[6]
        dump_dir = norm_path(sys.argv[7])
        layers_arg = sys.argv[8]
        ref_iou = float(sys.argv[9]) if len(sys.argv) > 9 else None

        import acc_compare as ac
        import cv2
        import numpy as np

        layers = [s.strip() for s in layers_arg.split(",") if s.strip()]
        if not layers:
            print(json.dumps({"ok": False, "error": "未勾选任何层——至少保 1 层 fp16 再开工"}))
            return
        log(f"混合量化开工：保 fp16 的层 {len(layers)} 个")
        for l in layers[:10]:
            log(f"  · {l}")
        if len(layers) > 10:
            log(f"  …共 {len(layers)} 个")
        write_cfg_selection(cfg, layers)

        from rknn.api import RKNN
        rknn = RKNN(verbose=False)
        base = os.path.splitext(os.path.basename(onnx_path))[0]
        hq = os.path.dirname(cfg)
        model_in = os.path.join(hq, base + ".model")
        data_in = os.path.join(hq, base + ".data")
        log("step2 构建（量化重排）…")
        if rknn.hybrid_quantization_step2(model_input=model_in, data_input=data_in,
                                          model_quantization_cfg=cfg) != 0:
            print(json.dumps({"ok": False, "error": "step2 failed"})); return
        # step2 只导到内部临时目录（verbose 实测 /tmp/tmpXXX/check.rknn），须 export_rknn 落盘
        hybrid_rknn = os.path.join(hq, base + "_hybrid.rknn")
        if rknn.export_rknn(hybrid_rknn) != 0:
            print(json.dumps({"ok": False, "error": "export_rknn failed"})); return
        log(f"混合模型已生成: {hybrid_rknn}")
        try:
            rknn.release()
        except Exception:
            pass

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(json.dumps({"ok": False, "error": f"cannot read image: {image_path}"})); return
        import onnxruntime as ort
        so = ort.SessionOptions(); so.log_severity_level = 3
        sess0 = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        inp = sess0.get_inputs()[0]
        dims = [d for d in inp.shape if isinstance(d, int) and d > 1]
        input_size = dims[-1] if len(dims) >= 2 and dims[-1] in (320, 416, 512, 640, 960, 1280) else 640
        lb, scale, left, top = ac.letterbox(img, input_size)
        blob = cv2.dnn.blobFromImage(lb, 1.0 / 255.0, swapRB=True)
        outs1 = sess0.run(None, {inp.name: blob})
        d1, _ = ac.decode(outs1, input_size, scale, left, top, 0.25, 0.45)
        base_dets = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in (d1 or [])]
        log(f"① 基线复测: 检出 {len(base_dets)} 框")

        report = {"ok": True, "hybrid_rknn": hybrid_rknn,
                  "n_fp16_layers": len(layers), "fp16_layers": layers,
                  "ref_iou": ref_iou, "board": {}, "match": {}}
        if device:
            from rknn_acc import restart_board_server
            log("板复测前自愈重启 rknn_server…")
            restart_board_server(device)
            rk2 = RKNN(verbose=False)
            if rk2.load_rknn(hybrid_rknn) != 0:
                print(json.dumps({"ok": False, "error": "load_rknn(hybrid) failed"})); return
            if rk2.init_runtime(target="rk3576", device_id=device) != 0:
                print(json.dumps({"ok": False, "error": f"init_runtime(board {device}) failed"}))
                return
            nhwc = np.ascontiguousarray(blob.transpose(0, 2, 3, 1))
            if meanstd.replace("/", "|") != "none|none":
                nhwc = np.ascontiguousarray(nhwc * 255.0)   # mean/std 进图内：喂原始 0~255
            log("混合模型板上推理…")
            outs = rk2.inference(inputs=[nhwc])
            if outs is None:
                print(json.dumps({"ok": False, "error": "hybrid board inference None"})); return
            dets, _ = ac.decode(outs, input_size, scale, left, top, 0.25, 0.45)
            bb = [[d[0], d[1], d[2], d[3], d[4]] for d in (dets or [])]
            m = ac.match_report(base_dets, bb, "混合量化 vs ONNX")
            report["board"] = {"n": len(bb)}
            report["match"] = m
            report["tensor_cosine"] = tensor_cosine(outs1, outs)
            log(f"混合(int8+fp16敏感层) 板上: 检出 {len(bb)} 框 · mean IoU "
                f"{m.get('mean_iou', 0)}")
            try:
                rk2.release()
            except Exception:
                pass
        else:
            log("未填板地址——跳过板复测（混合 .rknn 已生成）")

        print(json.dumps(report, ensure_ascii=False))
        return

    print(json.dumps({"ok": False, "error": f"unknown mode: {mode}"}))


if __name__ == "__main__":
    main()
