#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RKNN 板端性能测试（toolkit2 eval_perf，经 adb 连板、NPU 真跑）。

前提（一次性）:
  1. 板上 /data/local/tmp/{rknn_server, librknnrt.so}（2.3.2 配套版）已推、rknn_server 已启动
  2. WSL ~/bin/adb wrapper（共用 Windows adb server）且 ~/bin 在 PATH

用法:
  python3 rknn_perf.py <model.onnx> <device_id> [meanstd] [frame_loops]

输出 JSON: {"ok":true,"total_ms":..,"avg_ms":..,"layers":[{name,op,us}...],"raw":...}
逐层耗时来自 eval_perf 的 NPU 真实测——这才是"rknn 的性能"。
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rknn_sim import parse_meanstd  # noqa: E402
from onnx_infer import norm_path  # noqa: E402

# adb wrapper 所在目录（共用 Windows adb server 的 ~/bin/adb）；可用 RVS_BOARD_BIN 覆盖
HOME_BIN = os.environ.get("RVS_BOARD_BIN") or os.path.expanduser("~/bin")


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"ok": False, "error": "usage: rknn_perf.py <model.onnx> <device_id> [meanstd] [frame_loops]"}))
        return
    onnx_path = norm_path(sys.argv[1])
    device_id = sys.argv[2]
    meanstd = sys.argv[3] if len(sys.argv) > 3 else "0,0,0/255,255,255"
    frame_loops = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    os.environ["PATH"] = HOME_BIN + os.pathsep + os.environ.get("PATH", "")

    mean, std = parse_meanstd(meanstd)
    from rknn.api import RKNN
    rknn = RKNN(verbose=True)
    try:
        if mean is not None:
            # target_platform 是 config 必填项；与 adb 探测结果一致性取决于
            # 板上 rknn_server 版本与 toolkit2 匹配（必须是 2.3.2 配对版）
            if rknn.config(mean_values=mean, std_values=std, target_platform="rk3576") != 0:
                raise RuntimeError("config failed")
        if rknn.load_onnx(model=onnx_path) != 0:
            raise RuntimeError("load_onnx failed")
        if rknn.build(do_quantization=False) != 0:
            raise RuntimeError("build failed")

        # C 扩展日志直写 fd1——重定向到文件再解析（python 级 redirect_stdout 抓不到）
        tmpf = "/tmp/rknn_perf_out.txt"
        saved = os.dup(1)
        fd = os.open(tmpf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(fd, 1)
        try:
            if rknn.init_runtime(target="rk3576", device_id=device_id) != 0:
                raise RuntimeError("init_runtime(target=adb) failed——检查板上 rknn_server 与 adb")
            res = rknn.eval_perf(is_print=True, fix_freq=False)
        finally:
            os.dup2(saved, 1)
            os.close(fd)
    except Exception as e:                                   # noqa: BLE001
        print(json.dumps({"ok": False, "error": "perf: %s" % e}))
        return
    finally:
        try:
            rknn.release()
        except Exception:                                    # noqa: BLE001
            pass

    perf_text = open(tmpf, encoding="utf-8", errors="replace").read()

    # 解析 perf 表：ID ... OpsType ... Time(us)；以及 Total Time
    layers = []
    total_us = None
    fps = None
    for ln in perf_text.splitlines():
        m = re.search(r"Total Time\(us\):\s*([0-9.]+)", ln)
        if m:
            total_us = float(m.group(1))
        mf = re.search(r"FPS:\s*([0-9.]+)", ln)
        if mf:
            fps = float(mf.group(1))
        mm = re.match(
            r"\s*(\d+)\s+\S+\s+\S+\s+(\S+)\s+(\S+)\s+([0-9.]+)\s+([0-9.]+)%?\s*$", ln)
        if mm and "----" not in ln:
            layers.append({"idx": int(mm.group(1)), "op": mm.group(3),
                           "backend": mm.group(4), "us": float(mm.group(5))})

    avg_ms = round(total_us / 1000.0, 2) if total_us is not None else None

    print(json.dumps({
        "ok": True,
        "model": os.path.basename(onnx_path),
        "device": device_id,
        "frame_loops": frame_loops,
        "total_us": total_us,
        "fps": fps,
        "avg_ms": avg_ms,
        "layers": layers[:40],
        "layer_count": len(layers),
        "raw": perf_text[-2000:],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
