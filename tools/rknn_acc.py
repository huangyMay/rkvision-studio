#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RKNN 量化精度分析（toolkit2 accuracy_analysis，真板执行——模拟器跑不了）。

官方正确流程（docs.bit-brick.com / Rockchip 文档）：load_rknn 之后不能调
accuracy_analysis，必须从 ONNX 走量化构建：
  config → load_onnx → build(do_quantization=True, dataset) → init_runtime(target, device_id)
  → accuracy_analysis(inputs, output_dir, target, device_id)

前提（一次性）:
  1. 板上 /data/local/tmp/{rknn_server, librknnrt.so}（2.3.2 配套）已推、rknn_server 已启动
  2. adb 可达：USB（udev 规则放行后 WSL adb 直见）或网络（IP:5555）
  3. dataset.txt：校准图路径列表（每行一个），至少一张

用法:
  python3 rknn_acc.py <model.onnx> <dataset.txt> <device> [meanstd] [out_dir]
    meanstd 默认 "0,0,0/255,255,255"；out_dir 默认 /tmp/rknn_acc_out

输出 JSON: {"ok":true,"out_dir":..,"layers":[{name,cosine}...],"n_layers":N}
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rknn_sim import parse_meanstd  # noqa: E402
from onnx_infer import norm_path  # noqa: E402


def adb_path_prepend():
    """PATH 前插 toolkit2 自带 Linux adb（与 rknn 包同源，网络/USB 通吃）。"""
    try:
        import rknn  # noqa: F401
        private = os.path.join(os.path.dirname(os.path.dirname(rknn.__file__)),
                               "3rdparty", "platform-tools", "adb", "linux-x86_64")
        if os.path.isdir(private):
            os.environ["PATH"] = private + os.pathsep + os.environ.get("PATH", "")
            return
    except Exception:
        pass
    home_bin = os.environ.get("RVS_BOARD_BIN") or os.path.expanduser("~/bin")
    os.environ["PATH"] = home_bin + os.pathsep + os.environ.get("PATH", "")


def start_keepalive(device):
    """网络设备（IP:端口）时启动保活守护线程：accuracy_analysis 内部 adb root
    会重启 adbd、切断 TCP 传输，导致工具挂死；每 3s 重连+保 root 即可预防。"""
    if ":" not in device:
        return
    import shutil
    import threading
    import subprocess
    adb = shutil.which("adb")
    if not adb:
        return

    def loop():
        while True:
            subprocess.run([adb, "connect", device],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run([adb, "root"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            threading.Event().wait(3)

    threading.Thread(target=loop, daemon=True).start()


def parse_error_analysis(path):
    """error_analysis.txt → [{name, sim_entire_cos, sim_single_cos, rt_entire_cos, rt_single_cos}]。
    真实列序（2.3.2）：sim_entire(cos|euc) sim_single(cos|euc) runtime_entire(cos|euc) single_sim(cos|euc)。
    runtime 侧仅对部分层输出；无数据的行（如 [Conv] getitem）跳过。"""
    layers = []
    if not os.path.exists(path):
        return layers
    num = re.compile(r"-?\d+\.\d+")
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip()
        if not line or line.startswith("#") or set(line) <= {"-", " "}:
            continue
        if "layer_name" in line or "cos " in line[:40] and "euc" in line[:40]:
            continue
        floats = num.findall(line)
        if not floats:
            continue
        name = line.split(floats[0])[0].strip()
        if not name:
            continue
        d = {"name": name,
             "sim_entire_cos": float(floats[0])}
        if len(floats) >= 3:
            d["sim_single_cos"] = float(floats[2])
        if len(floats) >= 7:
            d["rt_entire_cos"] = float(floats[4])
            d["rt_single_cos"] = float(floats[6])
        layers.append(d)
    return layers


def main():
    if len(sys.argv) < 4:
        print(json.dumps({"ok": False,
                          "error": "usage: rknn_acc.py <model.onnx> <dataset.txt> <device> [meanstd] [out_dir]"}))
        return
    onnx_path = norm_path(sys.argv[1])
    dataset = norm_path(sys.argv[2])
    device = sys.argv[3]
    meanstd = sys.argv[4] if len(sys.argv) > 4 else "0,0,0/255,255,255"
    out_dir = norm_path(sys.argv[5]) if len(sys.argv) > 5 else "/tmp/rknn_acc_out"

    adb_path_prepend()
    start_keepalive(device)
    mean, std = parse_meanstd(meanstd)

    from rknn.api import RKNN
    rknn = RKNN(verbose=False)

    if rknn.config(mean_values=mean, std_values=std,
                   target_platform="rk3576") != 0:
        print(json.dumps({"ok": False, "error": "config failed"}))
        return
    if rknn.load_onnx(model=onnx_path) != 0:
        print(json.dumps({"ok": False, "error": f"load_onnx failed: {onnx_path}"}))
        return
    if rknn.build(do_quantization=True, dataset=dataset) != 0:
        print(json.dumps({"ok": False, "error": "build(do_quantization) failed——查 dataset.txt 图片路径"}))
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
    inputs = [l.strip() for l in open(dataset, encoding="utf-8") if l.strip()]
    inputs = [norm_path(p) for p in inputs if os.path.exists(norm_path(p))]
    if not inputs:
        print(json.dumps({"ok": False, "error": "dataset 图片全部不存在"}))
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
