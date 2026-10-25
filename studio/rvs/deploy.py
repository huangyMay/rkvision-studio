"""部署与板上执行。

板上布局（remote_root 默认 /data/local/tmp/rvs）：
    <remote_root>/rvs_engine          引擎（aarch64 ELF）
    <remote_root>/librknnrt.so        toolkit2 配套 runtime
    <remote_root>/<name>/             工程包（pipeline.json/models/assets）
    <remote_root>/<name>/output/      结果（拉回本地）
"""
import os
import shlex
from pathlib import Path

from .pack import pack


def deploy(project_dir, adb, remote_root="/data/local/tmp/rvs",
           engine_bin=None, librknnrt=None):
    project_dir = Path(project_dir)
    engine_bin = Path(engine_bin) if engine_bin else _default_engine()
    librknnrt = Path(librknnrt) if librknnrt else _default_librknnrt()
    if not engine_bin.is_file():
        raise RuntimeError(f"未找到板端引擎: {engine_bin}（先运行 scripts/build_board.sh）")
    if not librknnrt.is_file():
        raise RuntimeError(f"未找到 librknnrt.so: {librknnrt}")

    staging = pack(project_dir)
    name = staging.name
    q = _q(adb)

    adb.push(str(staging), f"{remote_root}/{name}")
    adb.push(str(engine_bin), f"{remote_root}/rvs_engine")
    adb.push(str(librknnrt), f"{remote_root}/librknnrt.so")
    adb.shell(f"chmod +x {q(remote_root)}/rvs_engine")
    return staging


def run_on_board(project_dir, adb, remote_root="/data/local/tmp/rvs",
                 engine_bin=None, librknnrt=None, timeout_s=600):
    project_dir = Path(project_dir)
    # 用 pack 实际生成的目录名，而不是工程目录名：spec.name 与目录名可以不同
    staging = deploy(project_dir, adb, remote_root, engine_bin, librknnrt)
    name = staging.name
    remote_project = f"{remote_root}/{name}"
    q = _q(adb)

    run_cmd = (f"cd {q(remote_project)} && "
               f"LD_LIBRARY_PATH={q(remote_root)} {q(remote_root + '/rvs_engine')} .")
    # timeout_s 之前是死参数：板上整轮推理可能远超 AdbHost 默认 30s
    prev_timeout = getattr(adb, "timeout_s", None)
    if prev_timeout is not None:
        adb.timeout_s = timeout_s
    try:
        summary = adb.shell(run_cmd)
    finally:
        if prev_timeout is not None:
            adb.timeout_s = prev_timeout

    # pull 到已存在的工程目录 → 落地为 <project_dir>/output/
    adb.pull(f"{remote_project}/output", str(project_dir))
    return summary


def _q(adb):
    """优先用 AdbHost.quote（shlex）；保留自定义 adb 实现的兼容性。"""
    quote = getattr(adb, "quote", None)
    return quote if callable(quote) else shlex.quote


def _default_engine():
    root = Path(__file__).resolve().parents[2]
    return root / "build-board" / "engine" / "rvs_engine"


def _default_librknnrt():
    env = os.environ.get("RVS_LIBRKNNRT")
    if env:
        return Path(env)
    sdk = os.environ.get("RVS_RKNN_SDK_DIR")
    if sdk:
        return Path(sdk) / "runtime/Android/librknn_api/arm64-v8a/librknnrt.so"
    raise RuntimeError(
        "未找到 librknnrt.so：请 export RVS_LIBRKNNRT=<so 路径> "
        "或 export RVS_RKNN_SDK_DIR=<rknpu2 根目录>")
