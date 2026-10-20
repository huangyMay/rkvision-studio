"""部署与板上执行。

板上布局（remote_root 默认 /data/local/tmp/rvs）：
    <remote_root>/rvs_engine          引擎（aarch64 ELF）
    <remote_root>/librknnrt.so        toolkit2 配套 runtime
    <remote_root>/<name>/             工程包（pipeline.json/models/assets）
    <remote_root>/<name>/output/      结果（拉回本地）
"""
import os
from pathlib import Path

from .pack import pack


def deploy(project_dir, adb, remote_root="/data/local/tmp/rvs",
           engine_bin=None, librknnrt=None):
    project_dir = Path(project_dir)
    engine_bin = Path(engine_bin) if engine_bin else _default_engine()
    librknnrt = Path(librknnrt) if librknnrt else _default_librknnrt()

    staging = pack(project_dir)
    name = staging.name

    adb.push(str(staging), f"{remote_root}/{name}")
    adb.push(str(engine_bin), f"{remote_root}/rvs_engine")
    adb.push(str(librknnrt), f"{remote_root}/librknnrt.so")
    adb.shell(f"chmod +x {remote_root}/rvs_engine")
    return staging


def run_on_board(project_dir, adb, remote_root="/data/local/tmp/rvs",
                 engine_bin=None, librknnrt=None, timeout_s=600):
    deploy(project_dir, adb, remote_root, engine_bin, librknnrt)
    project_dir = Path(project_dir)
    spec_name = project_dir.name

    run_cmd = (f"cd {remote_root}/{spec_name} && "
               f"LD_LIBRARY_PATH={remote_root} {remote_root}/rvs_engine .")
    summary = adb.shell(run_cmd)

    local_out = project_dir / "output"
    adb.pull(f"{remote_root}/{spec_name}/output", str(local_out.parent / spec_name))
    return summary


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
