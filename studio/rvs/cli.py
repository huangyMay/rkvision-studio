"""rvs 命令行入口：pack / deploy / run（compare、convert 见后续任务）。"""
import argparse
import sys
from pathlib import Path

from .adb import AdbHost
from .deploy import deploy, run_on_board
from .pack import pack


def _make_adb(args):
    return AdbHost(bin_path=args.adb_bin, serial=args.serial)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rvs", description="RKVision Studio M0 CLI")
    parser.add_argument("--adb-bin", default=None, help="adb 可执行路径（默认 $RVS_ADB 或 adb）")
    parser.add_argument("--serial", default=None, help="adb 设备序列号")
    parser.add_argument("--remote-root", default="/data/local/tmp/rvs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pack = sub.add_parser("pack", help="打包工程包到 build-staging/")
    p_pack.add_argument("--project", required=True)

    p_deploy = sub.add_parser("deploy", help="打包并推板（含引擎与 librknnrt）")
    p_deploy.add_argument("--project", required=True)
    p_deploy.add_argument("--engine", default=None)
    p_deploy.add_argument("--librknnrt", default=None)

    p_run = sub.add_parser("run", help="deploy + 板上执行 + 拉回结果")
    p_run.add_argument("--project", required=True)
    p_run.add_argument("--engine", default=None)
    p_run.add_argument("--librknnrt", default=None)
    p_run.add_argument("--timeout", type=int, default=600)

    p_conv = sub.add_parser("convert", help="生成（并可选执行）yolov5 ONNX->RKNN 转换脚本")
    p_conv.add_argument("--onnx", required=True)
    p_conv.add_argument("--out", required=True, help="输出 .rknn 路径")
    p_conv.add_argument("--dataset", default=None, help="量化校准集目录/列表")
    p_conv.add_argument("--target", default="rk3576")
    p_conv.add_argument("--dry-run", action="store_true", default=True)
    p_conv.add_argument("--toolkit2-python", default=None)

    args = parser.parse_args(argv)
    adb = _make_adb(args)

    if args.cmd == "pack":
        staging = pack(Path(args.project))
        print(f"staged: {staging}")
        return 0
    if args.cmd == "deploy":
        staging = deploy(Path(args.project), adb, args.remote_root, args.engine, args.librknnrt)
        print(f"deployed: {staging.name} -> {args.remote_root}")
        return 0
    if args.cmd == "run":
        summary = run_on_board(Path(args.project), adb, args.remote_root,
                               args.engine, args.librknnrt, timeout_s=args.timeout)
        print(summary)
        return 0
    if args.cmd == "convert":
        from .convert import cmd_convert
        return cmd_convert(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
