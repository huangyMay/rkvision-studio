#!/bin/bash
# 启动 RKVision Studio（WSL 内脱管；adb 走 Windows 侧 interop）
PORT=${1:-8800}
export RVS_ADB="/mnt/d/Tools/ADBGJB_V1.0.41_XiTongZhiJia/adb/adb.exe"
cd "$(dirname "$0")/../studio"
exec python3 rvs/studio_server.py "$PORT"
