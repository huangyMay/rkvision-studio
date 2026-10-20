#!/bin/bash
# 板端(RK3576 aarch64/Android)交叉构建：NDK r18b + OpenCV(android 静态) + rknpu2
#
# 依赖外置（Rockchip SDK / 业务资产，不入库，须自行准备）：
#   export RVS_RKNN_SDK_DIR=~/.../rknpu2                 # rknn_api.h + librknnrt.so
#       （上游：https://github.com/airockchip/rknn-toolkit2 的 rknpu2/ 目录）
#   export RVS_OPENCV_ANDROID_DIR=~/.../opencv-android-sdk-build/sdk/native
#   export RVS_BASELINE_DEMO_DIR=~/.../mydemos/xxx       # 可选，M0 gate 基线
#   export ANDROID_NDK_ROOT=~/.../android-ndk-r18b       # bionic + c++_static
# 均可改用 cmake -D<VAR>=... 传入（优先级高于环境变量）。
set -e
ROOT=$(cd "$(dirname "$0")/.."; pwd)
ARGS=(-DRVS_TARGET=board -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/board.cmake")
for V in RVS_RKNN_SDK_DIR RVS_OPENCV_ANDROID_DIR RVS_BASELINE_DEMO_DIR; do
    [ -n "${!V}" ] && ARGS+=(-D${V}="${!V}")
done
cmake -S "$ROOT" -B "$ROOT/build-board" "${ARGS[@]}"
cmake --build "$ROOT/build-board" -j"$(nproc)"
file "$ROOT/build-board/engine/rvs_engine"
