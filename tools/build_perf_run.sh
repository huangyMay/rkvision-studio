#!/bin/bash
# perf_run 交叉编译（Android arm64，NDK r27c；运行时用板上配对 librknnrt，编译期只借符号）
set -e
NDK=/home/hmorey/ndk_dir/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin
BASE="$(cd "$(dirname "$0")" && pwd)"
INC2X="$BASE/../engine/third_party"                       # 2.x 配套头（rvs_engine 同款；老 rknpu2 头 ABI 不配 2.x 库）
LINKAPI="$HOME/android/rk/16-rknn开源参考/rknpu2/runtime/RK356X/Android/librknn_api"
mkdir -p "$BASE/bin"
"$NDK/aarch64-linux-android28-clang" -O2 -Wall -o "$BASE/bin/perf_run_android_arm64" \
    "$BASE/perf_run.c" -I"$INC2X" -L"$LINKAPI/arm64-v8a" -lrknnrt
ls -la "$BASE/bin/perf_run_android_arm64"
