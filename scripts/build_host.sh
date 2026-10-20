#!/bin/bash
# host(WSL x86) 构建 + ctest：单元测试主战场
set -e
ROOT=$(cd "$(dirname "$0")/.."; pwd)
cmake -S "$ROOT" -B "$ROOT/build-host" -DRVS_TARGET=host
cmake --build "$ROOT/build-host" -j"$(nproc)" || {
    echo "hint: sudo apt install -y libopencv-dev"
    exit 1
}
