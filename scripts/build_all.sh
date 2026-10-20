#!/bin/bash
# 全量构建 + 测试，输出落文件（规避 wsl.exe 管道挂死）
set -e
ROOT=$(cd "$(dirname "$0")/.."; pwd)
cd "$ROOT"
bash scripts/build_host.sh > build/host_build.log 2>&1
ctest --test-dir build-host > build/host_test.log 2>&1
grep -E "tests passed|tests failed" build/host_test.log || true
bash scripts/build_board.sh > build/board_build.log 2>&1
tail -1 build/board_build.log
echo ALL_DONE
