# RKVision Studio

[![ci](https://github.com/huangyMay/rkvision-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/huangyMay/rkvision-studio/actions/workflows/ci.yml)

English | [简体中文](README.md)

> One workbench for the whole edge-AI chain: model → RKNN → on-board NPU → live video.

**RKVision Studio** is an open-source edge-vision workbench for Rockchip RK3576/RK356x NPUs: a Qt desktop app on the PC (live preview + ONNX LAB + RKNN LAB) + a plugin-based C++ inference engine on the board + an Android camera-streaming app. It covers the full path from model tuning to on-board delivery.

**If any of these sound familiar, this is for you:**

- Tweaking an rknn_model_zoo demo — changing one output head means re-reading the whole C++ postprocess;
- Juggling toolkit2 CLI, adb, and hand-written verification scripts, with no way to match results before/after conversion;
- Changing one line of postprocessing and having to verify detection boxes by eye.

**What you get:**

- **Declarative `pipeline.json`**: source → preprocess → infer → post → sink. Swap models or post-processing with zero code; a new operator is just one `REGISTER_STAGE` file.
- **Dual-LAB comparison**: same image, same thresholds, ONNX (onnxruntime) vs RKNN (toolkit2 simulator) side by side — box-by-box numerical match or it doesn't count. RKNN LAB's by-product is a flashable `.rknn`.
- **Accuracy analysis center**: ⚖️ one-click 7-channel comparison (ONNX → fp16/int8/int16 in simulator → fp16/int8/int16 on board, with per-layer cosine + annotated images), 🧬 hybrid-quantization wizard (keep sensitive layers in fp16), 🚀 on-board perf profiling (core_mask matrix + devfreq lock + set/run/get staged waterfall), 🔪 offline `.rknn` container dissection.
- **Bit-exact golden regression**: tensor-level comparison of on-board outputs (gold_regression) — whether a postprocessing change altered results is decided by the machine.
- **One-command deploy loop**: cross-compile → adb push → run on board → pull results back.

| ONNX LAB — local inference + thresholds + model analysis | RKNN LAB — toolkit2 simulated inference + one-click conversion |
|---|---|
| ![ONNX LAB](docs/images/onnx_lab.png) | ![RKNN LAB](docs/images/rknn_lab.png) |

```
┌────────────────────────── PC (Windows + WSL) ──────────────────────────┐
│  Qt Studio (rvs_studio.exe)                                            │
│  ├─ LIVE       board preview (MJPEG direct / RTSP) + live infer        │
│  ├─ ONNX LAB   local image × local ONNX: thresholds / model analysis   │
│  ├─ RKNN LAB   toolkit2 simulator + layer probe + one-click .rknn      │
│  └─ Deploy     one-command cross-build + push + run + fetch results    │
│        │ QProcess(u32LE JSON frames)      │ adb push/reverse           │
│  ┌──────▼────────────────┐        ┌──────▼──────────────────────┐     │
│  │ WSL python3 toolchain │        │  mediamtx (RTSP)  ◄─ rvs-cam │     │
│  │ onnxruntime / rknn    │        └─────────────────────────────┘     │
│  └───────────────────────┘                │ RTSP/MJPEG                 │
└───────────────────────────────────────────┼────────────────────────────┘
                                            ▼
┌───────────────────── Board (RK3576, Android) ──────────────────────────┐
│  rvs_engine — plugin-based pipeline engine                             │
│  source(v4l2/image_dir) → preprocess(letterbox) → infer(rknn)          │
│                          → post(yolov5/yolov8) → sink(image_out/mjpeg) │
│  pipeline.json declares topology & params; plugins registered at       │
│  compile time (no dlopen) — assemble projects with zero code           │
└────────────────────────────────────────────────────────────────────────┘
```

## Repository layout

```
engine/        on-board C++ engine (core pipeline + stage plugins)
studio/        Python package: packaging / deploy / board-side comparison
studio-qt/     PC-side Qt 6 desktop workbench
tools/         WSL-side inference / analysis / perf scripts (onnxruntime + rknn-toolkit2)
app-rvscam/    Android camera-streaming app (MJPEG → PC/mediamtx)
projects/      sample project packages (pipeline.json + assets)
scripts/       build & verification scripts
cmake/         board cross-compilation toolchain wrappers
docs/          design docs, build & verification manual, workflow design (Chinese)
```

## Build

### 1) PC-side engine + unit tests (Linux/WSL)

```bash
sudo apt install cmake g++ libopencv-dev
./scripts/build_host.sh && ctest --test-dir build-host --output-on-failure
```

### 2) Board engine (cross-compile, RK3576 aarch64/Android)

The Rockchip SDK and OpenCV Android libs are **not** distributed with this repo — provide them via env vars / `-D`:

```bash
export ANDROID_NDK_ROOT=~/android-ndk-r18b                    # NDK (r18b verified)
export RVS_RKNN_SDK_DIR=~/rknn-toolkit2/rknpu2                # upstream: github.com/airockchip/rknn-toolkit2
export RVS_OPENCV_ANDROID_DIR=~/opencv-android-sdk-build/sdk/native
# optional (M0 baseline demo): export RVS_BASELINE_DEMO_DIR=~/mydemos/algorithm-helmet
./scripts/build_board.sh
```

### 3) PC Qt workbench (Windows + Qt 6.7)

```bash
cmake -S studio-qt -B build-qt -DCMAKE_PREFIX_PATH=C:/Qt/6.7.3/mingw_64 -G "MinGW Makefiles"
cmake --build build-qt -j
```

Note: the Qt workbench invokes the python scripts in `tools/` through `wsl.exe`; `RVS_TOOLS_DIR` / `RVS_WSL_DISTRO` are injected at CMake configure time — no source edits needed.

### 4) Android streaming app

```bash
cd app-rvscam && ./gradlew assembleDebug
```

## Models & assets

The repo contains **no** model files or business assets (`.rknn` / golden tensors / test images are not committed). Use your own ONNX in ONNX LAB, or convert to a board `.rknn` in RKNN LAB with one click. `scripts/fetch_gold.sh.example` is a template for golden fixtures.

## Third-party components

| Component | License | Purpose |
|---|---|---|
| [rknn-toolkit2 / rknpu2](https://github.com/airockchip/rknn-toolkit2) | proprietary (Rockchip) | board runtime & conversion tools, **not distributed here** — get from upstream |
| [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) samples | Apache-2.0 | `postprocess_yolov5` references its implementation |
| [OpenCV](https://opencv.org) (≥4.5) | Apache-2.0 | image processing (host dynamic / Android static) |
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | `pipeline.json` parsing (single header vendored) |
| [onnxruntime](https://onnxruntime.ai) | MIT | PC-side ONNX inference (pip, WSL side) |
| Qt 6 | LGPLv3/GPL | desktop workbench |

## Docs

Deep-dive documents are written in Chinese (screenshots and code speak for themselves):

- [一帧的旅程-推理导学](docs/2026-10-21-一帧的旅程-推理导学.md) — **learning entry**: follow one real frame through infer → decode → decision (real numbers at every step)
- [框架设计](docs/2026-10-16-框架设计.md) — architecture & plugin mechanism
- [构建与验证手册](docs/2026-10-17-构建与验证手册.md) — from zero to running on board
- [工业工作流设计](docs/2026-10-18-工业工作流设计.md) — industrial vision templates / decision DSL / roadmap
- [性能剖析页与模型解剖](docs/2026-10-25-性能剖析页与模型解剖.md) — staged NPU timing (set/run/get), core-matrix, .rknn container layout

## License

Apache-2.0 (see [LICENSE](LICENSE)). Rockchip SDK (rknpu2 runtime, toolkit2) is proprietary; obtain it from upstream and comply with its license.
