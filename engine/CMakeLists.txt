include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

set(ENGINE_TARGET rvs_engine)

if(RVS_TARGET STREQUAL "board")
    # ---- 第三方依赖外置（Rockchip SDK 专有许可 / 业务资产，不随仓库分发）----
    # 本地开发：export 环境变量（见 scripts/build_board.sh 头注释）；其他机器：cmake -D<VAR>=... 覆盖
    foreach(V RVS_RKNN_SDK_DIR RVS_OPENCV_ANDROID_DIR RVS_BASELINE_DEMO_DIR)
        if(NOT ${V} AND DEFINED ENV{${V}})
            set(${V} "$ENV{${V}}")
        endif()
    endforeach()

    if(NOT RVS_RKNN_SDK_DIR)
        message(FATAL_ERROR
            "板端构建需要 Rockchip rknpu2 SDK（rknn_api.h + librknnrt.so，专有许可，不随本仓库分发）\n"
            "  cmake -DRVS_RKNN_SDK_DIR=/path/to/rknpu2\n"
            "  上游公开副本：https://github.com/airockchip/rknn-toolkit2 （rknpu2/ 目录）")
    endif()
    # rknn_api.h 位置（默认取 rknpu2 官方布局；不同时 -DRVS_RKNN_INC=<dir> 覆盖）
    if(NOT RVS_RKNN_INC)
        set(RVS_RKNN_INC "${RVS_RKNN_SDK_DIR}/runtime/Android/librknn_api/include")
    endif()
    if(NOT EXISTS "${RVS_RKNN_INC}/rknn_api.h")
        message(FATAL_ERROR "未找到 rknn_api.h: ${RVS_RKNN_INC} ，请用 -DRVS_RKNN_INC=<rknn_api.h 所在目录> 指定")
    endif()
    set(RVS_RKNNRT "${RVS_RKNN_SDK_DIR}/runtime/Android/librknn_api/arm64-v8a/librknnrt.so")
    if(NOT EXISTS "${RVS_RKNNRT}")
        message(FATAL_ERROR "未找到 librknnrt.so: ${RVS_RKNNRT}")
    endif()

    if(NOT RVS_OPENCV_ANDROID_DIR)
        message(FATAL_ERROR "板端构建需要 OpenCV Android 静态库：cmake -DRVS_OPENCV_ANDROID_DIR=<.../opencv-android-sdk-build/sdk/native>")
    endif()
    set(OpenCV_ANDROID "${RVS_OPENCV_ANDROID_DIR}")
    if(NOT EXISTS "${OpenCV_ANDROID}/staticlibs/arm64-v8a/libopencv_core.a")
        message(FATAL_ERROR "OpenCV android 静态库缺失: ${OpenCV_ANDROID}")
    endif()
    set(RVS_OPENCV_INC ${OpenCV_ANDROID}/jni/include)
    set(RVS_3P_LIBS ${OpenCV_ANDROID}/3rdparty/libs/arm64-v8a)
    set(RVS_OPENCV_LIBS
        ${OpenCV_ANDROID}/staticlibs/arm64-v8a/libopencv_imgcodecs.a
        ${OpenCV_ANDROID}/staticlibs/arm64-v8a/libopencv_imgproc.a
        ${OpenCV_ANDROID}/staticlibs/arm64-v8a/libopencv_core.a
        ${RVS_3P_LIBS}/libtegra_hal.a
        ${RVS_3P_LIBS}/libIlmImf.a
        ${RVS_3P_LIBS}/liblibjasper.a
        ${RVS_3P_LIBS}/liblibtiff.a
        ${RVS_3P_LIBS}/liblibwebp.a
        ${RVS_3P_LIBS}/liblibjpeg-turbo.a
        ${RVS_3P_LIBS}/liblibpng.a
        ${RVS_3P_LIBS}/libzlib.a
        ${RVS_3P_LIBS}/libcpufeatures.a)
else()
    find_package(OpenCV REQUIRED)
    set(RVS_OPENCV_INC ${OpenCV_INCLUDE_DIRS})
    set(RVS_OPENCV_LIBS ${OpenCV_LIBS})
endif()

set(ENGINE_SOURCES
    core/spec.cpp
    core/registry.cpp
    core/assembler.cpp
    core/executor.cpp
    core/run_project.cpp
    stages/source_image_dir.cpp
    stages/source_v4l2.cpp
    stages/preprocess_letterbox.cpp
    stages/postprocess_yolov5.cpp
    stages/postprocess_yolov8.cpp
    stages/judge.cpp
    stages/sink_archive.cpp
    stages/sink_image_out.cpp
    stages/sink_mjpeg.cpp
)

add_executable(${ENGINE_TARGET}
    ${ENGINE_SOURCES}
    main/rvs_engine.cpp
)
target_include_directories(${ENGINE_TARGET} PRIVATE ${RVS_OPENCV_INC})
target_link_libraries(${ENGINE_TARGET} ${RVS_OPENCV_LIBS})
if(RVS_TARGET STREQUAL "board")
    target_link_libraries(${ENGINE_TARGET} ${RVS_RKNNRT} z log dl)   # bionic 无独立 libpthread
    target_compile_definitions(${ENGINE_TARGET} PRIVATE RVS_BOARD=1)
    target_include_directories(${ENGINE_TARGET} PRIVATE ${RVS_RKNN_INC})
    target_sources(${ENGINE_TARGET} PRIVATE stages/infer_rknn_board.cpp)
    # 物料探针：板上打印模型输出 zp/scale（金标准回归用）
    add_executable(attrs_probe ../tools/attrs_probe.cpp)
    target_include_directories(attrs_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${RVS_RKNN_INC})
    target_link_libraries(attrs_probe ${RVS_RKNNRT})
    # V4L2 格式探针：枚举节点支持的像素格式（HDMI-in/摄像头采集调试用）
    add_executable(v4l2_probe ../tools/v4l2_probe.cpp)
    target_include_directories(v4l2_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    # M0 gate 基线：外部 demo 源码（业务资产，路径外置；不设 RVS_BASELINE_DEMO_DIR 则跳过）
    if(RVS_BASELINE_DEMO_DIR AND EXISTS "${RVS_BASELINE_DEMO_DIR}/helmet_detect.cpp")
        add_executable(baseline_demo
            "${RVS_BASELINE_DEMO_DIR}/test-helmet_detect.cpp"
            "${RVS_BASELINE_DEMO_DIR}/helmet_detect.cpp"
            "${RVS_BASELINE_DEMO_DIR}/helmet_postprocess.cc")
        target_include_directories(baseline_demo PRIVATE
            ${RVS_BASELINE_DEMO_DIR}
            ${RVS_OPENCV_INC}
            ${RVS_RKNN_INC}
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv_shim)
        target_link_libraries(baseline_demo ${RVS_RKNNRT} ${RVS_OPENCV_LIBS} z log dl)
    endif()
elseif(RVS_TARGET STREQUAL "host")
    target_sources(${ENGINE_TARGET} PRIVATE stages/infer_rknn_stub.cpp)
    target_link_libraries(${ENGINE_TARGET} pthread)
endif()

# ---- 单元测试（仅 host；plain assert + ctest）----
if(RVS_TARGET STREQUAL "host")
    macro(rvs_test NAME)
        add_executable(${NAME} ${ARGN})
        target_include_directories(${NAME} PRIVATE ${RVS_OPENCV_INC})
        target_link_libraries(${NAME} ${RVS_OPENCV_LIBS} pthread)
        add_test(NAME ${NAME} COMMAND ${NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endmacro()
    rvs_test(test_spec tests/test_spec.cpp core/spec.cpp)
    rvs_test(test_assemble tests/test_assemble.cpp core/registry.cpp core/assembler.cpp)
    rvs_test(test_executor tests/test_executor.cpp core/executor.cpp)
    rvs_test(test_source tests/test_source.cpp core/executor.cpp core/registry.cpp stages/source_image_dir.cpp)
    rvs_test(test_letterbox tests/test_letterbox.cpp core/registry.cpp stages/preprocess_letterbox.cpp)
    rvs_test(test_infer_stub tests/test_infer_stub.cpp core/registry.cpp stages/infer_rknn_stub.cpp)
    rvs_test(test_postprocess tests/test_postprocess.cpp core/registry.cpp stages/postprocess_yolov5.cpp)
    rvs_test(test_sink tests/test_sink.cpp core/registry.cpp stages/sink_image_out.cpp)
    rvs_test(test_main tests/test_main.cpp core/spec.cpp core/registry.cpp core/assembler.cpp
             core/executor.cpp core/run_project.cpp
             stages/source_image_dir.cpp stages/preprocess_letterbox.cpp stages/postprocess_yolov5.cpp
             stages/sink_image_out.cpp stages/infer_rknn_stub.cpp)
    rvs_test(test_mjpeg tests/test_mjpeg.cpp core/registry.cpp stages/sink_mjpeg.cpp)
    rvs_test(test_judge tests/test_judge.cpp core/registry.cpp stages/judge.cpp)
    rvs_test(test_archive tests/test_archive.cpp core/registry.cpp stages/sink_archive.cpp)
    rvs_test(gold_regression tests/gold_regression.cpp core/registry.cpp stages/postprocess_yolov5.cpp)
endif()
