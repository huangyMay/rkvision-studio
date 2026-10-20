# 板端工具链包装：NDK r18b（bionic + c++_static，产物自包含）
# NDK 不随仓库分发，须自行准备：export ANDROID_NDK_ROOT=~/.../android-ndk-r18b
if(NOT DEFINED ENV{ANDROID_NDK_ROOT})
    message(FATAL_ERROR
        "未设置 ANDROID_NDK_ROOT\n"
        "  export ANDROID_NDK_ROOT=/path/to/android-ndk-r18b\n"
        "  （需 r18b 及以上，r18b 为本项目验证版本）")
endif()
set(NDK_ROOT "$ENV{ANDROID_NDK_ROOT}")
if(NOT EXISTS "${NDK_ROOT}/build/cmake/android.toolchain.cmake")
    message(FATAL_ERROR "NDK 工具链不存在: ${NDK_ROOT}/build/cmake/android.toolchain.cmake")
endif()
set(ANDROID_ABI arm64-v8a CACHE STRING "" FORCE)
set(ANDROID_PLATFORM android-24 CACHE STRING "" FORCE)
set(ANDROID_STL c++_static CACHE STRING "" FORCE)
include("${NDK_ROOT}/build/cmake/android.toolchain.cmake")
