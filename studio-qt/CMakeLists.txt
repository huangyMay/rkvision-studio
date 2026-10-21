cmake_minimum_required(VERSION 3.16)
project(rvs_studio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Network Multimedia MultimediaWidgets Concurrent)

add_executable(rvs_studio
    src/main.cpp
    # core 服务层（无 UI 依赖，可单测）
    src/core/adbclient.cpp
    src/core/mjpegclient.cpp
    src/core/deployer.cpp
    src/core/projectregistry.cpp
    src/core/onnxrunner.cpp
    src/core/onnxanalyzer.cpp
    src/core/rknnsimrunner.cpp
    src/core/boardperf.cpp
    src/core/boardmonitor.cpp
    src/core/inferstream.cpp
    src/model/pipelinespec.cpp
    src/model/workflowgen.cpp
    # ui 界面层
    src/ui/mainwindow.cpp
    src/ui/widgets/previewwidget.cpp
    res/res.qrc
)

target_include_directories(rvs_studio PRIVATE src)
target_link_libraries(rvs_studio PRIVATE
    Qt6::Widgets Qt6::Network Qt6::Multimedia Qt6::MultimediaWidgets Qt6::Concurrent)
set(RVS_WSL_DISTRO "Ubuntu-22.04" CACHE STRING "wsl.exe -d 使用的发行版名")
# 本工程以 -S studio-qt 独立配置，CMAKE_SOURCE_DIR 即 studio-qt/；脚本在仓库根 tools/
cmake_path(SET RVS_TOOLS_DIR NORMALIZE "${CMAKE_SOURCE_DIR}/../tools")
target_compile_definitions(rvs_studio PRIVATE
    PROJECTS_ROOT="${CMAKE_SOURCE_DIR}/projects"
    RVS_TOOLS_DIR="${RVS_TOOLS_DIR}"
    RVS_WSL_DISTRO="${RVS_WSL_DISTRO}")

set_target_properties(rvs_studio PROPERTIES WIN32_EXECUTABLE ON)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(rvs_studio PRIVATE -s)
endif()
