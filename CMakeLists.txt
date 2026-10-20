cmake_minimum_required(VERSION 3.13)
project(rkvision_studio CXX)

option(RVS_TARGET "host | board" "host")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()
add_subdirectory(engine)
