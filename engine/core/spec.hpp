// pipeline.json schema v1 的内存表示（见 docs/2026-10-16-框架设计.md §5.1）
#pragma once
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

struct StageSpec {
    std::string id;
    std::string plugin;
    std::string model;        // 仅 infer 类 stage 有：models 段的键名
    nlohmann::json params;    // 透传给插件，由插件自解
};

struct ModelSpec {
    std::string family;       // 插件家族名（如 yolov5-det），决定配哪个后处理插件
    std::string file;         // .rknn 路径（相对工程包根）
    nlohmann::json input;     // 输入张量描述（name/shape/layout/dtype）
};

struct PipelineSpec {
    int version = 0;
    std::string name;
    std::string core_mask = "auto";   // "auto" | "0x1" | "0x2" | "0x3"（RK3576 双核）
    std::map<std::string, ModelSpec> models;
    std::vector<StageSpec> pipeline;
};

// 解析并校验工程包描述文件。校验失败抛 std::runtime_error，
// what() 恒含字段名或键名，便于上游定位（Review Focus 5）。
PipelineSpec load_spec(const std::string& json_path);
