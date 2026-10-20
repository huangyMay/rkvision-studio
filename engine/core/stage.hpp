// Stage 契约：插件的生命周期与能力标记（借鉴 VisionMix ToolBase / OpenIVS2 ISequenceNode）
#pragma once
#include <string>

#include <nlohmann/json.hpp>

#include "core/frame.hpp"
#include "core/spec.hpp"

struct StageContext {
    const PipelineSpec* spec = nullptr;  // 工程包引用（models 等）
    std::string project_root;            // 工程包根目录，相对路径的解析基准
};

struct StageMeta {
    const char* name;      // 注册名（与 pipeline.json 的 plugin 字段对应）
    bool is_source;
    bool is_sink;
    bool needs_model;
};

class IStage {
public:
    virtual ~IStage() = default;

    // 参数初始化。返回非 0 表示失败；失败信息写 stderr（装配器会带 stage id 再包装）
    virtual int Init(const nlohmann::json& params, const StageContext& ctx) = 0;

    // 处理一帧。source 返回 false 表示不再产帧（流水线终止）
    virtual bool Process(Frame& frame) = 0;

    virtual const StageMeta& Meta() const = 0;

    // stage id 由装配器按 pipeline.json 注入
    void SetId(std::string id) { id_ = std::move(id); }
    const std::string& Id() const { return id_; }

private:
    std::string id_;
};
