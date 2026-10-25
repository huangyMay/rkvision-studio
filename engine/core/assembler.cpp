#include "core/assembler.hpp"

#include <stdexcept>

#include "core/registry.hpp"

namespace {

// models.family 的文档语义是"决定配哪个后处理插件"（spec.hpp）。
// 这里把它变成可执行校验：infer 的下一个 stage 必须认识这个家族。
const char* ExpectedFamily(const std::string& plugin) {
    if (plugin == "yolov5_det") return "yolov5-det";
    if (plugin == "yolov8_det") return "yolov8-det";
    return nullptr;
}

bool FamilyMatches(const std::string& got, const char* want) {
    if (got == want) return true;
    // 容忍省略 -det 后缀（yolov5 / yolov8）
    const std::string w(want);
    const size_t dash = w.rfind('-');
    return dash != std::string::npos && got == w.substr(0, dash);
}

void CheckModelFamilies(const PipelineSpec& spec) {
    for (size_t i = 0; i + 1 < spec.pipeline.size(); ++i) {
        const StageSpec& ss = spec.pipeline[i];
        if (ss.model.empty()) continue;   // 只有 infer 类 stage 带 model
        const std::map<std::string, ModelSpec>::const_iterator m = spec.models.find(ss.model);
        if (m == spec.models.end()) continue;   // load_spec 已校验引用存在
        const char* want = ExpectedFamily(spec.pipeline[i + 1].plugin);
        if (!want) continue;                    // 后面不是已知后处理，不做约束
        if (!FamilyMatches(m->second.family, want)) {
            throw std::runtime_error(
                "assemble: model '" + ss.model + "' family '" + m->second.family +
                "' does not match postprocess plugin '" + spec.pipeline[i + 1].plugin +
                "' (expected family '" + want + "')");
        }
    }
}

} // namespace

std::vector<std::unique_ptr<IStage>> Assemble(const PipelineSpec& spec, const StageContext& ctx) {
    if (spec.pipeline.empty()) {
        throw std::runtime_error("assemble: pipeline is empty (field: pipeline)");
    }
    CheckModelFamilies(spec);

    std::vector<std::unique_ptr<IStage>> chain;
    chain.reserve(spec.pipeline.size());
    for (const StageSpec& ss : spec.pipeline) {
        std::unique_ptr<IStage> stage = StageRegistry::Instance().Create(ss.plugin);
        if (!stage) {
            throw std::runtime_error("assemble: unknown plugin '" + ss.plugin +
                                     "' at stage '" + ss.id + "'");
        }
        stage->SetId(ss.id);
        // spec 层的 model 引用注入 params，插件只认一个真值源
        nlohmann::json params = ss.params.is_null() ? nlohmann::json::object() : ss.params;
        if (!ss.model.empty()) params["model"] = ss.model;
        if (stage->Init(params, ctx) != 0) {
            throw std::runtime_error("assemble: stage '" + ss.id + "' (" + ss.plugin +
                                     ") Init failed");
        }
        chain.push_back(std::move(stage));
    }

    if (!chain.front()->Meta().is_source) {
        throw std::runtime_error("assemble: first stage '" + chain.front()->Id() +
                                 "' must be a source");
    }
    if (!chain.back()->Meta().is_sink) {
        throw std::runtime_error("assemble: last stage '" + chain.back()->Id() +
                                 "' must be a sink");
    }
    return chain;
}
