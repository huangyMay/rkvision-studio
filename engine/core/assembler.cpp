#include "core/assembler.hpp"

#include <stdexcept>

#include "core/registry.hpp"

std::vector<std::unique_ptr<IStage>> Assemble(const PipelineSpec& spec, const StageContext& ctx) {
    if (spec.pipeline.empty()) {
        throw std::runtime_error("assemble: pipeline is empty (field: pipeline)");
    }

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
