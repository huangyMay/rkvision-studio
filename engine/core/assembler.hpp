// 装配器：PipelineSpec → Stage 链。创建、注入 id、Init、校验 source/sink 位置。
#pragma once
#include <memory>
#include <vector>

#include "core/spec.hpp"
#include "core/stage.hpp"

// 失败抛 std::runtime_error，what() 含 plugin 名与 stage id（Review Focus 5）
std::vector<std::unique_ptr<IStage>> Assemble(const PipelineSpec& spec, const StageContext& ctx);
