// infer stage：librknnrt 推理（板）/ 金张量桩（host）。
// 板实现逐位复刻 $DEMO/helmet_detect.cpp 的输入/输出处理：
//   uint8 NHWC RGB 透传（量化由 NPU 完成）→ int8 输出 want_float=0 → zp/scale 运行时查询。
#pragma once
#include <string>
#include <vector>

#include <cstdint>

#include "core/stage.hpp"

#ifdef RVS_BOARD
#include "rknn_api.h"
#endif

class InferStage : public IStage {
public:
    int Init(const nlohmann::json& params, const StageContext& ctx) override;
    bool Process(Frame& frame) override;
    const StageMeta& Meta() const override {
        static StageMeta m{"infer", false, false, true};
        return m;
    }

private:
    std::string model_key_;
    std::string model_path_;     // 解析后的 .rknn 绝对/工程包相对路径
    std::string stub_dir_;       // host 桩：out{i}.bin + attrs.json 所在目录
    int n_outputs_ = 3;

#ifdef RVS_BOARD
    rknn_context ctx_ = 0;
    bool loaded_ = false;
#endif
};
