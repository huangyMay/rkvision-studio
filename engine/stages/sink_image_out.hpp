// image_out sink：画框存图 + 每帧结果 JSON（compare 的数据源）。
#pragma once
#include "core/stage.hpp"

class ImageOutSink : public IStage {
public:
    int Init(const nlohmann::json& params, const StageContext& ctx) override;
    bool Process(Frame& frame) override;
    const StageMeta& Meta() const override {
        static StageMeta m{"image_out", false, true, false};
        return m;
    }

private:
    std::string out_dir_;
    bool draw_ = true;
    bool save_json_ = true;
};
