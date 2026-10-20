// letterbox 预处理：数值逐位复刻 $DEMO/helmet_detect.cpp:105-119。
// 差异即事故：lround（非截断）、白边 255（非 114）、单 scale（坐标还原同源）。
#pragma once
#include "core/stage.hpp"

class LetterboxStage : public IStage {
public:
    int Init(const nlohmann::json& params, const StageContext& ctx) override;
    bool Process(Frame& frame) override;
    const StageMeta& Meta() const override {
        static StageMeta m{"letterbox", false, false, false};
        return m;
    }

private:
    int size_ = 640;
    int pad_value_ = 255;   // 官方 letter_box 反汇编: copyMakeBorder 标量 = 255 白边
};
