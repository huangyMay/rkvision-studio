// yolov5_det 后处理：骨架移植自 $DEMO/helmet_postprocess.cc（其源头是瑞芯微
// rknpu2 examples 的 yolov5 postprocess，Apache-2.0），与官方 .o 逐位一致：
//   int8 反量化(zp/scale 运行时下发) → 解码(*2-0.5 / 平方*anchor) → 阈值 →
//   greedy 类内 NMS → letterbox 还原原图坐标。
// 家族常量：3 尺度 stride 8/16/32，YOLOv5 默认 anchor；类别数由 pipeline.json 下发。
#pragma once
#include <string>
#include <vector>

#include "core/stage.hpp"

class Yolov5DetPost : public IStage {
public:
    int Init(const nlohmann::json& params, const StageContext& ctx) override;
    bool Process(Frame& frame) override;
    const StageMeta& Meta() const override {
        static StageMeta m{"yolov5_det", false, false, true};
        return m;
    }

private:
    float conf_thres_ = 0.25f;
    float iou_thres_ = 0.45f;
    std::vector<std::string> classes_;
};
