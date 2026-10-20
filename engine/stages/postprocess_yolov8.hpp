// yolov8_det 后处理：yolov8/11 裸头（model_zoo 分组导出）
//   每层 3 张量 [(1,64,H,W) DFL, (1,nc,H,W) cls, (1,1,H,W) obj]，层 = 80/40/20
//   DFL int8 softmax 16 bins → dist；cls/obj 已 sigmoid 激活（量化域直比）
//   框几何 x1=((j+0.5)-l)*stride；解码逻辑与 tools/onnx_infer.py 形态D 逐位对拍
#pragma once
#include <string>
#include <vector>

#include "core/stage.hpp"

class Yolov8DetPost : public IStage {
public:
    int Init(const nlohmann::json& params, const StageContext& ctx) override;
    bool Process(Frame& frame) override;
    const StageMeta& Meta() const override {
        static StageMeta m{"yolov8_det", false, false, true};
        return m;
    }

private:
    float conf_thres_ = 0.25f;
    float iou_thres_ = 0.45f;
    std::vector<std::string> classes_;
};
