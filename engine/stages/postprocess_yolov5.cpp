#include "stages/postprocess_yolov5.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "core/registry.hpp"

REGISTER_STAGE("yolov5_det", Yolov5DetPost)

namespace {

/* anchor 常量与 $DEMO 一致：18 int 连续，process 内按 anchor[a*2] 跨步索引 */
const int kAnchorFlat[18] = {
    10, 13, 16, 30, 33, 23,         /* P3, stride 8  */
    30, 61, 62, 45, 59, 119,        /* P4, stride 16 */
    116, 90, 156, 198, 373, 326,    /* P5, stride 32 */
};

constexpr int kObjNumbMaxSize = 64;

inline int clampi(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

inline float deqnt(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

inline int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    float c = dst_val <= -128 ? -128 : (dst_val >= 127 ? 127 : dst_val);
    return (int8_t)c;
}

/* 单尺度解码（逐位复刻 $DEMO/helmet_postprocess.cc process()，含官方怪癖：
 * 阈值以 logit 域量化到 int8 后比较；x/y/w/h 不过 sigmoid（my 模型已内置）） */
int ProcessScale(const std::vector<int8_t>& input, const int* anchor, int grid_h, int grid_w,
                 int stride, float threshold, int32_t zp, float scale,
                 int n_class, std::vector<float>& boxes, std::vector<float>& objProbs,
                 std::vector<int>& classId) {
    const int prop_box = 5 + n_class;
    const size_t grid_len = (size_t)grid_h * grid_w;
    int validCount = 0;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[((size_t)(prop_box * a + 4)) * grid_len + (size_t)i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    size_t base = (size_t)(prop_box * a) * grid_len + (size_t)i * grid_w + j;
                    const int8_t* in_ptr = input.data() + base;
                    float box_x = deqnt(in_ptr[0], zp, scale) * 2.0 - 0.5;
                    float box_y = deqnt(in_ptr[grid_len], zp, scale) * 2.0 - 0.5;
                    float box_w = deqnt(in_ptr[2 * grid_len], zp, scale) * 2.0;
                    float box_h = deqnt(in_ptr[3 * grid_len], zp, scale) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < n_class; ++k) {
                        int8_t prob = in_ptr[(size_t)(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8) {
                        objProbs.push_back(deqnt(maxClassProbs, zp, scale) *
                                           deqnt(box_confidence, zp, scale));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

} // namespace

int Yolov5DetPost::Init(const nlohmann::json& params, const StageContext&) {
    if (!params.is_null()) {
        conf_thres_ = params.value("conf_thres", 0.25f);
        iou_thres_ = params.value("iou_thres", 0.45f);
        if (params.contains("classes")) {
            for (const auto& c : params["classes"]) classes_.push_back(c.get<std::string>());
        }
    }
    if (classes_.empty()) {
        fprintf(stderr, "yolov5_det: params.classes is required\n");
        return -1;
    }
    return 0;
}

bool Yolov5DetPost::Process(Frame& frame) {
    if (frame.image.empty()) {
        fprintf(stderr, "yolov5_det: frame.image is empty\n");
        return false;
    }
    if (frame.raw_outputs.size() != 3 || frame.out_zps.size() != 3 ||
        frame.out_scales.size() != 3) {
        fprintf(stderr, "yolov5_det: expect 3 raw outputs + 3 zp + 3 scale, got %zu/%zu/%zu\n",
                frame.raw_outputs.size(), frame.out_zps.size(), frame.out_scales.size());
        return false;
    }
    const int model_in_w = frame.image.cols;
    const int model_in_h = frame.image.rows;
    const int n_class = (int)classes_.size();
    if (n_class <= 0) {
        fprintf(stderr, "yolov5_det: params.classes is empty\n");
        return false;
    }
    // 张量长度必须容纳 3 anchor × (5 + n_class) × grid；否则下面的跨步索引会越界读。
    const size_t grid[3] = {
        (size_t)(model_in_h / 8) * (size_t)(model_in_w / 8),
        (size_t)(model_in_h / 16) * (size_t)(model_in_w / 16),
        (size_t)(model_in_h / 32) * (size_t)(model_in_w / 32),
    };
    for (int i = 0; i < 3; ++i) {
        const size_t need = (size_t)3 * (size_t)(5 + n_class) * grid[i];
        if (frame.raw_outputs[i].size() < need) {
            fprintf(stderr,
                    "yolov5_det: output[%d] too short: %zu < %zu "
                    "(classes=%d, model=%dx%d; check params.classes against the model)\n",
                    i, frame.raw_outputs[i].size(), need, n_class, model_in_w, model_in_h);
            return false;
        }
    }

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    /* 与 $DEMO 相同的 anchor 跨步偏移（+0/+6/+12），a 循环内读 anchor[a*2] */
    int v0 = ProcessScale(frame.raw_outputs[0], kAnchorFlat + 0, model_in_h / 8, model_in_w / 8, 8,
                          conf_thres_, frame.out_zps[0], frame.out_scales[0], n_class,
                          filterBoxes, objProbs, classId);
    int v1 = ProcessScale(frame.raw_outputs[1], kAnchorFlat + 6, model_in_h / 16, model_in_w / 16, 16,
                          conf_thres_, frame.out_zps[1], frame.out_scales[1], n_class,
                          filterBoxes, objProbs, classId);
    int v2 = ProcessScale(frame.raw_outputs[2], kAnchorFlat + 12, model_in_h / 32, model_in_w / 32, 32,
                          conf_thres_, frame.out_zps[2], frame.out_scales[2], n_class,
                          filterBoxes, objProbs, classId);
    int validCount = v0 + v1 + v2;

    frame.detections.clear();
    if (validCount <= 0) return true;

    /* greedy 类内 NMS（$DEMO 替换版，行为与官方对齐） */
    std::vector<int> indexArray(validCount);
    std::iota(indexArray.begin(), indexArray.end(), 0);
    std::sort(indexArray.begin(), indexArray.end(),
              [&](int A, int B) { return objProbs[A] > objProbs[B]; });

    std::vector<bool> suppressed(validCount, false);
    std::vector<int> kept;
    for (int i = 0; i < validCount; ++i) {
        int n = indexArray[i];
        if (suppressed[n]) continue;
        kept.push_back(n);
        float xmin = filterBoxes[n * 4 + 0], ymin = filterBoxes[n * 4 + 1];
        float xmax = xmin + filterBoxes[n * 4 + 2], ymax = ymin + filterBoxes[n * 4 + 3];
        for (int j = i + 1; j < validCount; ++j) {
            int m = indexArray[j];
            if (suppressed[m] || classId[m] != classId[n]) continue;
            float mxn = filterBoxes[m * 4 + 0], myn = filterBoxes[m * 4 + 1];
            float mxf = mxn + filterBoxes[m * 4 + 2], myf = myn + filterBoxes[m * 4 + 3];
            float ix = std::max(0.0f, std::min(xmax, mxf) - std::max(xmin, mxn));
            float iy = std::max(0.0f, std::min(ymax, myf) - std::max(ymin, myn));
            float inter = ix * iy;
            float u = filterBoxes[n * 4 + 2] * filterBoxes[n * 4 + 3] +
                      filterBoxes[m * 4 + 2] * filterBoxes[m * 4 + 3] - inter;
            if (u > 0 && inter / u > iou_thres_) suppressed[m] = true;
        }
    }

    int last_count = 0;
    for (int n : kept) {
        if (last_count >= kObjNumbMaxSize) break;

        /* 还原：先减 pad，再 clamp 到模型域，再除 scale（与 $DEMO 同序） */
        float x1 = filterBoxes[n * 4 + 0] - frame.pad_l;
        float y1 = filterBoxes[n * 4 + 1] - frame.pad_t;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        Detection d;
        d.label = classes_[classId[n]];
        d.score = objProbs[n];
        /* demo 语义: (int)(clamp(x,0,model_in_w) / scale_w) —— 除法后还有一次 int 截断，
         * 少这次截断会差 1px，正好压在金标准 IoU>=0.98 验收线上 */
        d.x1 = (float)(int)(clampi(x1, 0, model_in_w) / frame.scale);
        d.y1 = (float)(int)(clampi(y1, 0, model_in_h) / frame.scale);
        d.x2 = (float)(int)(clampi(x2, 0, model_in_w) / frame.scale);
        d.y2 = (float)(int)(clampi(y2, 0, model_in_h) / frame.scale);
        frame.detections.push_back(d);
        ++last_count;
    }
    return true;
}
