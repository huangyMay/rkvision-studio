// yolov8_det：yolov8/11 裸头解码（model_zoo 分组金字塔）
// 输出形态（以 yolov8n_coco 为例，9 路 int8）：
//   每层 3 张量 [(1,64,H,W) DFL, (1,nc,H,W) cls, (1,1,H,W) obj]，层 = 80/40/20
//   DFL int8 域 softmax 16 bins → dist；cls/obj int8 域已激活（黑图全 0 可证），
//   置信度 = deqnt(obj) * max(deqnt(cls))
// 框几何：x1 = ((j + 0.5) - l) * stride（格坐标 l/t/r/b ∈ [0,16)），anchor 在格心
// 坐标还原与 yolov5_det 同序：减 pad → clamp → 除 scale → int 截断
#include "stages/postprocess_yolov8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "core/registry.hpp"

REGISTER_STAGE("yolov8_det", Yolov8DetPost)

namespace {

constexpr int kObjNumbMaxSize = 64;
constexpr int kDflCh = 64;                       // 4 边 × 16 bins

inline int clampi(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

inline float deqnt(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

/* softmax（int8 deqnt 后）沿 16 bins */
void softmax16(const std::vector<int8_t>& in, size_t base, size_t grid_len,
               int H, int W, float* out /* [4][H*W] */) {
    for (int c = 0; c < 4; ++c) {
        for (int i = 0; i < H * W; ++i) {
            float mx = -1e9f;
            float v[16];
            for (int b = 0; b < 16; ++b) {
                v[b] = deqnt(in[base + (size_t)((c * 16 + b)) * grid_len + (size_t)i], 0, 1);
                if (v[b] > mx) mx = v[b];
            }
            float sum = 0;
            for (int b = 0; b < 16; ++b) {
                v[b] = std::exp(v[b] - mx);
                sum += v[b];
            }
            float acc = 0;
            for (int b = 0; b < 16; ++b) acc += v[b] * (float)b;
            out[c * H * W + i] = acc;
        }
    }
}

} // namespace

int Yolov8DetPost::Init(const nlohmann::json& params, const StageContext&) {
    if (!params.is_null()) {
        conf_thres_ = params.value("conf_thres", 0.25f);
        iou_thres_ = params.value("iou_thres", 0.45f);
        if (params.contains("classes")) {
            for (const auto& c : params["classes"]) classes_.push_back(c.get<std::string>());
        }
    }
    /* classes 可省略：缺省标签 cls<id>（label 兜底在 Process 内）*/
    return 0;
}

bool Yolov8DetPost::Process(Frame& frame) {
    const size_t n_out = frame.raw_outputs.size();
    const int model_in_w = frame.image.cols;
    const int model_in_h = frame.image.rows;
    if (n_out < 3 || n_out % 3 != 0) {
        fprintf(stderr, "yolov8_det: expect 3k raw outputs, got %zu\n", n_out);
        return false;
    }

    /* ===== 分层：dfl 张量（len=64*grid_len）定层，cls/obj 按同 grid_len 归层 ===== */
    for (size_t i = 0; i < n_out; ++i)
        fprintf(stderr, "[y8dbg] out[%zu] len=%zu zp=%d scale=%f\n", i,
                frame.raw_outputs[i].size(), frame.out_zps[i], frame.out_scales[i]);
    struct FinalLevel {
        size_t grid_len;
        int dfl_idx = -1, cls_idx = -1, obj_idx = -1;
    };
    std::vector<FinalLevel> fin;
    for (size_t i = 0; i < n_out; ++i) {
        const size_t len = frame.raw_outputs[i].size();
        if (len % kDflCh != 0) continue;
        const size_t gl = len / kDflCh;
        bool found = false;
        for (auto& f : fin)
            if (f.grid_len == gl) { found = true; break; }
        if (!found) {
            FinalLevel nf;
            nf.grid_len = gl;
            nf.dfl_idx = (int)i;
            nf.cls_idx = -1;
            nf.obj_idx = -1;
            fin.push_back(nf);
        }
    }
    if (fin.empty() || fin.size() > 4) {
        fprintf(stderr, "yolov8_det: dfl levels not found (%zu)\n", fin.size());
        return false;
    }
    for (size_t i = 0; i < n_out; ++i) {
        const size_t len = frame.raw_outputs[i].size();
        bool is_dfl = false;
        for (auto& f : fin)
            if (len == f.grid_len * kDflCh) { is_dfl = true; break; }
        if (is_dfl) continue;
        for (auto& f : fin) {
            if (len == f.grid_len && f.obj_idx < 0) { f.obj_idx = (int)i; break; }
            if (len % f.grid_len == 0 && f.cls_idx < 0) { f.cls_idx = (int)i; break; }
        }
    }
    int nc = 0;
    for (auto& f : fin) {
        if (f.dfl_idx < 0 || f.cls_idx < 0 || f.obj_idx < 0) {
            fprintf(stderr, "yolov8_det: level incomplete (dfl=%d cls=%d obj=%d grid=%zu)\n",
                    f.dfl_idx, f.cls_idx, f.obj_idx, f.grid_len);
            return false;
        }
        const int lnc = (int)(frame.raw_outputs[f.cls_idx].size() / f.grid_len);
        if (nc == 0) nc = lnc;
        else if (nc != lnc) { fprintf(stderr, "yolov8_det: nc mismatch\n"); return false; }
    }
    if (nc <= 0) { fprintf(stderr, "yolov8_det: nc infer failed\n"); return false; }
    if ((int)classes_.size() != nc)
        fprintf(stderr, "yolov8_det: warn classes=%zu != model nc=%d\n", classes_.size(), nc);

    std::vector<float> filterBoxes;                  // x,y,w,h（模型域像素）
    std::vector<float> objProbs;
    std::vector<int> classId;

    for (auto& f : fin) {
        const int H = (int)std::lround(std::sqrt((double)f.grid_len));
        const int W = H;
        const size_t gl = f.grid_len;
        const int stride = model_in_w / W;
        const int8_t* dfl = frame.raw_outputs[f.dfl_idx].data();
        const int32_t dzp = frame.out_zps[f.dfl_idx];
        const float dscale = frame.out_scales[f.dfl_idx];
        const int8_t* clsp = frame.raw_outputs[f.cls_idx].data();
        const int32_t czp = frame.out_zps[f.cls_idx];
        const float cscale = frame.out_scales[f.cls_idx];
        const int8_t* objp = frame.raw_outputs[f.obj_idx].data();
        const int32_t ozp = frame.out_zps[f.obj_idx];
        const float oscale = frame.out_scales[f.obj_idx];

        /* dist[c][i]：softmax 期望（bins 0..15） */
        std::vector<float> dist(4 * gl, 0.f);
        {
            std::vector<float> e(16);
            for (int c = 0; c < 4; ++c)
                for (int i = 0; i < gl; ++i) {
                    float mx = -1e9f;
                    for (int b = 0; b < 16; ++b) {
                        float v = deqnt(dfl[(size_t)((c * 16 + b)) * gl + (size_t)i], dzp, dscale);
                        e[b] = v;
                        if (v > mx) mx = v;
                    }
                    float sum = 0, acc = 0;
                    for (int b = 0; b < 16; ++b) {
                        e[b] = std::exp(e[b] - mx);
                        sum += e[b];
                    }
                    for (int b = 0; b < 16; ++b) acc += e[b] * (float)b;
                    dist[(size_t)c * gl + (size_t)i] = acc / sum;
                }
        }

        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                const size_t cell = (size_t)i * W + (size_t)j;
                const int8_t objq = objp[cell];
                /* cls：量化域直接比大小（已 sigmoid 激活）*/
                int8_t maxq = clsp[cell];
                int maxId = 0;
                for (int k = 1; k < nc; ++k) {
                    int8_t q = clsp[(size_t)k * gl + cell];
                    if (q > maxq) { maxq = q; maxId = k; }
                }
                const float conf = deqnt(objq, ozp, oscale) * deqnt(maxq, czp, cscale);
                if (conf < conf_thres_) continue;

                const float l = dist[0 * gl + cell];
                const float t = dist[1 * gl + cell];
                const float r = dist[2 * gl + cell];
                const float b = dist[3 * gl + cell];
                float x1 = ((float)j + 0.5f - l) * (float)stride;
                float y1 = ((float)i + 0.5f - t) * (float)stride;
                float x2 = ((float)j + 0.5f + r) * (float)stride;
                float y2 = ((float)i + 0.5f + b) * (float)stride;

                filterBoxes.push_back(x1);
                filterBoxes.push_back(y1);
                filterBoxes.push_back(x2 - x1);
                filterBoxes.push_back(y2 - y1);
                objProbs.push_back(conf);
                classId.push_back(maxId);
            }
    }

    const int validCount = (int)objProbs.size();
    frame.detections.clear();
    if (validCount <= 0) return true;

    /* greedy 类内 NMS（与 yolov5_det 同实现） */
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
        float x1 = filterBoxes[n * 4 + 0] - frame.pad_l;
        float y1 = filterBoxes[n * 4 + 1] - frame.pad_t;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        Detection d;
        d.label = classId[n] < (int)classes_.size() ? classes_[classId[n]]
                                                    : std::to_string(classId[n]);
        d.score = objProbs[n];
        d.x1 = (float)(int)(clampi(x1, 0, model_in_w) / frame.scale);
        d.y1 = (float)(int)(clampi(y1, 0, model_in_h) / frame.scale);
        d.x2 = (float)(int)(clampi(x2, 0, model_in_w) / frame.scale);
        d.y2 = (float)(int)(clampi(y2, 0, model_in_h) / frame.scale);
        frame.detections.push_back(d);
        ++last_count;
    }
    return true;
}
