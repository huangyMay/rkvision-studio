// Task 8: yolov5 后处理（demo 逐位复刻）的行为规格 —— 合成张量，不依赖板上。
// 张量布局: [anchor*7+c][grid]  (PROP_BOX_SIZE=7: 4 box + 1 conf + n_classes)
// zp=0/scale=1.0 → deqnt(q)=q，阈值 thres_i8 = (int)0.25 = 0
#include "core/frame.hpp"
#include "core/registry.hpp"
#include "stages/postprocess_yolov5.hpp"

#include <cstdio>
#include <cstring>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

namespace {

constexpr int GW = 80, GH = 80, ANCHORS = 3, CH = 7;   // stride 8: 7 = 5+2 类
constexpr size_t T0_SIZE = (size_t)GW * GH * ANCHORS * CH;

void set_q(std::vector<int8_t>& t, int a, int c, int i, int j, int8_t q) {
    t[((size_t)(a * CH + c)) * GW * GH + (size_t)i * GW + j] = q;
}

// 构造一个候选：q_x/q_y/q_w/q_h/conf/cls
void put_candidate(std::vector<int8_t>& t, int a, int i, int j,
                   int8_t qx, int8_t qy, int8_t qw, int8_t qh, int8_t conf, int cls, int8_t clsq) {
    set_q(t, a, 0, i, j, qx);
    set_q(t, a, 1, i, j, qy);
    set_q(t, a, 2, i, j, qw);
    set_q(t, a, 3, i, j, qh);
    set_q(t, a, 4, i, j, conf);
    set_q(t, a, 5 + cls, i, j, clsq);
}

Frame make_frame(std::vector<int8_t> t0, std::vector<int8_t> t1, std::vector<int8_t> t2) {
    Frame f;
    f.image.create(640, 640, CV_8UC3);   // letterbox 后的 640 方图
    f.orig_w = 640; f.orig_h = 640;
    f.scale = 1.0f; f.pad_l = 0; f.pad_t = 0;
    f.raw_outputs = {t0, t1, t2};
    f.out_zps = {0, 0, 0};
    f.out_scales = {1.0f, 1.0f, 1.0f};
    return f;
}

} // namespace

int main() {
    StageContext ctx;

    nlohmann::json p;
    p["conf_thres"] = 0.25;
    p["iou_thres"] = 0.45;
    p["classes"] = {"head", "helmet"};

    // ① 合成候选：A(过阈值 head q2)、B(与 A 同类高 IoU，q1 → NMS 抑制)、
    //    C(conf=-1 不过阈值)、D(负坐标 → clamp，helmet q1)
    std::vector<int8_t> t0(T0_SIZE, 0), t1((size_t)40 * 40 * ANCHORS * CH, 0), t2((size_t)20 * 20 * ANCHORS * CH, 0);
    // A: anchor0 grid(10,20)  qx=qy=qw=qh=1 → x=(1*2-0.5+20)*8=164? 精确算:
    //   box_x=(deqnt(1)*2-0.5+20)*8 = (1.5+20)*8 = 172; box_w=(1*2)^2*10=40 → x1=172-20=152
    //   box_y=(1.5+10)*8=92;    box_h=(2)^2*13=52 → y1=92-26=66 → box(152,66,192,118)
    put_candidate(t0, 0, 10, 20, 1, 1, 1, 1, 1, /*cls=*/0, /*clsq=*/2);
    // B: grid(10,21) 同参数 → x1=160,x2=200,y(66,118)；与 A IoU = (192-160)*52 / (2*40*52-...) ≈ 0.667>0.45
    put_candidate(t0, 0, 10, 21, 1, 1, 1, 1, 1, /*cls=*/0, /*clsq=*/1);
    // C: 不过阈值
    put_candidate(t0, 1, 5, 5, 1, 1, 1, 1, /*conf=*/-1, 0, 1);
    // D: grid(0,0) qx=-64 → x=(-128.5+0)*8=-1028 → clamp 0
    put_candidate(t0, 0, 0, 0, /*qx=*/-64, /*qy=*/0, /*qw=*/1, /*qh=*/1, 1, /*cls=*/1, 1);

    {
        Yolov5DetPost st;
        CHECK(st.Init(p, ctx) == 0);
        Frame f = make_frame(t0, t1, t2);
        CHECK(st.Process(f) == true);
        CHECK(f.detections.size() == 2);                       // A、D；B 被 NMS、C 被阈值
        CHECK(f.detections[0].label == "head");
        CHECK(f.detections[0].score == 2.0f);
        CHECK(f.detections[0].x1 == 152.f && f.detections[0].y1 == 66.f);
        CHECK(f.detections[0].x2 == 192.f && f.detections[0].y2 == 118.f);
        CHECK(f.detections[1].label == "helmet");
        CHECK(f.detections[1].score == 1.0f);
        CHECK(f.detections[1].x1 == 0.f && f.detections[1].y1 == 0.f);   // clamp
    }

    // ② 全零张量 → 0 检出不崩（Review Focus 2）
    {
        std::vector<int8_t> z(T0_SIZE, 0);
        Yolov5DetPost st;
        CHECK(st.Init(p, ctx) == 0);
        Frame f = make_frame(z, z, z);
        CHECK(st.Process(f) == true);
        CHECK(f.detections.empty());
    }

    // ③ pad 还原：scale=0.5、pad_l=4、pad_t=2 → 模型坐标 (152,66) → 原图 ((152-4)/0.5, (66-2)/0.5)
    {
        std::vector<int8_t> t0b(T0_SIZE, 0);
        put_candidate(t0b, 0, 10, 20, 1, 1, 1, 1, 1, 0, 2);
        Yolov5DetPost st;
        CHECK(st.Init(p, ctx) == 0);
        Frame f = make_frame(t0b, t1, t2);
        f.scale = 0.5f; f.pad_l = 4; f.pad_t = 2;
        f.orig_w = 1200; f.orig_h = 798;
        CHECK(st.Process(f) == true);
        CHECK(f.detections.size() == 1);
        CHECK(f.detections[0].x1 == 296.f);   // (152-4)/0.5
        CHECK(f.detections[0].y1 == 128.f);   // (66-2)/0.5
        CHECK(f.detections[0].x2 == 376.f);   // (192-4)/0.5
        CHECK(f.detections[0].y2 == 232.f);   // (118-2)/0.5
    }

    // ③b 还原坐标非整数时按 demo 语义 (int) 截断（152/0.7=217.14 → 217）
    {
        std::vector<int8_t> t0b(T0_SIZE, 0);
        put_candidate(t0b, 0, 10, 20, 1, 1, 1, 1, 1, 0, 2);
        Yolov5DetPost st;
        CHECK(st.Init(p, ctx) == 0);
        Frame f = make_frame(t0b, t1, t2);
        f.scale = 0.7f; f.pad_l = 0; f.pad_t = 0;
        f.orig_w = 1200; f.orig_h = 798;
        CHECK(st.Process(f) == true);
        CHECK(f.detections.size() == 1);
        CHECK(f.detections[0].x1 == 217.f);   // demo: (int)(clamp(x,0,640)/scale) 截断
        CHECK(f.detections[0].x2 == 274.f);   // 192/0.7=274.28 → 274
    }

    // ④ 注册名可查
    CHECK(StageRegistry::Instance().Create("yolov5_det") != nullptr);

    printf("test_postprocess: all checks passed\n");
    return 0;
}
