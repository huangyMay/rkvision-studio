// 决定性实验夹具：调用【原版】helmet_postprocess.cc 的 post_process，
// 喂 npu_outputs 金张量 + 给定 (zp,scale)，输出与 gold 日志同格式的框。
// argv: out0 out1 out2 zp scale
#include "postprocess.h"

#include <cstdio>
#include <fstream>
#include <vector>

using namespace std;

/* 与 helmet_detect.cpp 中的 extern 声明一致（postprocess.h 未含此声明） */
int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

static bool load(const char* p, vector<int8_t>& v) {
    ifstream f(p, ios::binary);
    if (!f.good()) return false;
    f.seekg(0, ios::end);
    long n = f.tellg();
    f.seekg(0);
    v.resize(n);
    f.read((char*)v.data(), n);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 6) { printf("usage: harness out0 out1 out2 zp scale\n"); return 1; }
    vector<int8_t> t0, t1, t2;
    if (!load(argv[1], t0) || !load(argv[2], t1) || !load(argv[3], t2)) {
        printf("load fail\n"); return 1;
    }
    int32_t zp = atoi(argv[4]);
    float scale = atof(argv[5]);

    vector<int32_t> zps = {zp, zp, zp};
    vector<float> scales = {scale, scale, scale};

    // 金标准 test 图 = 1200x798, letterbox: scale=640/1200, dw=0, dh=107
    BOX_RECT pads;
    pads.left = 0; pads.right = 0; pads.top = 107; pads.bottom = 107;
    detect_result_group_t group;
    post_process((int8_t*)t0.data(), (int8_t*)t1.data(), (int8_t*)t2.data(),
                 640, 640, 0.25f, 0.45f, pads, 640.f / 1200.f, 640.f / 1200.f,
                 zps, scales, &group);
    for (int i = 0; i < group.count; ++i) {
        printf("%s @ (%d %d %d %d) %.6f\n", group.results[i].name,
               group.results[i].box.left, group.results[i].box.top,
               group.results[i].box.right, group.results[i].box.bottom,
               group.results[i].prop);
    }
    return 0;
}
