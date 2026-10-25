// 金张量回归：用真实板上输出的 int8 张量跑后处理，打印 engine 结果 JSON。
// attrs.json 缺失（板上 zp/scale 未提取）时显式 SKIP —— e2e_gold.sh 负责真正的 17 框断言。
// 固件常量（金标准 test 图 = 1200x798，letterbox 参数已被 test_letterbox 钉死）。
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/frame.hpp"
#include "core/registry.hpp"
#include "stages/postprocess_yolov5.hpp"

static bool read_all(const std::string& path, std::vector<int8_t>* out) {
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f.good()) return false;
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    f.seekg(0);
    out->resize((size_t)sz);
    f.read((char*)out->data(), sz);
    return f.good();
}

int main() {
    std::ifstream af("assets/gold/attrs.json");
    if (!af.good()) {
        printf("SKIP: assets/gold/attrs.json not found (board zp/scale not extracted yet)\n");
        // 77 = CTest SKIP（不伪装成通过）；CI 需要硬门禁时设 RVS_REQUIRE_GOLD=1
        return std::getenv("RVS_REQUIRE_GOLD") ? 1 : 77;
    }
    nlohmann::json attrs;
    try {
        af >> attrs;
    } catch (const std::exception& e) {
        fprintf(stderr, "bad attrs.json: %s\n", e.what());
        return 1;
    }

    Frame f;
    f.image.create(640, 640, CV_8UC3);
    f.orig_w = 1200; f.orig_h = 798;
    f.scale = 640.f / 1200.f;
    f.pad_l = 0; f.pad_t = 107;

    for (int i = 0; i < 3; ++i) {
        char path[128];
        snprintf(path, sizeof(path), "assets/gold/out%d.bin", i);
        std::vector<int8_t> t;
        if (!read_all(path, &t)) {
            fprintf(stderr, "missing fixture: %s (run scripts/fetch_gold.sh)\n", path);
            return 1;
        }
        f.raw_outputs.push_back(std::move(t));
    }
    for (int i = 0; i < 3; ++i) {
        f.out_zps.push_back(attrs["zps"][i].get<int32_t>());
        f.out_scales.push_back(attrs["scales"][i].get<float>());
    }

    StageContext ctx;
    Yolov5DetPost post;
    nlohmann::json p;
    p["conf_thres"] = 0.25;
    p["iou_thres"] = 0.45;
    p["classes"] = {"head", "helmet"};
    if (post.Init(p, ctx) != 0) return 1;
    if (!post.Process(f)) return 1;

    nlohmann::json out = nlohmann::json::array();
    for (const Detection& d : f.detections) {
        out.push_back({{"label", d.label}, {"score", d.score},
                       {"box", {d.x1, d.y1, d.x2, d.y2}}});
    }
    printf("%s\n", out.dump().c_str());
    return 0;
}
