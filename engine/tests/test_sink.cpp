// Task 9: image_out sink 的行为规格
#include "core/frame.hpp"
#include "core/registry.hpp"
#include "stages/sink_image_out.hpp"

#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include <nlohmann/json.hpp>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

int main() {
    if (system("rm -rf /tmp/rvs_test_sink && mkdir -p /tmp/rvs_test_sink") != 0) return 1;

    StageContext ctx;
    ctx.project_root = "/tmp/rvs_test_sink";   // 相对输出路径按工程包根解析（与真实用法一致）

    // ① 两个检测（含负坐标框 → clamp）→ 图与 JSON 落盘、JSON 可解析、坐标非负
    {
        ImageOutSink sink;
        nlohmann::json p;
        p["path"] = "out";
        CHECK(sink.Init(p, ctx) == 0);

        Frame f;
        f.image.create(80, 60, CV_8UC3);
        f.image_name = "img1";
        Detection d1;
        d1.label = "helmet"; d1.score = 0.9f;
        d1.x1 = 5; d1.y1 = 6; d1.x2 = 30; d1.y2 = 40;
        Detection d2;
        d2.label = "head"; d2.score = 0.5f;
        d2.x1 = -3; d2.y1 = 0; d2.x2 = 10; d2.y2 = 12;
        f.detections = {d1, d2};
        f.timing["s0"] = 1.5;
        CHECK(sink.Process(f) == true);

        std::ifstream jf("/tmp/rvs_test_sink/out/img1_result.json");
        CHECK(jf.good());
        nlohmann::json j = nlohmann::json::parse(jf);
        CHECK(j["detections"].size() == 2);
        CHECK(j["detections"][0]["label"] == "helmet");
        CHECK(std::abs(j["detections"][0]["score"].get<double>() - 0.9) < 1e-6);
        CHECK(j["detections"][0]["box"][0] == 5.0);
        CHECK(j["detections"][1]["box"][0] == -3.0);       // sink 忠实写出；clamp 是后处理的职责
        CHECK(j["timing"].contains("s0"));
        std::ifstream imgf("/tmp/rvs_test_sink/out/img1_result.jpg");
        CHECK(imgf.good());
    }

    // ② 零检测 → JSON detections 为空数组、图仍写出（Review Focus 2）
    {
        ImageOutSink sink;
        nlohmann::json p;
        p["path"] = "out";
        CHECK(sink.Init(p, ctx) == 0);
        Frame f;
        f.image.create(32, 32, CV_8UC3);
        f.image_name = "img2";
        CHECK(sink.Process(f) == true);
        std::ifstream jf("/tmp/rvs_test_sink/out/img2_result.json");
        nlohmann::json j = nlohmann::json::parse(jf);
        CHECK(j["detections"].empty());
        std::ifstream imgf("/tmp/rvs_test_sink/out/img2_result.jpg");
        CHECK(imgf.good());
    }

    // ③ 注册名可查
    CHECK(StageRegistry::Instance().Create("image_out") != nullptr);

    // ④ frame 带原图（letterbox 后）时画框画在原图上（Review: 结果图尺寸=原图）
    {
        ImageOutSink sink;
        nlohmann::json p;
        p["path"] = "out";
        CHECK(sink.Init(p, ctx) == 0);
        Frame f;
        f.orig_image.create(60, 80, CV_8UC3);   // 原图 80 宽 x 60 高
        f.image.create(640, 640, CV_8UC3);      // letterbox 后的方图
        f.image_name = "img3";
        Detection d;
        d.label = "helmet"; d.score = 0.9f;
        d.x1 = 5; d.y1 = 5; d.x2 = 50; d.y2 = 40;
        f.detections = {d};
        CHECK(sink.Process(f) == true);
        cv::Mat saved = cv::imread("/tmp/rvs_test_sink/out/img3_result.jpg");
        CHECK(!saved.empty());
        CHECK(saved.cols == 80 && saved.rows == 60);   // 画在原图坐标系上
    }

    printf("test_sink: all checks passed\n");
    return 0;
}
