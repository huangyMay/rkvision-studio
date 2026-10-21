// Task 10: RunProject 的行为规格
#include "core/run_project.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

static void write_file(const std::string& path, const char* text) {
    std::ofstream f(path.c_str()); f << text;
}

int main() {
    // ① 最小可跑工程：image_dir → letterbox → image_out，1 帧
    if (system("rm -rf /tmp/rvs_test_proj && mkdir -p /tmp/rvs_test_proj/imgs") != 0) return 1;
    {
        std::ofstream img("/tmp/rvs_test_proj/imgs/a.ppm");
        img << "P6\n8 6\n255\n";
        for (int i = 0; i < 8 * 6; ++i) img << (char)255 << (char)0 << (char)0;
    }
    write_file("/tmp/rvs_test_proj/pipeline.json", R"({
      "version": 1, "name": "smoke",
      "pipeline": [
        { "id": "src", "plugin": "image_dir", "params": { "path": "imgs" } },
        { "id": "pre", "plugin": "letterbox", "params": { "size": 64, "pad_value": 255 } },
        { "id": "out", "plugin": "image_out", "params": { "path": "out" } }
      ]
    })");
    std::string summary;
    int ret = RunProject("/tmp/rvs_test_proj", &summary);
    CHECK(ret == 0);
    CHECK(summary.find("\"frames\":1") != std::string::npos);
    CHECK(summary.find("e2e_avg_ms") != std::string::npos);
    std::ifstream out("/tmp/rvs_test_proj/out/a_result.jpg");
    CHECK(out.good());

    // ② 声明了模型但文件缺失 → 返回 3（RunProject 固定读 pipeline.json，覆写之）
    write_file("/tmp/rvs_test_proj/pipeline.json", R"({
      "version": 1, "name": "badmodel",
      "models": { "det": { "family": "yolov5-det", "file": "models/nope.rknn" } },
      "pipeline": [
        { "id": "src", "plugin": "image_dir", "params": { "path": "imgs" } },
        { "id": "inf", "plugin": "infer", "model": "det" },
        { "id": "out", "plugin": "image_out", "params": { "path": "out" } }
      ]
    })");
    if (system("mkdir -p /tmp/rvs_test_proj/models") != 0) return 1;
    ret = RunProject("/tmp/rvs_test_proj", &summary);
    CHECK(ret == 3);

    // ③ 工程目录不存在 → 返回 2
    ret = RunProject("/tmp/rvs_no_such_proj", &summary);
    CHECK(ret == 2);

    // ④ 整链集成：真实五级链（infer 走 host 桩 + 金张量），inf 不写 params（model 由 spec 注入）
    //    金张量缺失（CI/未 fetch）时与 gold_regression 同款 SKIP
    {
        std::ifstream probe("assets/gold/out0.bin");
        if (!probe.good()) {
            printf("SKIP: case4 needs assets/gold (run scripts/fetch_gold.sh for the real suite)\n");
            printf("test_main: all checks passed\n");
            return 0;
        }
    }
    if (system("rm -rf /tmp/rvs_test_full && mkdir -p /tmp/rvs_test_full/assets/test_images"
               " /tmp/rvs_test_full/assets/gold"
               " && cp assets/gold/out0.bin assets/gold/out1.bin assets/gold/out2.bin"
               " /tmp/rvs_test_full/assets/gold/") != 0) return 1;
    {
        std::ofstream img("/tmp/rvs_test_full/assets/test_images/b.ppm");
        img << "P6\n1200 798\n255\n";
        for (int i = 0; i < 1200 * 798; ++i) img << (char)128 << (char)64 << (char)32;
    }
    write_file("/tmp/rvs_test_full/pipeline.json", R"({
      "version": 1, "name": "full_chain",
      "models": { "det": { "family": "yolov5-det", "file": "models/any.rknn" } },
      "pipeline": [
        { "id": "src",  "plugin": "image_dir",  "params": { "path": "assets/test_images" } },
        { "id": "pre",  "plugin": "letterbox",  "params": { "size": 640, "pad_value": 255 } },
        { "id": "inf",  "plugin": "infer",      "model": "det" },
        { "id": "post", "plugin": "yolov5_det", "params": { "conf_thres": 0.25, "iou_thres": 0.45,
                                                            "classes": ["head", "helmet"] } },
        { "id": "out",  "plugin": "image_out",  "params": { "path": "output" } }
      ]
    })");
    if (system("mkdir -p /tmp/rvs_test_full/models && touch /tmp/rvs_test_full/models/any.rknn") != 0) return 1;
    ret = RunProject("/tmp/rvs_test_full", &summary);
    CHECK(ret == 0);                                   // C1: yolov5_det 插件必须已链入
    CHECK(summary.find("\"frames\":1") != std::string::npos);
    {
        std::ifstream jf("/tmp/rvs_test_full/output/b_result.json");
        CHECK(jf.good());                              // C2: inf 无 params 不崩（model 由 spec 注入）
    }

    printf("test_main: all checks passed\n");
    return 0;
}
