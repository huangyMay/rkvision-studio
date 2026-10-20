// Task 7: infer stage（host 桩路径）+ 金张量接线
#include "core/frame.hpp"
#include "core/registry.hpp"
#include "stages/infer_rknn.hpp"

#include <cstdio>
#include <fstream>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

static std::streamoff file_size(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f.good()) return -1;
    f.seekg(0, std::ios::end);
    return f.tellg();
}

int main() {
    // 金张量固件（scripts/fetch_gold.sh 拷入 assets/gold/）
    std::streamoff sz0 = file_size("assets/gold/out0.bin");
    if (sz0 <= 0) {
        fprintf(stderr, "fixtures missing: run scripts/fetch_gold.sh first\n");
        return 1;
    }

    PipelineSpec spec;
    spec.version = 1;
    ModelSpec det;
    det.family = "yolov5-det";
    det.file = "assets/models/helmet_detect_my.rknn";
    spec.models["det"] = det;

    StageContext ctx;
    ctx.spec = &spec;
    ctx.project_root = ".";

    InferStage inf;
    nlohmann::json p;
    p["model"] = "det";
    p["stub_dir"] = "assets/gold";
    p["n_outputs"] = 3;
    CHECK(inf.Init(p, ctx) == 0);

    Frame f;
    f.image.create(640, 640, CV_8UC3);   // 桩不看内容
    CHECK(inf.Process(f) == true);
    CHECK(f.raw_outputs.size() == 3);
    CHECK((std::streamoff)f.raw_outputs[0].size() == sz0);   // out0 字节数与文件一致
    CHECK(f.out_zps.size() == 3);
    CHECK(f.out_scales.size() == 3);

    // attrs.json 缺失时默认 zp=0/scale=1（不致命）
    CHECK(inf.Process(f) == true);

    printf("test_infer_stub: all checks passed\n");
    return 0;
}
