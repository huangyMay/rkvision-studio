// Task 2: load_spec 的行为规格（plain assert + ctest）
#include "core/spec.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

#define CHECK_THROWS_MSG(expr, needle) do { \
    bool threw_ = false; std::string msg_; \
    try { expr; } catch (const std::exception& e) { threw_ = true; msg_ = e.what(); } \
    if (!threw_) { fprintf(stderr, "expected throw at line %d\n", __LINE__); return 1; } \
    if (msg_.find(needle) == std::string::npos) { \
        fprintf(stderr, "line %d: what()='%s' lacks '%s'\n", __LINE__, msg_.c_str(), needle); return 1; } \
} while(0)

static void write_file(const std::string& path, const char* text) {
    std::ofstream f(path.c_str()); f << text;
}

static const char* kValid = R"({
  "version": 1,
  "name": "helmet_det_demo",
  "target": { "platform": "rk3576", "runtime": "android-native", "core_mask": "auto" },
  "models": {
    "det": {
      "family": "yolov5-det",
      "file": "models/helmet_detect_my.rknn",
      "input": { "name": "images", "shape": [1,3,640,640], "layout": "NCHW", "dtype": "int8" }
    }
  },
  "pipeline": [
    { "id": "src",  "plugin": "image_dir",  "params": { "path": "assets/test_images", "loop": false } },
    { "id": "pre",  "plugin": "letterbox",  "params": { "size": 640, "pad_value": 255 } },
    { "id": "inf",  "plugin": "infer",      "model": "det" },
    { "id": "post", "plugin": "yolov5_det", "params": { "conf_thres": 0.25, "iou_thres": 0.45,
                                                        "classes": ["head", "helmet"] } },
    { "id": "out",  "plugin": "image_out",  "params": { "path": "output", "draw": true, "save_json": true } }
  ]
})";

int main() {
    // ① 合法工程包：逐字段断言
    write_file("/tmp/rvs_test_valid.json", kValid);
    PipelineSpec spec = load_spec("/tmp/rvs_test_valid.json");
    CHECK(spec.version == 1);
    CHECK(spec.name == "helmet_det_demo");
    CHECK(spec.core_mask == "auto");
    CHECK(spec.models.size() == 1);
    CHECK(spec.models.count("det") == 1);
    CHECK(spec.models.at("det").family == "yolov5-det");
    CHECK(spec.models.at("det").file == "models/helmet_detect_my.rknn");
    CHECK(spec.pipeline.size() == 5);
    CHECK(spec.pipeline[0].id == "src");
    CHECK(spec.pipeline[0].plugin == "image_dir");
    CHECK(spec.pipeline[1].params["pad_value"] == 255);
    CHECK(spec.pipeline[2].plugin == "infer");
    CHECK(spec.pipeline[2].model == "det");
    CHECK(spec.pipeline[3].params["classes"].size() == 2);
    CHECK(spec.pipeline[3].params["conf_thres"].get<double>() == 0.25);
    CHECK(spec.pipeline[4].plugin == "image_out");

    // ② 缺 version → 抛错，what() 含 "version"
    write_file("/tmp/rvs_test_noversion.json", R"({ "name": "x", "pipeline": [] })");
    CHECK_THROWS_MSG(load_spec("/tmp/rvs_test_noversion.json"), "version");

    // ③ stage 缺 plugin → 抛错，what() 含 "plugin"
    write_file("/tmp/rvs_test_noplugin.json", R"({
        "version": 1, "name": "x", "models": {}, "pipeline": [ { "id": "a" } ] })");
    CHECK_THROWS_MSG(load_spec("/tmp/rvs_test_noplugin.json"), "plugin");

    // ④ infer 引用不存在的 model 键 → 抛错含该键名
    write_file("/tmp/rvs_test_badmodel.json", R"({
        "version": 1, "name": "x", "models": {},
        "pipeline": [ { "id": "inf", "plugin": "infer", "model": "nope" } ] })");
    CHECK_THROWS_MSG(load_spec("/tmp/rvs_test_badmodel.json"), "nope");

    // ⑤ 版本不支持 → 抛错
    write_file("/tmp/rvs_test_badversion.json",
               R"({ "version": 99, "name": "x", "pipeline": [] })");
    CHECK_THROWS_MSG(load_spec("/tmp/rvs_test_badversion.json"), "version");

    // ⑥ 文件不存在 → 抛错
    CHECK_THROWS_MSG(load_spec("/tmp/rvs_no_such_file.json"), "cannot open");

    // ⑦ stage id 重复 → 抛错（executor 按 id 聚合计时，重复会混淆）
    write_file("/tmp/rvs_test_dup_id.json", R"({
        "version": 1, "name": "x",
        "pipeline": [
            { "id": "a", "plugin": "p1" },
            { "id": "a", "plugin": "p2" }
        ] })");
    CHECK_THROWS_MSG(load_spec("/tmp/rvs_test_dup_id.json"), "duplicate");

    printf("test_spec: all checks passed\n");
    return 0;
}
