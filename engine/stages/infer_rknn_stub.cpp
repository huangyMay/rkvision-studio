// host 桩：不经 NPU，从 stub_dir 读金张量 out{i}.bin + attrs.json（zps/scales）。
// 用途：host 单测打通 postprocess 全链（Task 8/12 的金张量回归）。
#include "stages/infer_rknn.hpp"

#include <cstdio>
#include <fstream>

#include "core/registry.hpp"

REGISTER_STAGE("infer", InferStage)

int InferStage::Init(const nlohmann::json& params, const StageContext& ctx) {
    model_key_ = params.is_null() ? "det" : params.value("model", "det");
    if (!ctx.spec || ctx.spec->models.count(model_key_) != 1) {
        fprintf(stderr, "infer[stub]: unknown model key '%s'\n", model_key_.c_str());
        return -1;
    }
    stub_dir_ = params.value("stub_dir", "assets/gold");
    if (!stub_dir_.empty() && stub_dir_[0] != '/' && !ctx.project_root.empty()) {
        stub_dir_ = ctx.project_root + "/" + stub_dir_;
    }
    n_outputs_ = params.value("n_outputs", 3);
    return 0;
}

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

bool InferStage::Process(Frame& frame) {
    frame.raw_outputs.assign(n_outputs_, {});
    frame.out_zps.assign(n_outputs_, 0);
    frame.out_scales.assign(n_outputs_, 1.f);

    for (int i = 0; i < n_outputs_; ++i) {
        char path[512];
        snprintf(path, sizeof(path), "%s/out%d.bin", stub_dir_.c_str(), i);
        if (!read_all(path, &frame.raw_outputs[i])) {
            fprintf(stderr, "infer[stub]: cannot read '%s' (run scripts/fetch_gold.sh)\n", path);
            return false;
        }
    }

    // attrs.json: {"zps":[..], "scales":[..]}
    std::ifstream af(stub_dir_ + "/attrs.json");
    if (af.good()) {
        try {
            nlohmann::json a = nlohmann::json::parse(af);
            auto zps = a.value("zps", std::vector<int>{});
            auto scales = a.value("scales", std::vector<float>{});
            for (int i = 0; i < n_outputs_; ++i) {
                if (i < (int)zps.size()) frame.out_zps[i] = zps[i];
                if (i < (int)scales.size()) frame.out_scales[i] = scales[i];
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "infer[stub]: bad attrs.json: %s\n", e.what());
            return false;
        }
    }
    return true;
}
