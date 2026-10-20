// 板端实现：librknnrt。数值处理与 $DEMO/helmet_detect.cpp 一致：
//   - 模型读入内存后 rknn_init（与官方 .o 的文件加载路径一致）
//   - 输入 uint8 NHWC RGB 透传（RKNN_TENSOR_UINT8 + RKNN_TENSOR_NHWC，量化由 NPU 完成）
//   - 输出 int8（want_float=0），zp/scale 由 RKNN_QUERY_OUTPUT_ATTR 查询
//   - target.core_mask != "auto" 时尝试 rknn_set_core_mask（失败打日志不中断，保持 demo 行为）
#include "stages/infer_rknn.hpp"

#include <cstdio>
#include <cstdlib>

#include "core/registry.hpp"
#include "rknn_api.h"

REGISTER_STAGE("infer", InferStage)

int InferStage::Init(const nlohmann::json& params, const StageContext& ctx) {
    model_key_ = params.is_null() ? "det" : params.value("model", "det");
    if (!ctx.spec || ctx.spec->models.count(model_key_) != 1) {
        fprintf(stderr, "infer: unknown model key '%s'\n", model_key_.c_str());
        return -1;
    }
    model_path_ = ctx.spec->models.at(model_key_).file;
    if (!model_path_.empty() && model_path_[0] != '/' && !ctx.project_root.empty()) {
        model_path_ = ctx.project_root + "/" + model_path_;
    }

    // 读模型进内存再 rknn_init
    FILE* fp = fopen(model_path_.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "infer: cannot open model '%s'\n", model_path_.c_str());
        return -2;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void* model = malloc(sz);
    if (!model) { fclose(fp); return -3; }
    if (fread(model, 1, sz, fp) != (size_t)sz) {
        free(model); fclose(fp); return -4;
    }
    fclose(fp);

    int ret = rknn_init(&ctx_, model, sz, 0, NULL);
    free(model);
    if (ret < 0) {
        fprintf(stderr, "infer: rknn_init failed ret=%d (%s)\n", ret, model_path_.c_str());
        return -5;
    }

    fprintf(stderr, "[infdbg] rknn_init ok\n");
    fflush(stderr);
    // 输出数量：从模型查询（yolov8 等多输出模型 ≠ 3）
    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    if (rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) == 0) {
        n_outputs_ = (int)io_num.n_output;
        fprintf(stderr, "infer: n_outputs=%d\n", n_outputs_);
        fflush(stderr);
    }

    // 核亲和（可选）：RK3576 双核取 0x1/0x2/0x3；"auto" 保持默认
    const std::string& mask = ctx.spec->core_mask;
    if (!mask.empty() && mask != "auto") {
        rknn_core_mask core = (rknn_core_mask)strtoul(mask.c_str(), NULL, 0);
        ret = rknn_set_core_mask(ctx_, core);
        if (ret < 0) {
            fprintf(stderr, "infer: rknn_set_core_mask(%s) failed ret=%d, keep default\n",
                    mask.c_str(), ret);
        }
    }

    // 查询输出张量量化参数（zp/scale），供后处理反量化使用
    std::vector<rknn_tensor_attr> attrs(n_outputs_);
    for (int i = 0; i < n_outputs_; ++i) {
        memset(&attrs[i], 0, sizeof(rknn_tensor_attr));
        attrs[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            fprintf(stderr, "infer: rknn_query OUTPUT_ATTR index=%d failed ret=%d\n", i, ret);
            return -6;
        }
    }
    loaded_ = true;
    return 0;
}

bool InferStage::Process(Frame& frame) {
    if (!loaded_) {
        fprintf(stderr, "infer: model not loaded\n");
        return false;
    }

    // 输入：uint8 NHWC RGB 透传
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = frame.image.cols * frame.image.rows * 3;
    inputs[0].buf = frame.image.data;
    int ret = rknn_inputs_set(ctx_, 1, inputs);
    if (ret < 0) {
        fprintf(stderr, "infer: rknn_inputs_set failed ret=%d\n", ret);
        return false;
    }

    ret = rknn_run(ctx_, NULL);
    if (ret < 0) {
        fprintf(stderr, "infer: rknn_run failed ret=%d\n", ret);
        return false;
    }

    std::vector<rknn_output> outputs(n_outputs_);
    memset(outputs.data(), 0, sizeof(rknn_output) * outputs.size());
    for (int i = 0; i < n_outputs_; ++i) {
        outputs[i].want_float = 0;
        outputs[i].index = i;
    }
    ret = rknn_outputs_get(ctx_, n_outputs_, outputs.data(), NULL);
    if (ret < 0) {
        fprintf(stderr, "infer: rknn_outputs_get failed ret=%d\n", ret);
        return false;
    }

    frame.raw_outputs.assign(n_outputs_, {});
    frame.out_zps.assign(n_outputs_, 0);
    frame.out_scales.assign(n_outputs_, 0.f);
    for (int i = 0; i < n_outputs_; ++i) {
        const int8_t* p = (const int8_t*)outputs[i].buf;
        frame.raw_outputs[i].assign(p, p + outputs[i].size);
        // 与 demo 同序：zp/scale 在 Init 查询；这里再查一次以保持 Process 自洽
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        if (rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) == 0) {
            frame.out_zps[i] = attr.zp;
            frame.out_scales[i] = attr.scale;
        }
        rknn_outputs_release(ctx_, 1, &outputs[i]);
    }
    return true;
}
