// Frame：流水线上一帧的完整数据包（图像 + 中间量 + 结果）
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct Detection {
    std::string label;
    float score = 0.f;
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;  // 原图坐标系
};

struct Frame {
    cv::Mat image;                 // source: BGR 原图；letterbox 后替换为 RGB 方图（Task 6）
    cv::Mat orig_image;            // letterbox 覆写前的原图（sink 画框用；不带时 sink 退回 image）
    std::string image_name;        // 源图名（去扩展名），sink 输出命名用
    int orig_w = 0, orig_h = 0;    // 原图尺寸（letterbox 覆写 image 前捕获）

    // letterbox 中间量：preprocess 写、postprocess 读（坐标还原用）
    float scale = 1.f;
    int pad_l = 0, pad_t = 0;

    // infer 产物：int8 原始输出张量 + 反量化参数（Task 7 写、Task 8 读）
    std::vector<std::vector<int8_t>> raw_outputs;
    std::vector<int32_t> out_zps;
    std::vector<float> out_scales;

    std::vector<Detection> detections;
    std::map<std::string, double> timing;   // 分段耗时 ms，key = stage id
};
