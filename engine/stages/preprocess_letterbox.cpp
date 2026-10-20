#include "stages/preprocess_letterbox.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "core/registry.hpp"

REGISTER_STAGE("letterbox", LetterboxStage)

int LetterboxStage::Init(const nlohmann::json& params, const StageContext&) {
    if (!params.is_null()) {
        size_ = params.value("size", 640);
        pad_value_ = params.value("pad_value", 255);
    }
    if (size_ <= 0) {
        fprintf(stderr, "letterbox: params.size must be positive\n");
        return -1;
    }
    return 0;
}

bool LetterboxStage::Process(Frame& frame) {
    if (frame.image.empty()) {
        fprintf(stderr, "letterbox: frame.image is empty\n");
        return false;
    }
    const int iw = size_, ih = size_;
    const int img_w = frame.image.cols, img_h = frame.image.rows;

    // 原图尺寸在覆写前捕获（坐标还原的基准）；原图本体留给 sink 画框
    frame.orig_w = img_w;
    frame.orig_h = img_h;
    frame.orig_image = frame.image.clone();

    float scale_w = (float)iw / img_w;
    float scale_h = (float)ih / img_h;
    float scale = scale_w < scale_h ? scale_w : scale_h;
    // 官方 letter_box 用 std::round（798*0.5333=425.6 -> 426），截断会差 1px 导致张量全图微差
    int new_w = lround(img_w * scale);
    int new_h = lround(img_h * scale);
    int dw = (iw - new_w) / 2;
    int dh = (ih - new_h) / 2;

    cv::Mat img_rgb, img_resized, img_letterbox;
    cv::cvtColor(frame.image, img_rgb, cv::COLOR_BGR2RGB);
    cv::resize(img_rgb, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(img_resized, img_letterbox, dh, ih - new_h - dh,
                       dw, iw - new_w - dw, cv::BORDER_CONSTANT,
                       cv::Scalar(pad_value_, pad_value_, pad_value_));

    frame.image = img_letterbox;
    frame.scale = scale;
    frame.pad_l = dw;
    frame.pad_t = dh;
    return true;
}
