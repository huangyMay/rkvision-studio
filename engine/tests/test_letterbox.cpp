// Task 6: letterbox 数值复刻的行为规格
// 数值锚点来自 $DEMO/helmet_detect.cpp:105-119 及其注释（1200x798 -> new_h=426）
#include "core/frame.hpp"
#include "core/registry.hpp"
#include "stages/preprocess_letterbox.hpp"

#include <cstdio>

#include <opencv2/imgproc.hpp>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

namespace {

cv::Mat make_gradient(int cols, int rows) {
    cv::Mat m(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            m.at<cv::Vec3b>(y, x) = cv::Vec3b((x * 7) % 256, (y * 5) % 256, (x + y) % 256);
        }
    return m;
}

} // namespace

int main() {
    StageContext ctx;

    // ① 1200x798（宽x高）→ scale=640/1200，new_h=lround(798*scale)==426（demo 注释实证）
    {
        LetterboxStage st;
        nlohmann::json p; p["size"] = 640; p["pad_value"] = 255;
        CHECK(st.Init(p, ctx) == 0);
        Frame f;
        f.image = make_gradient(1200, 798);
        f.orig_w = 1200; f.orig_h = 798;
        CHECK(st.Process(f) == true);
        CHECK(f.image.cols == 640 && f.image.rows == 640);
        CHECK(f.orig_w == 1200 && f.orig_h == 798);          // 覆写前捕获
        CHECK(f.scale == 640.f / 1200.f);
        CHECK(f.pad_l == 0);
        CHECK(f.pad_t == 107);                                // (640-426)/2
        // 白边抽查：上边框（pad_t=107）、下边框（640-426-107=107）；此纵横比 pad_l=0 无左右边
        cv::Vec3b top = f.image.at<cv::Vec3b>(0, 320);
        cv::Vec3b bottom = f.image.at<cv::Vec3b>(639, 320);
        CHECK(top[0] == 255 && top[1] == 255 && top[2] == 255);
        CHECK(bottom[0] == 255 && bottom[1] == 255 && bottom[2] == 255);
        // 中心区与 cvtColor+resize 结果一致（接线正确性抽查）
        cv::Mat rgb, resized;
        cv::cvtColor(make_gradient(1200, 798), rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(640, 426), 0, 0, cv::INTER_LINEAR);
        cv::Vec3b got = f.image.at<cv::Vec3b>(107 + 100, 300);
        cv::Vec3b want = resized.at<cv::Vec3b>(100, 300);
        CHECK(got[0] == want[0] && got[1] == want[1] && got[2] == want[2]);
    }

    // ② 640x640 输入 → scale=1.0、pad 全 0
    {
        LetterboxStage st;
        nlohmann::json p; p["size"] = 640; p["pad_value"] = 255;
        CHECK(st.Init(p, ctx) == 0);
        Frame f;
        f.image = make_gradient(640, 640);
        CHECK(st.Process(f) == true);
        CHECK(f.scale == 1.0f);
        CHECK(f.pad_l == 0 && f.pad_t == 0);
    }

    // ③ 默认参数：size=640、pad_value=255
    {
        LetterboxStage st;
        CHECK(st.Init(nlohmann::json{}, ctx) == 0);
        Frame f;
        f.image = make_gradient(300, 200);
        CHECK(st.Process(f) == true);
        CHECK(f.image.cols == 640);
        cv::Vec3b c = f.image.at<cv::Vec3b>(5, 5);
        CHECK(c[0] == 255 && c[1] == 255 && c[2] == 255);
    }

    printf("test_letterbox: all checks passed\n");
    return 0;
}
