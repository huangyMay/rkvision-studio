#include "stages/sink_image_out.hpp"

#include <cstdio>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/registry.hpp"

REGISTER_STAGE("image_out", ImageOutSink)

static void MkdirP(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' && i > 0) mkdir(cur.c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

int ImageOutSink::Init(const nlohmann::json& params, const StageContext& ctx) {
    if (!params.is_null()) {
        out_dir_ = params.value("path", "output");
        draw_ = params.value("draw", true);
        save_json_ = params.value("save_json", true);
    }
    if (!out_dir_.empty() && out_dir_[0] != '/' && !ctx.project_root.empty()) {
        out_dir_ = ctx.project_root + "/" + out_dir_;
    }
    if (out_dir_.empty()) {
        fprintf(stderr, "image_out: params.path is required\n");
        return -1;
    }
    MkdirP(out_dir_);
    return 0;
}

bool ImageOutSink::Process(Frame& frame) {
    // 画框永远画在原图坐标系上（letterbox 后的 frame.image 是 640 方图 RGB，不能直接用）
    cv::Mat canvas = frame.orig_image.empty() ? frame.image : frame.orig_image;
    if (draw_ && !frame.detections.empty()) {
        canvas = canvas.clone();
        for (const Detection& d : frame.detections) {
            cv::Point p1((int)d.x1, (int)d.y1), p2((int)d.x2, (int)d.y2);
            cv::rectangle(canvas, p1, p2, cv::Scalar(0, 255, 0), 2);
            char label[64];
            snprintf(label, sizeof(label), "%s %.2f", d.label.c_str(), d.score);
            cv::putText(canvas, label, cv::Point((int)d.x1, std::max(0, (int)d.y1 - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }

    std::string base = out_dir_ + "/" + (frame.image_name.empty() ? "frame" : frame.image_name);
    if (!cv::imwrite(base + "_result.jpg", canvas)) {
        fprintf(stderr, "image_out: imwrite failed: %s_result.jpg\n", base.c_str());
        return false;
    }

    if (save_json_) {
        nlohmann::json detections = nlohmann::json::array();
        for (const Detection& d : frame.detections) {
            detections.push_back({
                {"label", d.label},
                {"score", d.score},
                {"box", {d.x1, d.y1, d.x2, d.y2}},
            });
        }
        nlohmann::json j = {
            {"image", frame.image_name},
            {"detections", detections},
            {"timing", frame.timing},
        };
        std::ofstream f(base + "_result.json");
        if (!f.good()) {
            fprintf(stderr, "image_out: cannot write %s_result.json\n", base.c_str());
            return false;
        }
        f << j.dump(2);
    }
    return true;
}
