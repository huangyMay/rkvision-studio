// MJPEG sink：内置 HTTP 服务，把帧以 multipart/x-mixed-replace 直出。
// 传输链：板上引擎 :8000/stream → adb forward tcp:本地 → 浏览器 <img>。
// 预览画布 = 原图坐标系（与 image_out 同规则）；无检出时回写干净原图。
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cstdint>

#include "core/stage.hpp"

class MjpegSink : public IStage {
public:
    MjpegSink() = default;
    ~MjpegSink() override;

    int Init(const nlohmann::json& params, const StageContext& ctx) override;
    bool Process(Frame& frame) override;
    const StageMeta& Meta() const override {
        static StageMeta m{"mjpeg", false, true, false};
        return m;
    }

private:
    void ServerLoop();

    int port_ = 8000;
    std::string path_ = "/stream";
    bool draw_ = true;
    int fps_ = 10;
    int listen_fd_ = -1;
    std::thread server_;
    std::atomic<bool> running_{false};

    std::mutex mtx_;
    std::vector<uint8_t> jpeg_;   // 最新一帧（JPEG 编码后）
    bool has_frame_ = false;
};
