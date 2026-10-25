// MJPEG sink：内置 HTTP 服务，把帧以 multipart/x-mixed-replace 直出。
// 传输链：板上引擎 :8000/stream → adb forward tcp:本地 → 浏览器 <img>。
// 预览画布 = 原图坐标系（与 image_out 同规则）；无检出时回写干净原图。
// 端点：/stream、/snapshot.jpg、/health、/
#pragma once
#include <atomic>
#include <memory>
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

    // 连接线程只持有这份堆上状态，不碰 MjpegSink 自身：
    // 析构时客户端线程可能还没退出，捕获 this 会造成 use-after-free。
    struct Shared {
        std::mutex mtx;
        std::vector<uint8_t> jpeg;
        std::atomic<bool> running;
        std::atomic<int> clients;
        Shared() : running(false), clients(0) {}
    };

private:
    void ServerLoop(const std::shared_ptr<Shared>& st);

    int port_ = 8000;
    std::string path_ = "/stream";
    bool draw_ = true;
    int fps_ = 10;
    int max_clients_ = 4;
    int listen_fd_ = -1;
    std::thread server_;
    std::shared_ptr<Shared> state_;
};
