// V4L2 摄像头源：mmap 采集，MJPEG 直接解码、YUYV 转 BGR。
// 板上无内置 sensor 时，插 USB UVC 摄像头即为 /dev/video*；节点名可用
// /sys/class/video4linux/videoN/name 确认。
#pragma once
#include <string>
#include <vector>

#include "core/stage.hpp"

class V4l2Source : public IStage {
public:
    V4l2Source() = default;
    ~V4l2Source() override;

    int Init(const nlohmann::json& params, const StageContext& ctx) override;
    bool Process(Frame& frame) override;
    const StageMeta& Meta() const override {
        static StageMeta m{"v4l2", true, false, false};
        return m;
    }

private:
    bool SetupFormat();
    bool QueueAll();

    std::string dev_ = "/dev/video0";
    int width_ = 640, height_ = 480;
    bool prefer_mjpeg_ = true;

    int fd_ = -1;
    bool streaming_ = false;
    uint32_t pixfmt_ = 0;
    struct Buffer { void* start; size_t len; };   // C++11: 无默认成员初始化才是聚合
    std::vector<Buffer> buffers_;
};
