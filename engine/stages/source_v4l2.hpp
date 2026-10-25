// V4L2 采集源：MMAP + 最新帧优先，支持单平面与多平面（RK HDMI-RX 是 mplane）。
// 支持像素格式：BGR24 / RGB24 / NV12 / NV16 / YUYV / UYVY / MJPEG。
// 板上无内置 sensor 时，插 USB UVC 摄像头即为 /dev/video*；节点名可用
// /sys/class/video4linux/videoN/name 确认。
#pragma once
#include <linux/videodev2.h>

#include <cstdint>
#include <ctime>
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
    struct Buffer {
        void* start[3];
        size_t len[3];
        int planes;
        Buffer() : planes(0) {
            for (int i = 0; i < 3; ++i) { start[i] = nullptr; len[i] = 0; }
        }
    };

    uint32_t BufType() const;
    bool SetupFormat();
    bool TryFormat(uint32_t pf, int w, int h, bool exact);
    bool QueueAll();
    bool QueueBuffer(int index);
    bool Dequeue(int* index, size_t* used);
    bool Convert(const Buffer& b, size_t used, cv::Mat* out);
    void SetFps();

    std::string dev_ = "/dev/video0";
    int width_ = 640, height_ = 480;
    int fps_ = 0;
    bool prefer_mjpeg_ = false;
    bool forced_format_ = false;
    uint32_t forced_pixfmt_ = 0;
    int timeout_ms_ = 1000;
    int max_timeouts_ = 3;

    int fd_ = -1;
    bool streaming_ = false;
    bool mplane_ = false;
    uint32_t pixfmt_ = 0;
    int num_planes_ = 1;
    int stride_[3];
    v4l2_format fmt_;
    time_t started_at_ = 0;
    long frame_no_ = 0;

    std::vector<Buffer> buffers_;
};
