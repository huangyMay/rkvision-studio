#include "stages/source_v4l2.hpp"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/registry.hpp"

REGISTER_STAGE("v4l2", V4l2Source)

namespace {

int XioCtl(int fd, unsigned long req, void* arg) {
    int r;
    do {
        r = ::ioctl(fd, req, arg);
    } while (r < 0 && errno == EINTR);
    return r;
}

} // namespace

V4l2Source::~V4l2Source() {
    if (streaming_) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        XioCtl(fd_, VIDIOC_STREAMOFF, &type);
    }
    for (auto& b : buffers_) {
        if (b.start) munmap(b.start, b.len);
    }
    if (fd_ >= 0) ::close(fd_);
}

int V4l2Source::Init(const nlohmann::json& params, const StageContext&) {
    if (!params.is_null()) {
        dev_ = params.value("device", "/dev/video0");
        width_ = params.value("width", 640);
        height_ = params.value("height", 480);
        prefer_mjpeg_ = params.value("prefer_mjpeg", true);
    }

    fd_ = ::open(dev_.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        fprintf(stderr, "v4l2: cannot open %s errno=%d\n", dev_.c_str(), errno);
        return -1;
    }
    v4l2_capability cap{};
    if (XioCtl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        fprintf(stderr, "v4l2: %s is not a V4L2 device\n", dev_.c_str());
        return -2;
    }
    if (!SetupFormat()) return -3;
    if (!QueueAll()) return -4;

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (XioCtl(fd_, VIDIOC_STREAMON, &type) < 0) {
        fprintf(stderr, "v4l2: STREAMON failed errno=%d\n", errno);
        return -5;
    }
    streaming_ = true;
    return 0;
}

bool V4l2Source::SetupFormat() {
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = (uint32_t)width_;
    fmt.fmt.pix.height = (uint32_t)height_;
    fmt.fmt.pix.pixelformat = prefer_mjpeg_ ? V4L2_PIX_FMT_MJPEG : V4L2_PIX_FMT_YUYV;
    if (XioCtl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        fprintf(stderr, "v4l2: S_FMT failed errno=%d\n", errno);
        return false;
    }
    width_ = (int)fmt.fmt.pix.width;
    height_ = (int)fmt.fmt.pix.height;
    pixfmt_ = fmt.fmt.pix.pixelformat;
    return true;
}

bool V4l2Source::QueueAll() {
    v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (XioCtl(fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
        fprintf(stderr, "v4l2: REQBUFS failed errno=%d\n", errno);
        return false;
    }
    for (uint32_t i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (XioCtl(fd_, VIDIOC_QBUF, &buf) < 0) return false;
        v4l2_buffer query{};
        query.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        query.memory = V4L2_MEMORY_MMAP;
        query.index = i;
        if (XioCtl(fd_, VIDIOC_QUERYBUF, &query) < 0) return false;
        void* start = mmap(nullptr, query.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                           fd_, query.m.offset);
        if (start == MAP_FAILED) return false;
        buffers_.push_back(Buffer{start, query.length});
    }
    return true;
}

bool V4l2Source::Process(Frame& frame) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd_, &fds);
    timeval tv { 2, 0 };
    if (::select(fd_ + 1, &fds, nullptr, nullptr, &tv) <= 0) {
        fprintf(stderr, "v4l2: select timeout\n");
        return false;
    }
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (XioCtl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        fprintf(stderr, "v4l2: DQBUF failed errno=%d\n", errno);
        return false;
    }

    const uint8_t* data = (const uint8_t*)buffers_[buf.index].start;
    size_t bytes = buf.bytesused;
    cv::Mat img;
    if (pixfmt_ == V4L2_PIX_FMT_MJPEG) {
        std::vector<uchar> raw(data, data + bytes);
        img = cv::imdecode(raw, cv::IMREAD_COLOR);
    } else {   // YUYV
        cv::Mat yuyv((int)height_, (int)width_, CV_8UC2, (void*)data);
        cv::cvtColor(yuyv, img, cv::COLOR_YUV2BGR_YUYV);
    }
    XioCtl(fd_, VIDIOC_QBUF, &buf);

    if (img.empty()) {
        fprintf(stderr, "v4l2: decode failed (%zu bytes, fmt=%08x)\n", bytes, pixfmt_);
        return true;   // 丢一帧不断流
    }
    frame.image = img;
    frame.orig_w = img.cols;
    frame.orig_h = img.rows;
    return true;
}
