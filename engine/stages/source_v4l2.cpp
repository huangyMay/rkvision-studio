#include "stages/source_v4l2.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/registry.hpp"

REGISTER_STAGE("v4l2", V4l2Source)

namespace {

const int kMaxPlanes = 3;
const int kBufferCount = 4;

int XioCtl(int fd, unsigned long req, void* arg) {
    int r;
    do {
        r = ::ioctl(fd, req, arg);
    } while (r < 0 && errno == EINTR);
    return r;
}

const char* FourccStr(uint32_t f, char out[5]) {
    out[0] = (char)(f & 0xff);
    out[1] = (char)((f >> 8) & 0xff);
    out[2] = (char)((f >> 16) & 0xff);
    out[3] = (char)((f >> 24) & 0xff);
    out[4] = '\0';
    return out;
}

// bytesperline 在部分驱动里是 0；按格式给出安全默认值
int DefaultStride(uint32_t pf, int w) {
    switch (pf) {
    case V4L2_PIX_FMT_BGR24:
    case V4L2_PIX_FMT_RGB24:
        return w * 3;
    case V4L2_PIX_FMT_YUYV:
    case V4L2_PIX_FMT_UYVY:
        return w * 2;
    case V4L2_PIX_FMT_NV12:
        return w;
    default:
        return w * 2;
    }
}

// 名字或四字符码都接受；大小写不敏感（"mjpeg"/"MJPG"/"NV12"）
bool ParsePixelFormat(const std::string& raw, uint32_t* out) {    std::string s;
    for (size_t i = 0; i < raw.size(); ++i) {
        char c = raw[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        s += c;
    }
    if (s == "mjpeg" || s == "mjpg") { *out = V4L2_PIX_FMT_MJPEG; return true; }
    if (s == "jpeg") { *out = V4L2_PIX_FMT_JPEG; return true; }
    if (s == "yuyv" || s == "yuy2") { *out = V4L2_PIX_FMT_YUYV; return true; }
    if (s == "uyvy") { *out = V4L2_PIX_FMT_UYVY; return true; }
    if (s == "nv12") { *out = V4L2_PIX_FMT_NV12; return true; }
    if (s == "nv16") { *out = V4L2_PIX_FMT_NV16; return true; }
    if (s == "bgr24" || s == "bgr3") { *out = V4L2_PIX_FMT_BGR24; return true; }
    if (s == "rgb24" || s == "rgb3") { *out = V4L2_PIX_FMT_RGB24; return true; }
    if (raw.size() == 4) {
        *out = v4l2_fourcc(raw[0], raw[1], raw[2], raw[3]);
        return true;
    }
    return false;
}

} // namespace

V4l2Source::~V4l2Source() {
    if (streaming_) {
        enum v4l2_buf_type type = (enum v4l2_buf_type)BufType();
        XioCtl(fd_, VIDIOC_STREAMOFF, &type);
    }
    for (size_t i = 0; i < buffers_.size(); ++i) {
        for (int p = 0; p < buffers_[i].planes; ++p) {
            if (buffers_[i].start[p]) munmap(buffers_[i].start[p], buffers_[i].len[p]);
        }
    }
    if (fd_ >= 0) ::close(fd_);
}

uint32_t V4l2Source::BufType() const {
    return mplane_ ? V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE : V4L2_BUF_TYPE_VIDEO_CAPTURE;
}

int V4l2Source::Init(const nlohmann::json& params, const StageContext&) {
    if (!params.is_null()) {
        dev_ = params.value("device", "/dev/video0");
        width_ = params.value("width", 640);
        height_ = params.value("height", 480);
        fps_ = params.value("fps", 0);
        prefer_mjpeg_ = params.value("prefer_mjpeg", false);
        timeout_ms_ = params.value("timeout_ms", 1000);
        max_timeouts_ = params.value("max_timeouts", 3);
        if (params.contains("format")) {
            std::string f = params["format"].get<std::string>();
            if (!ParsePixelFormat(f, &forced_pixfmt_)) {
                fprintf(stderr, "v4l2: unknown format '%s' (try bgr24/nv12/yuyv/mjpeg or fourcc)\n",
                        f.c_str());
                return -1;
            }
            forced_format_ = true;
        }
    }
    if (timeout_ms_ <= 0) timeout_ms_ = 1000;
    if (max_timeouts_ <= 0) max_timeouts_ = 1;

    fd_ = ::open(dev_.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        fprintf(stderr, "v4l2: cannot open %s errno=%d\n", dev_.c_str(), errno);
        return -2;
    }
    v4l2_capability cap{};
    if (XioCtl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        fprintf(stderr, "v4l2: %s is not a V4L2 device\n", dev_.c_str());
        return -3;
    }
    // device_caps 才是"这个节点能做什么"；capabilities 在 mplane 设备上可能不含 MPLANE
    const uint32_t caps = cap.device_caps ? cap.device_caps : cap.capabilities;
    const bool has_mp = (caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) != 0;
    const bool has_sp = (caps & V4L2_CAP_VIDEO_CAPTURE) != 0;
    if (!has_mp && !has_sp) {
        fprintf(stderr, "v4l2: %s has no video capture capability\n", dev_.c_str());
        return -3;
    }
    mplane_ = has_mp;

    // width/height <= 0 表示"沿用设备当前几何"（HDMI RX 在无信号时是 640x480）
    if (width_ <= 0 || height_ <= 0) {
        v4l2_format cur{};
        cur.type = BufType();
        if (XioCtl(fd_, VIDIOC_G_FMT, &cur) == 0) {
            width_ = (int)(mplane_ ? cur.fmt.pix_mp.width : cur.fmt.pix.width);
            height_ = (int)(mplane_ ? cur.fmt.pix_mp.height : cur.fmt.pix.height);
        }
        if (width_ <= 0) width_ = 640;
        if (height_ <= 0) height_ = 480;
    }

    if (!SetupFormat()) return -4;
    if (!QueueAll()) return -5;
    if (fps_ > 0) SetFps();

    enum v4l2_buf_type type = (enum v4l2_buf_type)BufType();
    if (XioCtl(fd_, VIDIOC_STREAMON, &type) < 0) {
        fprintf(stderr, "v4l2: STREAMON failed errno=%d\n", errno);
        return -6;
    }
    streaming_ = true;
    started_at_ = time(nullptr);

    char cc[5];
    fprintf(stderr, "v4l2: %s %dx%d %s (%s, %d plane%s)\n", dev_.c_str(), width_, height_,
            FourccStr(pixfmt_, cc), mplane_ ? "mplane" : "single", num_planes_,
            num_planes_ == 1 ? "" : "s");
    return 0;
}

bool V4l2Source::SetupFormat() {
    std::vector<uint32_t> cands;
    if (forced_format_) {
        cands.push_back(forced_pixfmt_);
    } else if (prefer_mjpeg_) {
        const uint32_t list[] = {V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_JPEG, V4L2_PIX_FMT_YUYV,
                                 V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_NV16,
                                 V4L2_PIX_FMT_BGR24, V4L2_PIX_FMT_RGB24};
        cands.assign(list, list + sizeof(list) / sizeof(list[0]));
    } else {
        // 板端优先零转换的 BGR3；USB 摄像头通常只给 YUYV/MJPG
        const uint32_t list[] = {V4L2_PIX_FMT_BGR24, V4L2_PIX_FMT_RGB24, V4L2_PIX_FMT_NV12,
                                 V4L2_PIX_FMT_NV16, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_UYVY,
                                 V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_JPEG};
        cands.assign(list, list + sizeof(list) / sizeof(list[0]));
    }

    // 第一轮只接受精确尺寸，避免 1080p 设备悄悄退到 640x480；失败再要"设备能给的最好"
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < cands.size(); ++i) {
            if (TryFormat(cands[i], width_, height_, pass == 0)) return true;
        }
    }
    fprintf(stderr, "v4l2: no supported pixel format on %s (tried BGR3/RGB3/NV12/NV16/YUYV/"
                    "UYVY/MJPG)\n", dev_.c_str());
    return false;
}

bool V4l2Source::TryFormat(uint32_t pf, int w, int h, bool exact) {
    v4l2_format f{};
    f.type = BufType();
    if (mplane_) {
        f.fmt.pix_mp.width = (uint32_t)w;
        f.fmt.pix_mp.height = (uint32_t)h;
        f.fmt.pix_mp.pixelformat = pf;
        f.fmt.pix_mp.field = V4L2_FIELD_NONE;
        f.fmt.pix_mp.colorspace = V4L2_COLORSPACE_SRGB;
        f.fmt.pix_mp.num_planes = 0;   // 让驱动决定
    } else {
        f.fmt.pix.width = (uint32_t)w;
        f.fmt.pix.height = (uint32_t)h;
        f.fmt.pix.pixelformat = pf;
        f.fmt.pix.field = V4L2_FIELD_NONE;
        f.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;
    }
    if (XioCtl(fd_, VIDIOC_S_FMT, &f) < 0) return false;

    const uint32_t got = mplane_ ? f.fmt.pix_mp.pixelformat : f.fmt.pix.pixelformat;
    if (got != pf) return false;
    const int got_w = (int)(mplane_ ? f.fmt.pix_mp.width : f.fmt.pix.width);
    const int got_h = (int)(mplane_ ? f.fmt.pix_mp.height : f.fmt.pix.height);
    if (exact && (got_w != w || got_h != h)) return false;

    fmt_ = f;
    width_ = got_w;
    height_ = got_h;
    pixfmt_ = got;
    num_planes_ = mplane_ ? (int)f.fmt.pix_mp.num_planes : 1;
    if (num_planes_ < 1) num_planes_ = 1;
    if (num_planes_ > kMaxPlanes) num_planes_ = kMaxPlanes;
    for (int p = 0; p < kMaxPlanes; ++p) {
        stride_[p] = mplane_ ? (int)f.fmt.pix_mp.plane_fmt[p].bytesperline
                             : (int)f.fmt.pix.bytesperline;
    }
    return true;
}

void V4l2Source::SetFps() {
    v4l2_streamparm parm{};
    parm.type = BufType();
    if (XioCtl(fd_, VIDIOC_G_PARM, &parm) == 0 &&
        (parm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME)) {
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = (uint32_t)fps_;
        if (XioCtl(fd_, VIDIOC_S_PARM, &parm) < 0)
            fprintf(stderr, "v4l2: S_PARM fps=%d failed errno=%d (ignored)\n", fps_, errno);
    }
}

bool V4l2Source::QueueAll() {
    v4l2_requestbuffers req{};
    req.count = kBufferCount;
    req.type = BufType();
    req.memory = V4L2_MEMORY_MMAP;
    if (XioCtl(fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
        fprintf(stderr, "v4l2: REQBUFS failed errno=%d (count=%u)\n", errno, req.count);
        return false;
    }
    buffers_.resize(req.count);
    for (uint32_t i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        v4l2_plane planes[kMaxPlanes]{};
        buf.type = BufType();
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (mplane_) {
            buf.m.planes = planes;
            buf.length = (uint32_t)num_planes_;
        }
        if (XioCtl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            fprintf(stderr, "v4l2: QUERYBUF[%u] failed errno=%d\n", i, errno);
            return false;
        }
        Buffer& b = buffers_[i];
        b.planes = num_planes_;
        for (int p = 0; p < num_planes_; ++p) {
            const size_t len = mplane_ ? planes[p].length : buf.length;
            const off_t off = mplane_ ? (off_t)planes[p].m.mem_offset : (off_t)buf.m.offset;
            void* start = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, off);
            if (start == MAP_FAILED) {
                fprintf(stderr, "v4l2: mmap[%u/%d] failed errno=%d\n", i, p, errno);
                return false;
            }
            b.start[p] = start;
            b.len[p] = len;
        }
    }
    for (uint32_t i = 0; i < req.count; ++i) {
        if (!QueueBuffer((int)i)) return false;
    }
    return true;
}

bool V4l2Source::QueueBuffer(int index) {
    if (index < 0 || index >= (int)buffers_.size()) return false;
    v4l2_buffer buf{};
    v4l2_plane planes[kMaxPlanes]{};
    buf.type = BufType();
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = (uint32_t)index;
    if (mplane_) {
        buf.m.planes = planes;
        buf.length = (uint32_t)num_planes_;
        for (int p = 0; p < num_planes_; ++p)
            planes[p].length = (uint32_t)buffers_[index].len[p];
    }
    if (XioCtl(fd_, VIDIOC_QBUF, &buf) < 0) {
        fprintf(stderr, "v4l2: QBUF[%d] failed errno=%d\n", index, errno);
        return false;
    }
    return true;
}

bool V4l2Source::Dequeue(int* index, size_t* used) {
    v4l2_buffer buf{};
    v4l2_plane planes[kMaxPlanes]{};
    buf.type = BufType();
    buf.memory = V4L2_MEMORY_MMAP;
    if (mplane_) {
        buf.m.planes = planes;
        buf.length = (uint32_t)num_planes_;
    }
    if (XioCtl(fd_, VIDIOC_DQBUF, &buf) < 0) return false;
    if (buf.index >= buffers_.size()) {
        fprintf(stderr, "v4l2: DQBUF returned out-of-range index %u\n", buf.index);
        return false;
    }
    *index = (int)buf.index;
    *used = mplane_ ? (size_t)planes[0].bytesused : (size_t)buf.bytesused;
    return true;
}

bool V4l2Source::Process(Frame& frame) {
    // 采集卡偶发 1~2 秒不出帧（切分辨率 / HDMI 重同步）不应终止整轮；
    // 连续 max_timeouts_ 次超时才认输。
    int timeouts = 0;
    for (;;) {
        pollfd pfd{fd_, POLLIN, 0};
        const int pr = ::poll(&pfd, 1, timeout_ms_);
        if (pr < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "v4l2: poll failed errno=%d\n", errno);
            return false;
        }
        if (pr == 0) {
            if (++timeouts >= max_timeouts_) {
                fprintf(stderr, "v4l2: no frame after %d x %dms (device=%s)\n", max_timeouts_,
                        timeout_ms_, dev_.c_str());
                return false;
            }
            continue;
        }
        if (pfd.revents & (POLLERR | POLLNVAL)) {
            fprintf(stderr, "v4l2: device error revents=0x%x\n", pfd.revents);
            return false;
        }
        if (pfd.revents & POLLIN) break;
    }

    int keep = -1;
    size_t used = 0;
    if (!Dequeue(&keep, &used)) {
        fprintf(stderr, "v4l2: DQBUF failed errno=%d\n", errno);
        return false;
    }

    // 最新帧优先：把已经排队的旧帧丢掉，避免慢消费者看到延迟累积
    for (;;) {
        pollfd more{fd_, POLLIN, 0};
        if (::poll(&more, 1, 0) <= 0) break;
        int next = -1;
        size_t next_used = 0;
        if (!Dequeue(&next, &next_used)) break;
        QueueBuffer(keep);
        keep = next;
        used = next_used;
    }

    cv::Mat img;
    const bool ok = Convert(buffers_[keep], used, &img);
    QueueBuffer(keep);
    if (!ok) {
        char cc[5];
        fprintf(stderr, "v4l2: decode failed (%zu bytes, fmt=%s)\n", used,
                FourccStr(pixfmt_, cc));
        return true;   // 单帧坏数据不断流
    }

    frame.image = img;
    frame.orig_w = img.cols;
    frame.orig_h = img.rows;
    char name[64];
    snprintf(name, sizeof(name), "v4l2_%ld_%06ld", (long)started_at_, ++frame_no_);
    frame.image_name = name;
    return true;
}

bool V4l2Source::Convert(const Buffer& b, size_t used, cv::Mat* out) {
    if (!b.start[0] || width_ <= 0 || height_ <= 0) return false;
    const int s0 = stride_[0] > 0 ? stride_[0] : DefaultStride(pixfmt_, width_);
    // 单平面 NV12 的 UV 行跨距与 Y 相同；多平面才用 plane_fmt[1]
    const int s1 = stride_[1] > 0 ? stride_[1] : (b.planes > 1 ? width_ : s0);

    switch (pixfmt_) {
    case V4L2_PIX_FMT_BGR24: {
        cv::Mat m(height_, width_, CV_8UC3, b.start[0], (size_t)s0);
        *out = m.clone();
        return true;
    }
    case V4L2_PIX_FMT_RGB24: {
        cv::Mat m(height_, width_, CV_8UC3, b.start[0], (size_t)s0);
        cv::cvtColor(m, *out, cv::COLOR_RGB2BGR);
        return true;
    }
    case V4L2_PIX_FMT_NV12:
    case V4L2_PIX_FMT_NV16: {
        const bool nv12 = pixfmt_ == V4L2_PIX_FMT_NV12;
        const uint8_t* chroma = b.planes > 1 && b.start[1]
                                    ? (const uint8_t*)b.start[1]
                                    : (const uint8_t*)b.start[0] + (size_t)s0 * height_;
        const cv::Mat y(height_, width_, CV_8UC1, b.start[0], (size_t)s0);
        if (nv12) {
            const cv::Mat uv(height_ / 2, width_ / 2, CV_8UC2, (void*)chroma, (size_t)s1);
            cv::cvtColorTwoPlane(y, uv, *out, cv::COLOR_YUV2BGR_NV12);
            return true;
        }
        // NV16 是 4:2:2：色度平面整高、每行 w/2 个 UV 对；按字节交织后等同 YUYV
        cv::Mat yuyv(height_, width_, CV_8UC2);
        for (int row = 0; row < height_; ++row) {
            const uint8_t* yr = y.ptr<uint8_t>(row);
            const uint8_t* uv = chroma + (size_t)row * (size_t)s1;
            uint8_t* dst = yuyv.ptr<uint8_t>(row);
            for (int x = 0; x < width_; ++x) {
                dst[2 * x] = yr[x];
                dst[2 * x + 1] = uv[x];
            }
        }
        cv::cvtColor(yuyv, *out, cv::COLOR_YUV2BGR_YUYV);
        return true;
    }
    case V4L2_PIX_FMT_YUYV: {
        cv::Mat m(height_, width_, CV_8UC2, b.start[0], (size_t)s0);
        cv::cvtColor(m, *out, cv::COLOR_YUV2BGR_YUYV);
        return true;
    }
    case V4L2_PIX_FMT_UYVY: {
        cv::Mat m(height_, width_, CV_8UC2, b.start[0], (size_t)s0);
        cv::cvtColor(m, *out, cv::COLOR_YUV2BGR_UYVY);
        return true;
    }
    case V4L2_PIX_FMT_MJPEG:
    case V4L2_PIX_FMT_JPEG: {
        const size_t n = used > 0 ? used : (size_t)s0 * height_;
        if (n < 4) return false;
        const cv::Mat raw(1, (int)n, CV_8UC1, b.start[0]);
        const cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
        if (img.empty()) return false;
        *out = img;
        return true;
    }
    default:
        return false;
    }
}
