// v4l2_probe：枚举 /dev/videoN 支持的像素格式 + 实测 S_FMT 协商。
// 支持单平面与多平面（Rockchip HDMI-RX 是 mplane）。
// 用法: v4l2_probe </dev/videoN> [width] [height]
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

static int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r < 0 && errno == EINTR);
    return r;
}

static void fourcc(uint32_t f, char* out) {
    out[0] = (char)(f & 0xff);
    out[1] = (char)((f >> 8) & 0xff);
    out[2] = (char)((f >> 16) & 0xff);
    out[3] = (char)((f >> 24) & 0xff);
    out[4] = 0;
}

static uint32_t code(const char* c) {
    return (uint32_t)(c[0] | (c[1] << 8) | (c[2] << 16) | (c[3] << 24));
}

int main(int argc, char** argv) {
    const char* dev = argc > 1 ? argv[1] : "/dev/video0";
    int w = argc > 2 ? atoi(argv[2]) : 1920;
    int h = argc > 3 ? atoi(argv[3]) : 1080;

    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        printf("open %s failed errno=%d\n", dev, errno);
        return 1;
    }

    v4l2_capability cap{};
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
        printf("VIDIOC_QUERYCAP failed errno=%d\n", errno);
        close(fd);
        return 1;
    }
    const uint32_t caps = cap.device_caps ? cap.device_caps : cap.capabilities;
    const bool mplane = (caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) != 0;
    const uint32_t type = mplane ? V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
                                 : V4L2_BUF_TYPE_VIDEO_CAPTURE;
    printf("== %s (%s) : %s ==\n", dev, cap.card, mplane ? "mplane" : "single-planar");

    printf("== supported formats ==\n");
    v4l2_fmtdesc desc{};
    desc.type = type;
    for (desc.index = 0; xioctl(fd, VIDIOC_ENUM_FMT, &desc) == 0; desc.index++) {
        char f[5];
        fourcc(desc.pixelformat, f);
        printf("  [%u] %s (%s)\n", desc.index, f, desc.description);
    }

    printf("== G_FMT (current) ==\n");
    {
        v4l2_format fmt{};
        fmt.type = type;
        if (xioctl(fd, VIDIOC_G_FMT, &fmt) == 0) {
            char f[5];
            if (mplane) {
                fourcc(fmt.fmt.pix_mp.pixelformat, f);
                printf("  current: %s %ux%u planes=%u", f, fmt.fmt.pix_mp.width,
                       fmt.fmt.pix_mp.height, fmt.fmt.pix_mp.num_planes);
                for (unsigned p = 0; p < fmt.fmt.pix_mp.num_planes && p < 3; ++p)
                    printf(" [p%u bpl=%u size=%u]", p,
                           fmt.fmt.pix_mp.plane_fmt[p].bytesperline,
                           fmt.fmt.pix_mp.plane_fmt[p].sizeimage);
                printf("\n");
            } else {
                fourcc(fmt.fmt.pix.pixelformat, f);
                printf("  current: %s %ux%u bytesperline=%u sizeimage=%u\n", f, fmt.fmt.pix.width,
                       fmt.fmt.pix.height, fmt.fmt.pix.bytesperline, fmt.fmt.pix.sizeimage);
            }
        } else {
            printf("  G_FMT failed errno=%d\n", errno);
        }
    }

    printf("== S_FMT try (requested %dx%d) ==\n", w, h);
    const char* cands[] = {"BGR3", "RGB3", "NV12", "NV16", "YUYV", "UYVY", "MJPG", "JPEG"};
    for (const char* c : cands) {
        v4l2_format fmt{};
        fmt.type = type;
        const uint32_t pf = code(c);
        if (mplane) {
            fmt.fmt.pix_mp.width = (uint32_t)w;
            fmt.fmt.pix_mp.height = (uint32_t)h;
            fmt.fmt.pix_mp.pixelformat = pf;
            fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
            fmt.fmt.pix_mp.num_planes = 0;
        } else {
            fmt.fmt.pix.width = (uint32_t)w;
            fmt.fmt.pix.height = (uint32_t)h;
            fmt.fmt.pix.pixelformat = pf;
            fmt.fmt.pix.field = V4L2_FIELD_NONE;
        }
        if (xioctl(fd, VIDIOC_S_FMT, &fmt) == 0) {
            char f[5];
            if (mplane) {
                fourcc(fmt.fmt.pix_mp.pixelformat, f);
                printf("  %s -> OK, got %s %ux%u planes=%u\n", c, f, fmt.fmt.pix_mp.width,
                       fmt.fmt.pix_mp.height, fmt.fmt.pix_mp.num_planes);
            } else {
                fourcc(fmt.fmt.pix.pixelformat, f);
                printf("  %s -> OK, got %s %ux%u\n", c, f, fmt.fmt.pix.width, fmt.fmt.pix.height);
            }
        } else {
            printf("  %s -> EINVAL\n", c);
        }
    }
    close(fd);
    return 0;
}
