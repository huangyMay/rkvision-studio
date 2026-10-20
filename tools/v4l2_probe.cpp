// v4l2_probe：枚举 /dev/videoN 支持的像素格式 + 实测 S_FMT 协商。
// 用法: v4l2_probe </dev/videoN> [width] [height]
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

static int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r < 0 && errno == EINTR);
    return r;
}

static void fourcc(uint32_t f, char* out) {
    out[0] = f & 0xff; out[1] = (f >> 8) & 0xff; out[2] = (f >> 16) & 0xff; out[3] = (f >> 24) & 0xff;
    out[4] = 0;
}

int main(int argc, char** argv) {
    const char* dev = argc > 1 ? argv[1] : "/dev/video0";
    int w = argc > 2 ? atoi(argv[2]) : 1920;
    int h = argc > 3 ? atoi(argv[3]) : 1080;

    int fd = open(dev, O_RDWR);
    if (fd < 0) { printf("open %s failed errno=%d\n", dev, errno); return 1; }

    printf("== supported formats on %s ==\n", dev);
    v4l2_fmtdesc desc {};
    desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    for (desc.index = 0; xioctl(fd, VIDIOC_ENUM_FMT, &desc) == 0; desc.index++) {
        char f[5]; fourcc(desc.pixelformat, f);
        printf("  [%u] %s (%s)\n", desc.index, f, desc.description);
    }

    printf("== G_FMT (current) ==\n");
    {
        v4l2_format fmt {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd, VIDIOC_G_FMT, &fmt) == 0) {
            char f[5]; fourcc(fmt.fmt.pix.pixelformat, f);
            printf("  current: %s %ux%u bytesperline=%u sizeimage=%u\n", f,
                   fmt.fmt.pix.width, fmt.fmt.pix.height,
                   fmt.fmt.pix.bytesperline, fmt.fmt.pix.sizeimage);
        } else {
            printf("  G_FMT failed errno=%d\n", errno);
        }
    }

    printf("== S_FMT try (requested %dx%d) ==\n", w, h);
    const char* cands[] = {"YUYV", "NV12", "RGB3", "BGR3", "RG12"};
    for (const char* c : cands) {
        v4l2_format fmt {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = w;
        fmt.fmt.pix.height = h;
        fmt.fmt.pix.pixelformat = c[0] | (c[1] << 8) | (c[2] << 16) | (c[3] << 24);
        if (xioctl(fd, VIDIOC_S_FMT, &fmt) == 0) {
            char f[5]; fourcc(fmt.fmt.pix.pixelformat, f);
            printf("  %s -> OK, got %s %ux%u\n", c, f, fmt.fmt.pix.width, fmt.fmt.pix.height);
        } else {
            printf("  %s -> EINVAL\n", c);
        }
    }
    close(fd);
    return 0;
}
