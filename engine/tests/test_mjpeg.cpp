// MJPEG sink 行为规格：HTTP 头 + multipart 边界 + JPEG 载荷；GET / 出引导页。
#include "core/frame.hpp"
#include "core/registry.hpp"
#include "stages/sink_mjpeg.hpp"

#include <arpa/inet.h>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

static std::string HttpGet(const char* path, int port, size_t want) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { fprintf(stderr, "socket failed\n"); exit(1); }
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(0x7F000001);   // 127.0.0.1
    if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "connect failed\n"); exit(1);
    }
    char req[128];
    snprintf(req, sizeof(req), "GET %s HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n", path);
    if (::send(fd, req, strlen(req), 0) <= 0) { fprintf(stderr, "send failed\n"); exit(1); }
    std::string out;
    char buf[2048];
    while (out.size() < want) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        out.append(buf, (size_t)n);
    }
    ::close(fd);
    return out;
}

int main() {
    StageContext ctx;
    MjpegSink sink;
    nlohmann::json p;
    p["port"] = 18123;
    p["path"] = "/stream";
    p["fps"] = 30;
    CHECK(sink.Init(p, ctx) == 0);

    // 喂一帧（含检出 → 画框分支）
    Frame f;
    f.image.create(60, 80, CV_8UC3);
    Detection d;
    d.label = "helmet"; d.score = 0.9f;
    d.x1 = 5; d.y1 = 5; d.x2 = 50; d.y2 = 40;
    f.detections = {d};
    CHECK(sink.Process(f) == true);

    // ① /stream：multipart 头 + 边界 + JPEG 载荷（FFD8 起）
    std::string body = HttpGet("/stream", 18123, 2048);
    CHECK(body.find("200 OK") != std::string::npos);
    CHECK(body.find("multipart/x-mixed-replace") != std::string::npos);
    CHECK(body.find("--frame") != std::string::npos);
    CHECK(body.find("Content-Type: image/jpeg") != std::string::npos);
    size_t soi = body.find("\xFF\xD8");
    CHECK(soi != std::string::npos);

    // ② / 引导页 200
    std::string index = HttpGet("/", 18123, 256);
    CHECK(index.find("200 OK") != std::string::npos);

    // ③ 其他路径 404
    std::string nf = HttpGet("/nope", 18123, 128);
    CHECK(nf.find("404") != std::string::npos);

    printf("test_mjpeg: all checks passed\n");
    return 0;
}
