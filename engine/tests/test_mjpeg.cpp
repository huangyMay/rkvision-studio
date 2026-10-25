// MJPEG sink 行为规格：HTTP 头 + multipart 边界 + JPEG 载荷；GET / 出引导页。
// 所有 socket 都带超时：服务端不产帧时必须失败而不是永久挂起。
#include "core/frame.hpp"
#include "core/registry.hpp"
#include "stages/sink_mjpeg.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <thread>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

namespace {

int Connect(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { fprintf(stderr, "socket failed\n"); return -1; }
    struct timeval tv { 2, 0 };
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(0x7F000001);   // 127.0.0.1
    if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "connect :%d failed: %s\n", port, strerror(errno));
        ::close(fd);
        return -1;
    }
    return fd;
}

bool SendRequest(int fd, const char* path) {
    char req[160];
    snprintf(req, sizeof(req), "GET %s HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n", path);
    return ::send(fd, req, strlen(req), 0) > 0;
}

// 读满 want 字节或对端关闭；SO_RCVTIMEO 保证不会无限等
std::string HttpGet(const char* path, int port, size_t want) {
    int fd = Connect(port);
    if (fd < 0) return {};
    if (!SendRequest(fd, path)) { ::close(fd); return {}; }
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

} // namespace

int main() {
    // ① /stream：multipart 头 + 边界 + JPEG 载荷（FFD8 起）
    // ② / 引导页；③ 未知路径 404；④ /health；⑤ /snapshot.jpg
    {
        StageContext ctx;
        MjpegSink sink;
        nlohmann::json p;
        p["port"] = 18123;
        p["path"] = "/stream";
        p["fps"] = 30;
        CHECK(sink.Init(p, ctx) == 0);

        Frame f;
        f.image.create(60, 80, CV_8UC3);
        Detection d;
        d.label = "helmet"; d.score = 0.9f;
        d.x1 = 5; d.y1 = 5; d.x2 = 50; d.y2 = 40;
        f.detections = {d};
        CHECK(sink.Process(f) == true);

        std::string body = HttpGet("/stream", 18123, 2048);
        CHECK(body.find("200 OK") != std::string::npos);
        CHECK(body.find("multipart/x-mixed-replace") != std::string::npos);
        CHECK(body.find("--frame") != std::string::npos);
        CHECK(body.find("Content-Type: image/jpeg") != std::string::npos);
        CHECK(body.find("\xFF\xD8") != std::string::npos);

        std::string index = HttpGet("/", 18123, 256);
        CHECK(index.find("200 OK") != std::string::npos);

        std::string nf = HttpGet("/nope", 18123, 128);
        CHECK(nf.find("404") != std::string::npos);

        std::string health = HttpGet("/health", 18123, 256);
        CHECK(health.find("ok") != std::string::npos);

        std::string snap = HttpGet("/snapshot.jpg", 18123, 2048);
        CHECK(snap.find("200 OK") != std::string::npos);
        CHECK(snap.find("\xFF\xD8") != std::string::npos);
    }

    // ⑥ 析构时仍有活跃客户端：连接线程只持有 shared state，不能访问已销毁的 sink。
    //    旧实现 detach 线程捕获 this，在 ASAN 下这里会报 use-after-free。
    {
        int fd = -1;
        {
            StageContext ctx;
            MjpegSink sink;
            nlohmann::json p;
            p["port"] = 18124;
            p["fps"] = 30;
            CHECK(sink.Init(p, ctx) == 0);
            Frame f;
            f.image.create(32, 32, CV_8UC3);
            CHECK(sink.Process(f));

            fd = Connect(18124);
            CHECK(fd >= 0);
            CHECK(SendRequest(fd, "/stream"));
            char buf[64];
            CHECK(::recv(fd, buf, sizeof(buf), 0) > 0);
            // sink 在这里析构，客户端连接仍然开着
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        ::close(fd);
    }

    printf("test_mjpeg: all checks passed\n");
    return 0;
}
