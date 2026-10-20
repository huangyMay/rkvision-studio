#include "stages/sink_mjpeg.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/registry.hpp"

REGISTER_STAGE("mjpeg", MjpegSink)

namespace {

const char* kIndexHtml =
    "<html><head><title>RVS preview</title></head>"
    "<body style='margin:0;background:#111;text-align:center'>"
    "<img src='/stream' style='max-width:100vw;max-height:100vh'>"
    "</body></html>";

bool SendAll(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}

} // namespace

MjpegSink::~MjpegSink() {
    running_ = false;
    if (listen_fd_ >= 0) ::close(listen_fd_);
    if (server_.joinable()) server_.join();
}

int MjpegSink::Init(const nlohmann::json& params, const StageContext&) {
    if (!params.is_null()) {
        port_ = params.value("port", 8000);
        path_ = params.value("path", "/stream");
        draw_ = params.value("draw", true);
        fps_ = params.value("fps", 10);
    }
    if (fps_ <= 0 || fps_ > 60) fps_ = 10;

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        fprintf(stderr, "mjpeg: socket failed errno=%d\n", errno);
        return -1;
    }
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port_);
    if (::bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0 || ::listen(listen_fd_, 4) < 0) {
        fprintf(stderr, "mjpeg: bind/listen :%d failed errno=%d\n", port_, errno);
        return -2;
    }
    running_ = true;
    server_ = std::thread(&MjpegSink::ServerLoop, this);
    return 0;
}

void MjpegSink::ServerLoop() {
    while (running_) {
        // accept 阻塞时 close() 在 Linux 上不可靠，改 poll 超时轮询保证可退出
        pollfd pfd { listen_fd_, POLLIN, 0 };
        int r = ::poll(&pfd, 1, 200);
        if (!running_) break;
        if (r <= 0 || !(pfd.revents & POLLIN)) continue;
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        // 预览场景客户端极少（1~2 个），每连接一线程即可
        std::thread([this, fd]() {
            struct timeval tv { 5, 0 };
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            char req[1024] = {0};
            ssize_t n = ::recv(fd, req, sizeof(req) - 1, 0);
            if (n <= 0) { ::close(fd); return; }
            std::string request(req);
            std::string line = request.substr(0, request.find("\r\n"));
            // "GET /stream HTTP/1.1"
            std::string url = "/";
            {
                size_t sp1 = line.find(' ');
                size_t sp2 = line.rfind(' ');
                if (sp1 != std::string::npos && sp2 != sp1) url = line.substr(sp1 + 1, sp2 - sp1 - 1);
            }
            if (url == path_) {
                std::string head =
                    "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                    "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
                if (!SendAll(fd, head.data(), head.size())) { ::close(fd); return; }
                auto interval = std::chrono::milliseconds(1000 / fps_);
                while (running_) {
                    std::vector<uint8_t> frame;
                    {
                        std::lock_guard<std::mutex> lock(mtx_);
                        frame = jpeg_;   // 有帧才发；首帧前按 fps 节奏等待
                    }
                    if (!frame.empty()) {
                        char part_head[128];
                        snprintf(part_head, sizeof(part_head),
                                 "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
                                 frame.size());
                        if (!SendAll(fd, part_head, strlen(part_head))) break;
                        if (!SendAll(fd, (const char*)frame.data(), frame.size())) break;
                        if (!SendAll(fd, "\r\n", 2)) break;
                    }
                    std::this_thread::sleep_for(interval);
                }
            } else if (url == "/") {
                std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n";
                resp += kIndexHtml;
                SendAll(fd, resp.data(), resp.size());
            } else {
                const char* nf = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                SendAll(fd, nf, strlen(nf));
            }
            ::close(fd);
        }).detach();
    }
}

bool MjpegSink::Process(Frame& frame) {
    cv::Mat canvas = frame.orig_image.empty() ? frame.image : frame.orig_image;
    if (draw_ && !frame.detections.empty()) {
        canvas = canvas.clone();
        for (const Detection& d : frame.detections) {
            cv::rectangle(canvas, cv::Point((int)d.x1, (int)d.y1), cv::Point((int)d.x2, (int)d.y2),
                          cv::Scalar(0, 255, 0), 2);
            char label[64];
            snprintf(label, sizeof(label), "%s %.2f", d.label.c_str(), d.score);
            cv::putText(canvas, label, cv::Point((int)d.x1, std::max(0, (int)d.y1 - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
    std::vector<uchar> buf;
    if (!cv::imencode(".jpg", canvas, buf)) {
        fprintf(stderr, "mjpeg: imencode failed\n");
        return true;   // 单帧编码失败不断流
    }
    std::lock_guard<std::mutex> lock(mtx_);
    jpeg_.assign(buf.begin(), buf.end());
    has_frame_ = true;
    return true;
}
