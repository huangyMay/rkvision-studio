#include "stages/sink_mjpeg.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
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

void SendSimple(int fd, const char* status, const char* type, const std::string& body) {
    char head[256];
    snprintf(head, sizeof(head),
             "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
             "Cache-Control: no-store\r\nConnection: close\r\n\r\n",
             status, type, body.size());
    SendAll(fd, head, strlen(head));
    if (!body.empty()) SendAll(fd, body.data(), body.size());
}

// 每个连接一条 detached 线程；只持有 shared state，不持有 MjpegSink*
void ServeClient(const std::shared_ptr<MjpegSink::Shared>& st, int fd, std::string path,
                 int fps) {
    struct timeval tv { 2, 0 };
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    char req[1024] = {0};
    ssize_t n = ::recv(fd, req, sizeof(req) - 1, 0);
    if (n <= 0) {
        ::close(fd);
        st->clients.fetch_sub(1);
        return;
    }
    std::string request(req);
    std::string line = request.substr(0, request.find("\r\n"));
    std::string url = "/";
    {
        size_t sp1 = line.find(' ');
        size_t sp2 = line.rfind(' ');
        if (sp1 != std::string::npos && sp2 != sp1) url = line.substr(sp1 + 1, sp2 - sp1 - 1);
        const size_t q = url.find('?');
        if (q != std::string::npos) url = url.substr(0, q);
    }

    if (url == path) {
        const char* head =
            "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Cache-Control: no-cache, no-store\r\nConnection: close\r\n\r\n";
        if (!SendAll(fd, head, strlen(head))) {
            ::close(fd);
            st->clients.fetch_sub(1);
            return;
        }
        const auto interval = std::chrono::milliseconds(1000 / (fps > 0 ? fps : 10));
        while (st->running.load()) {
            std::vector<uint8_t> frame;
            {
                std::lock_guard<std::mutex> lock(st->mtx);
                frame = st->jpeg;   // 最新一帧；首帧前按 fps 节奏等待
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
        SendSimple(fd, "200 OK", "text/html", kIndexHtml);
    } else if (url == "/snapshot.jpg" || url == "/snapshot") {
        std::vector<uint8_t> frame;
        {
            std::lock_guard<std::mutex> lock(st->mtx);
            frame = st->jpeg;
        }
        if (frame.empty()) {
            SendSimple(fd, "503 Service Unavailable", "text/plain", "no frame yet");
        } else {
            SendSimple(fd, "200 OK", "image/jpeg",
                       std::string((const char*)frame.data(), frame.size()));
        }
    } else if (url == "/health") {
        std::vector<uint8_t> frame;
        {
            std::lock_guard<std::mutex> lock(st->mtx);
            frame = st->jpeg;
        }
        SendSimple(fd, "200 OK", "text/plain", frame.empty() ? "waiting" : "ok");
    } else {
        SendSimple(fd, "404 Not Found", "text/plain", "not found");
    }

    ::close(fd);
    st->clients.fetch_sub(1);
}

} // namespace

MjpegSink::~MjpegSink() {
    if (state_) state_->running.store(false);
    if (server_.joinable()) server_.join();   // poll 200ms 内退出
    if (listen_fd_ >= 0) ::close(listen_fd_);
    // 客户端线程可能还在 send/sleep；它们持有 state_ 的 shared_ptr，随进程退出自行结束
}

int MjpegSink::Init(const nlohmann::json& params, const StageContext&) {
    if (!params.is_null()) {
        port_ = params.value("port", 8000);
        path_ = params.value("path", "/stream");
        draw_ = params.value("draw", true);
        fps_ = params.value("fps", 10);
        max_clients_ = params.value("max_clients", 4);
    }
    if (fps_ <= 0 || fps_ > 60) fps_ = 10;
    if (max_clients_ <= 0) max_clients_ = 1;

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
    if (::bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0 || ::listen(listen_fd_, 8) < 0) {
        fprintf(stderr, "mjpeg: bind/listen :%d failed errno=%d\n", port_, errno);
        return -2;
    }

    state_ = std::make_shared<Shared>();
    state_->running.store(true);
    server_ = std::thread(&MjpegSink::ServerLoop, this, state_);
    return 0;
}

void MjpegSink::ServerLoop(const std::shared_ptr<Shared>& st) {
    while (st->running.load()) {
        // accept 阻塞时 close() 在 Linux 上不可靠，改 poll 超时轮询保证可退出
        pollfd pfd{listen_fd_, POLLIN, 0};
        int r = ::poll(&pfd, 1, 200);
        if (!st->running.load()) break;
        if (r <= 0 || !(pfd.revents & POLLIN)) continue;
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;

        if (st->clients.load() >= max_clients_) {
            SendSimple(fd, "503 Service Unavailable", "text/plain", "too many clients");
            ::close(fd);
            continue;
        }
        st->clients.fetch_add(1);
        const std::string path = path_;
        const int fps = fps_;
        std::thread([st, fd, path, fps]() { ServeClient(st, fd, path, fps); }).detach();
    }
}

bool MjpegSink::Process(Frame& frame) {
    cv::Mat canvas = frame.orig_image.empty() ? frame.image : frame.orig_image;
    if (canvas.empty()) return true;
    if (draw_ && !frame.detections.empty()) {
        canvas = canvas.clone();
        for (size_t i = 0; i < frame.detections.size(); ++i) {
            const Detection& d = frame.detections[i];
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
    std::lock_guard<std::mutex> lock(state_->mtx);
    state_->jpeg.assign(buf.begin(), buf.end());
    return true;
}
