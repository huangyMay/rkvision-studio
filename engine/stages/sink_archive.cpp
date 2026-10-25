#include "stages/sink_archive.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <time.h>

#include <opencv2/imgcodecs.hpp>

#include "core/registry.hpp"

REGISTER_STAGE("archive", ArchiveSink)

// 设计稿 §3.1/§6 使用的别名（宏以类名生成符号，二次注册须手写）
namespace {
const bool s_reg_sink_archive_alias = [] {
    StageRegistry::Instance().Register("sink_archive", [] {
        return std::unique_ptr<IStage>(new ArchiveSink());
    });
    return true;
}();
} // namespace

static void MkdirP(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' && i > 0) mkdir(cur.c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

int ArchiveSink::Init(const nlohmann::json& params, const StageContext& ctx) {
    if (!params.is_null()) {
        ng_dir_ = params.value("ng_dir", "ng_images");
        keep_days_ = params.value("keep_days", 7);
    }
    if (!ng_dir_.empty() && ng_dir_[0] != '/' && !ctx.project_root.empty())
        ng_dir_ = ctx.project_root + "/" + ng_dir_;
    if (ng_dir_.empty()) {
        fprintf(stderr, "archive: params.ng_dir is required\n");
        return -1;
    }
    MkdirP(ng_dir_);

    last_cleanup_ = time(nullptr);
    CleanupExpired(last_cleanup_);
    return 0;
}

void ArchiveSink::CleanupExpired(time_t now) {
    if (keep_days_ <= 0) return;
    const time_t cutoff = now - (time_t)keep_days_ * 24 * 3600;
    DIR* d = opendir(ng_dir_.c_str());
    if (!d) return;   // 目录不可达静默跳过，不阻断装配
    while (dirent* e = readdir(d)) {
        std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        const std::string full = ng_dir_ + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
        if (st.st_mtime < cutoff && remove(full.c_str()) != 0)
            fprintf(stderr, "archive: remove failed: %s\n", full.c_str());
    }
    closedir(d);
}

void ArchiveSink::MaybeCleanup() {
    if (keep_days_ <= 0) return;
    const time_t now = time(nullptr);
    // 长跑进程：每小时清一次，而不是只在 Init 清一次
    if (last_cleanup_ != 0 && now - last_cleanup_ < 3600) return;
    last_cleanup_ = now;
    CleanupExpired(now);
}

std::string ArchiveSink::UniqueBase(const std::string& stem) {
    const std::string base = ng_dir_ + "/" + stem;
    if (access((base + ".jpg").c_str(), F_OK) != 0 &&
        access((base + ".json").c_str(), F_OK) != 0)
        return base;
    // 同名证据（image_dir loop / 重复文件名）不覆盖，追加序号
    for (;;) {
        const std::string cand = base + "_" + std::to_string(++seq_);
        if (access((cand + ".jpg").c_str(), F_OK) != 0 &&
            access((cand + ".json").c_str(), F_OK) != 0)
            return cand;
    }
}

bool ArchiveSink::Process(Frame& frame) {
    if (frame.verdict != "NG") return true;   // 只存 NG 证据
    MaybeCleanup();

    const std::string stem =
        (frame.image_name.empty() ? std::string("frame") : frame.image_name) + "_ng";
    const std::string base = UniqueBase(stem);
    const cv::Mat img = frame.orig_image.empty() ? frame.image : frame.orig_image;
    if (!img.empty() && !cv::imwrite(base + ".jpg", img))
        fprintf(stderr, "archive: imwrite failed: %s.jpg\n", base.c_str());

    std::ofstream jf(base + ".json");
    if (!jf.good()) {
        fprintf(stderr, "archive: cannot write %s.json\n", base.c_str());
        return true;                          // 存档失败不阻断流水线（Review Focus 4）
    }
    jf << frame.judge.dump(2);
    return true;
}
