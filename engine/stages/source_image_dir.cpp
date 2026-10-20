#include "stages/source_image_dir.hpp"

#include <dirent.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <opencv2/imgcodecs.hpp>

#include "core/registry.hpp"

REGISTER_STAGE("image_dir", ImageDirSource)

namespace {

bool HasImageExt(const std::string& name) {
    size_t dot = name.rfind('.');
    if (dot == std::string::npos) return false;
    std::string ext = name.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "ppm" || ext == "bmp";
}

std::string Stem(const std::string& path) {
    size_t slash = path.find_last_of('/');
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return path;
    return path.substr((slash == std::string::npos ? 0 : slash + 1), dot - (slash == std::string::npos ? 0 : slash + 1));
}

} // namespace

void ImageDirSource::ScanDir(const std::string& path) {
    DIR* d = opendir(path.c_str());
    if (!d) return;
    while (struct dirent* e = readdir(d)) {
        std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        if (!HasImageExt(name)) continue;
        files_.push_back(path + "/" + name);
    }
    closedir(d);
    std::sort(files_.begin(), files_.end());
}

int ImageDirSource::Init(const nlohmann::json& params, const StageContext& ctx) {
    if (params.is_null()) {
        fprintf(stderr, "image_dir: params.path is required\n");
        return -1;
    }
    path_ = params.value("path", "");
    if (path_.empty()) {
        fprintf(stderr, "image_dir: params.path is required\n");
        return -1;
    }
    // 相对路径按工程包根解析
    if (path_[0] != '/' && !ctx.project_root.empty()) {
        path_ = ctx.project_root + "/" + path_;
    }
    loop_ = params.value("loop", false);
    max_frames_ = params.value("max_frames", 0);

    ScanDir(path_);
    if (files_.empty()) {
        fprintf(stderr, "image_dir: no image files in '%s'\n", path_.c_str());
        return -2;
    }
    return 0;
}

bool ImageDirSource::Process(Frame& frame) {
    if (max_frames_ > 0 && emitted_ >= max_frames_) return false;
    if (next_ >= files_.size()) {
        if (!loop_) return false;
        next_ = 0;
    }
    const std::string& file = files_[next_++];
    frame.image = cv::imread(file, cv::IMREAD_COLOR);
    if (frame.image.empty()) {
        fprintf(stderr, "image_dir: failed to decode '%s'\n", file.c_str());
        return false;
    }
    frame.image_name = Stem(file);
    frame.orig_w = frame.image.cols;
    frame.orig_h = frame.image.rows;
    ++emitted_;
    return true;
}
