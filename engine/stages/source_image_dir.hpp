// image_dir 源：目录内图片按文件名排序逐帧产出；loop 可循环，max_frames 限次。
#pragma once
#include <string>
#include <vector>

#include "core/stage.hpp"

class ImageDirSource : public IStage {
public:
    int Init(const nlohmann::json& params, const StageContext& ctx) override;
    bool Process(Frame& frame) override;
    const StageMeta& Meta() const override {
        static StageMeta m{"image_dir", true, false, false};
        return m;
    }

private:
    void ScanDir(const std::string& path);

    std::vector<std::string> files_;   // 绝对路径，已排序
    std::string path_;
    bool loop_ = false;
    int max_frames_ = 0;               // >0 时硬上限
    size_t next_ = 0;
    int emitted_ = 0;
};
