// archive sink：NG 证据存档（快照 jpg + judge 明细 json），keep_days 清理过期文件。
// 职责边界：NG 判定来自上游 judge 的 Frame.verdict，本插件只负责落盘与保留策略。
#pragma once
#include <ctime>
#include <string>

#include "core/stage.hpp"

class ArchiveSink : public IStage {
public:
    int Init(const nlohmann::json& params, const StageContext& ctx) override;
    bool Process(Frame& frame) override;
    const StageMeta& Meta() const override {
        static StageMeta m{"archive", false, true, false};
        return m;
    }

private:
    void CleanupExpired(time_t now);
    void MaybeCleanup();
    std::string UniqueBase(const std::string& stem);

    std::string ng_dir_;
    int keep_days_ = 7;     // 0 = 永不清理
    time_t last_cleanup_ = 0;
    long seq_ = 0;
};
