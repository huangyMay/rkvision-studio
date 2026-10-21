// archive sink：NG 证据存档（快照 jpg + judge 明细 json），keep_days 清理过期文件。
// 职责边界：NG 判定来自上游 judge 的 Frame.verdict，本插件只负责落盘与保留策略。
#pragma once
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
    std::string ng_dir_;
    int keep_days_ = 7;     // 0 = 永不清理
};
