// Stage 静态注册表：插件名 → 工厂。禁止按模型家族写分支，一切差异走这里。
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "core/stage.hpp"

class StageRegistry {
public:
    using Factory = std::function<std::unique_ptr<IStage>()>;

    static StageRegistry& Instance();

    void Register(const std::string& name, Factory factory);

    // 未知名返回 nullptr（装配器负责报错）
    std::unique_ptr<IStage> Create(const std::string& name);

private:
    StageRegistry() = default;
    std::map<std::string, Factory> factories_;
};

// 用法：REGISTER_STAGE("image_dir", ImageDirSource)
#define REGISTER_STAGE(NAME, CLS)                                          \
    static bool s_reg_##CLS = ([] {                                        \
        StageRegistry::Instance().Register(NAME, [] {                      \
            return std::unique_ptr<IStage>(new CLS());                     \
        });                                                                \
        return true;                                                       \
    })();
