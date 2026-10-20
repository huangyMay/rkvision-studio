#include "core/registry.hpp"

StageRegistry& StageRegistry::Instance() {
    static StageRegistry inst;
    return inst;
}

void StageRegistry::Register(const std::string& name, Factory factory) {
    factories_[name] = std::move(factory);
}

std::unique_ptr<IStage> StageRegistry::Create(const std::string& name) {
    auto it = factories_.find(name);
    if (it == factories_.end()) return nullptr;
    return it->second();
}
