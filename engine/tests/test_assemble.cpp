// Task 3: 注册表 + 装配器的行为规格
#include "core/assembler.hpp"
#include "core/frame.hpp"
#include "core/registry.hpp"
#include "core/stage.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

#define CHECK_THROWS_MSG(expr, needle) do { \
    bool threw_ = false; std::string msg_; \
    try { expr; } catch (const std::exception& e) { threw_ = true; msg_ = e.what(); } \
    if (!threw_) { fprintf(stderr, "expected throw at line %d\n", __LINE__); return 1; } \
    if (msg_.find(needle) == std::string::npos) { \
        fprintf(stderr, "line %d: what()='%s' lacks '%s'\n", __LINE__, msg_.c_str(), needle); return 1; } \
} while(0)

namespace {

class FakeStage : public IStage {
public:
    FakeStage(bool is_source, bool is_sink, std::string name)
        : meta_{"", is_source, is_sink, false}, name_(std::move(name)) {}
    int Init(const nlohmann::json&, const StageContext&) override { return 0; }
    bool Process(Frame&) override { return true; }
    const StageMeta& Meta() const override {
        static StageMeta m{"", false, false, false};
        m.name = name_.c_str(); m.is_source = meta_.is_source; m.is_sink = meta_.is_sink;
        return m;
    }
private:
    StageMeta meta_;
    std::string name_;
};

class FakeSrc : public FakeStage { public: FakeSrc() : FakeStage(true, false, "fake_src") {} };
class FakeMid : public FakeStage { public: FakeMid() : FakeStage(false, false, "fake_mid") {} };
class FakeSink : public FakeStage { public: FakeSink() : FakeStage(false, true, "fake_sink") {} };

// 注册表探测：注册即在本文件静态期完成
bool g_reg = true;
struct Reg {
    Reg() {
        StageRegistry::Instance().Register("fake_src", [] { return std::unique_ptr<IStage>(new FakeSrc()); });
        StageRegistry::Instance().Register("fake_mid", [] { return std::unique_ptr<IStage>(new FakeMid()); });
        StageRegistry::Instance().Register("fake_sink", [] { return std::unique_ptr<IStage>(new FakeSink()); });
    }
} g_reg_instance;

PipelineSpec make_spec(std::initializer_list<const char*> plugins) {
    PipelineSpec spec;
    spec.version = 1;
    spec.name = "t";
    int i = 0;
    for (const char* p : plugins) {
        StageSpec s;
        s.id = "s" + std::to_string(i++);
        s.plugin = p;
        spec.pipeline.push_back(s);
    }
    return spec;
}

} // namespace

int main() {
    StageContext ctx;
    ctx.spec = nullptr;
    ctx.project_root = "/tmp";

    // ① 合法链（src→mid→sink）：长度与顺序正确
    PipelineSpec spec = make_spec({"fake_src", "fake_mid", "fake_sink"});
    auto chain = Assemble(spec, ctx);
    CHECK(chain.size() == 3);
    CHECK(chain[0]->Id() == "s0");
    CHECK(chain[0]->Meta().is_source == true);
    CHECK(chain[1]->Id() == "s1");
    CHECK(chain[2]->Id() == "s2");
    CHECK(chain[2]->Meta().is_sink == true);

    // ② 未知插件 → 抛错含 plugin 名与 stage id
    PipelineSpec bad = make_spec({"fake_src", "nope_plugin", "fake_sink"});
    bad.pipeline[1].id = "mystery";
    CHECK_THROWS_MSG(Assemble(bad, ctx), "nope_plugin");
    CHECK_THROWS_MSG(Assemble(bad, ctx), "mystery");

    // ③ Create 未知名 → nullptr
    CHECK(StageRegistry::Instance().Create("no_such_stage") == nullptr);
    CHECK(StageRegistry::Instance().Create("fake_mid") != nullptr);

    // ④ 首级不是 source → 抛错含 "source"
    PipelineSpec nosrc = make_spec({"fake_mid", "fake_sink"});
    CHECK_THROWS_MSG(Assemble(nosrc, ctx), "source");

    // ⑤ 末级不是 sink → 抛错含 "sink"
    PipelineSpec nosink = make_spec({"fake_src", "fake_mid"});
    CHECK_THROWS_MSG(Assemble(nosink, ctx), "sink");

    // ⑥ 空 pipeline → 抛错
    PipelineSpec empty = make_spec({});
    CHECK_THROWS_MSG(Assemble(empty, ctx), "pipeline");

    // ⑦ model.family 与后处理插件不一致 → 抛错（family 不再是"只解析不消费"的摆设）
    PipelineSpec fam = make_spec({"fake_src", "yolov5_det", "fake_sink"});
    fam.pipeline[0].model = "det";
    ModelSpec ms;
    ms.family = "yolov8-det";
    ms.file = "models/x.rknn";
    fam.models["det"] = ms;
    CHECK_THROWS_MSG(Assemble(fam, ctx), "family");

    printf("test_assemble: all checks passed\n");
    return 0;
}
