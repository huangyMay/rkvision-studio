// Task 4: 同步执行器的行为规格
#include "core/executor.hpp"
#include "core/frame.hpp"
#include "core/stage.hpp"

#include <cstdio>
#include <utility>
#include <memory>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

namespace {

class FakeSource : public IStage {
public:
    FakeSource(int n) : total_(n) {}
    int Init(const nlohmann::json&, const StageContext&) override { return 0; }
    bool Process(Frame& f) override {
        if (emitted_ >= total_) return false;
        ++emitted_;
        f.orig_w = 8; f.orig_h = 6;
        return true;
    }
    const StageMeta& Meta() const override {
        static StageMeta m{"fake_src", true, false, false};
        return m;
    }
private:
    int total_, emitted_ = 0;
};

class CountingStage : public IStage {
public:
    int calls = 0;
    int Init(const nlohmann::json&, const StageContext&) override { return 0; }
    bool Process(Frame& f) override { ++calls; f.timing[Id()] = 1.0; return true; }
    const StageMeta& Meta() const override {
        static StageMeta m{"fake_count", false, false, false};
        return m;
    }
};

class CountingSink : public CountingStage {
public:
    CountingSink() { sink_meta_.is_sink = true; sink_meta_.name = "fake_sink"; }
    const StageMeta& Meta() const override { return sink_meta_; }
private:
    StageMeta sink_meta_{"fake_sink", false, true, false};
};

class VerdictStage : public IStage {
public:
    explicit VerdictStage(std::vector<const char*> v) : v_(std::move(v)) {}
    int Init(const nlohmann::json&, const StageContext&) override { return 0; }
    bool Process(Frame& f) override {
        f.verdict = v_[i_ < v_.size() ? i_ : v_.size() - 1];
        ++i_;
        return true;
    }
    const StageMeta& Meta() const override {
        static StageMeta m{"fake_verdict", false, false, false};
        return m;
    }
private:
    std::vector<const char*> v_;
    size_t i_ = 0;
};

} // namespace

int main() {
    // ① source 产 3 帧 → mid/sink 各 3 次，RunStats.frames==3，e2e>0
    std::vector<std::unique_ptr<IStage>> chain;
    auto* src = new FakeSource(3);
    auto* mid = new CountingStage();
    auto* sink = new CountingSink();
    chain.emplace_back(src);
    chain.emplace_back(mid);
    chain.emplace_back(sink);
    RunStats stats = Execute(chain, /*max_frames=*/0);
    CHECK(stats.frames == 3);
    CHECK(mid->calls == 3);
    CHECK(sink->calls == 3);
    CHECK(stats.e2e_avg_ms > 0.0);

    // ② max_frames 限制（loop 型 source 用）：2 < 3
    std::vector<std::unique_ptr<IStage>> chain2;
    chain2.emplace_back(new FakeSource(3));
    chain2.emplace_back(new CountingStage());
    chain2.emplace_back(new CountingSink());
    RunStats s2 = Execute(chain2, /*max_frames=*/2);
    CHECK(s2.frames == 2);

    // ③ 空链 → frames==0 不崩
    std::vector<std::unique_ptr<IStage>> empty;
    RunStats s3 = Execute(empty, 0);
    CHECK(s3.frames == 0);

    // ④ verdict 聚合：逐帧 OK/NG/空 → ok_frames==1, ng_frames==1（Task 6）
    std::vector<std::unique_ptr<IStage>> chain4;
    chain4.emplace_back(new FakeSource(3));
    chain4.emplace_back(new VerdictStage({"OK", "NG", ""}));   // 未判决帧不计入
    RunStats s4 = Execute(chain4, 0);
    CHECK(s4.frames == 3);
    CHECK(s4.ok_frames == 1);
    CHECK(s4.ng_frames == 1);

    printf("test_executor: all checks passed\n");
    return 0;
}
