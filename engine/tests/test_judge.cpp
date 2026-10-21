// Task 2/3: JudgeCore 规则求值行为规格（过滤器 + 判决规则 + persist 状态机）
#include "stages/judge.hpp"

#include <cstdio>
#include <stdexcept>

#include <nlohmann/json.hpp>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

static Detection Det(const char* label, float score, float x1, float y1, float x2, float y2) {
    Detection d;
    d.label = label;
    d.score = score;
    d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
    return d;
}

int main() {
    using nlohmann::json;

    // ① score 过滤：低于阈值滤除
    {
        auto rules = JudgeCore::ParseRules(json::array({json{{"type", "score"}, {"thres", 0.5}}}));
        auto out = JudgeCore::ApplyFilters(rules, {Det("a", 0.9f, 0, 0, 10, 10), Det("a", 0.3f, 0, 0, 10, 10)});
        CHECK(out.size() == 1);
        CHECK(out[0].score > 0.5f);
    }

    // ② size 过滤：面积窗 [min_px, max_px]
    {
        auto rules = JudgeCore::ParseRules(json::array({json{{"type", "size"}, {"min_px", 20}, {"max_px", 5000}}}));
        auto out = JudgeCore::ApplyFilters(rules, {Det("a", 0.9f, 0, 0, 15, 10),     // 150 保留
                                                   Det("a", 0.9f, 0, 0, 2, 2),       // 4 滤除
                                                   Det("a", 0.9f, 0, 0, 100, 100)}); // 10000 滤除
        CHECK(out.size() == 1);
    }

    // ③ region 过滤：框中心点 ray casting；凸/凹多边形；负坐标框不 clamp
    {
        auto rules = JudgeCore::ParseRules(json::array({
            json{{"type", "region"}, {"polygon", {{0,0},{100,0},{50,80}}}, {"inside", true}}}));
        auto out = JudgeCore::ApplyFilters(rules, {Det("a", 0.9f, 40, 20, 60, 40),    // 中心(50,30) 内
                                                   Det("a", 0.9f, 80, 60, 100, 80)}); // 中心(90,70) 外
        CHECK(out.size() == 1);
    }
    {
        // L 形凹多边形（实体=底条 y<50 ∪ 右列 x>50）：(75,25) 在实体内、(25,75) 在缺口内
        auto rules = JudgeCore::ParseRules(json::array({
            json{{"type", "region"},
                 {"polygon", {{0,0},{100,0},{100,100},{50,100},{50,50},{0,50}}},
                 {"inside", true}}}));
        auto out = JudgeCore::ApplyFilters(rules, {Det("a", 0.9f, 70, 20, 80, 30),   // (75,25) 内
                                                   Det("a", 0.9f, 20, 70, 30, 80)}); // (25,75) 缺口→外
        CHECK(out.size() == 1);
    }
    {
        // 负坐标框中心 (-5,10)：region 按原始中心点判定，不 clamp（Review Focus 5）
        auto rules = JudgeCore::ParseRules(json::array({
            json{{"type", "region"}, {"polygon", {{-20,-20},{20,-20},{20,20},{-20,20}}}, {"inside", true}}}));
        auto out = JudgeCore::ApplyFilters(rules, {Det("a", 0.9f, -10, 0, 0, 20)});
        CHECK(out.size() == 1);
    }
    {
        // inside=false：只保留多边形外的检出
        auto rules = JudgeCore::ParseRules(json::array({
            json{{"type", "region"}, {"polygon", {{0,0},{10,0},{10,10},{0,10}}}, {"inside", false}}}));
        auto out = JudgeCore::ApplyFilters(rules, {Det("a", 0.9f, 1, 1, 5, 5),        // 内→滤
                                                   Det("a", 0.9f, 20, 20, 30, 30)});  // 外→留
        CHECK(out.size() == 1 && out[0].x1 == 20);
    }

    // ④ region 退化输入：多边形点数 <3 → ParseRules 抛异常（Review Focus 1）
    {
        bool threw = false;
        try {
            JudgeCore::ParseRules(json::array({json{{"type", "region"}, {"polygon", {{0,0},{1,1}}}}}));
        } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }

    // ⑤ count 判决：数量越窗=违规，actual 记实际值；"*" 通配计全部类别
    {
        auto rules = JudgeCore::ParseRules(json::array({
            json{{"type", "count"}, {"class", "scratch"}, {"min", 1}, {"max", 2}}}));
        std::vector<int> counters(rules.size(), 0);
        auto v = JudgeCore::Evaluate(rules, {Det("scratch", 0.9f, 0, 0, 5, 5),
                                             Det("scratch", 0.8f, 6, 0, 9, 5),
                                             Det("scratch", 0.7f, 10, 0, 14, 5)}, &counters);
        CHECK(v.ng);
        CHECK(v.detail["violations"].size() == 1);
        CHECK(v.detail["violations"][0]["actual"] == 3);

        auto v2 = JudgeCore::Evaluate(rules, {Det("scratch", 0.9f, 0, 0, 5, 5)}, &counters);
        CHECK(!v2.ng);
    }
    {
        auto rules = JudgeCore::ParseRules(json::array({
            json{{"type", "count"}, {"class", "*"}, {"min", 5}, {"max", 99}}}));
        std::vector<int> counters(rules.size(), 0);
        auto v = JudgeCore::Evaluate(rules, {Det("helmet", 0.9f, 0, 0, 5, 5),
                                             Det("head", 0.9f, 6, 0, 9, 5),
                                             Det("scratch", 0.7f, 10, 0, 14, 5)}, &counters);
        CHECK(v.ng);                                    // 通配计 3 < 5 → 违规
        CHECK(v.detail["violations"][0]["actual"] == 3);
    }

    // ⑥ regex 判决：对 label 内容校验（OCR 文本 M-c 落 text 字段前的 v1 载体）
    {
        auto rules = JudgeCore::ParseRules(json::array({
            json{{"type", "regex"}, {"class", "*"}, {"pattern", "20[0-9]{2}-[01][0-9]-[0-3][0-9]"}}}));
        std::vector<int> counters(rules.size(), 0);
        auto v = JudgeCore::Evaluate(rules, {Det("2025-10-20", 0.9f, 0, 0, 5, 5)}, &counters);
        CHECK(!v.ng);
        auto v2 = JudgeCore::Evaluate(rules, {Det("2025-13-45", 0.9f, 0, 0, 5, 5)}, &counters);
        CHECK(v2.ng);
    }

    // ⑦ 零检出：不崩；"缺席型"规则（min 0,max 0=有检出才违规）恒 OK；
    //    对照：min>=1 的"必须在场型"规则对零检出判违规（count 语义正确性）
    {
        auto rules = JudgeCore::ParseRules(json::array({
            json{{"type", "count"}, {"class", "*"}, {"min", 0}, {"max", 0}}}));
        std::vector<int> counters(rules.size(), 0);
        auto v = JudgeCore::Evaluate(rules, {}, &counters);
        CHECK(!v.ng);
        auto v2 = JudgeCore::Evaluate(rules, {Det("scratch", 0.9f, 0, 0, 5, 5)}, &counters);
        CHECK(v2.ng);                                   // 有检出越 [0,0] 窗 → 违规
    }

    printf("test_judge: all checks passed\n");
    return 0;
}
