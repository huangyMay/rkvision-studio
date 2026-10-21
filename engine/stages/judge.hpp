// judge 判决插件：JudgeCore = 规则求值纯逻辑（IStage 壳 JudgeStage 在 Task 4 加入本文件）。
// v1 规则语义（docs/2026-10-18-工业工作流设计.md §2.1）：
//   score/size/region = 过滤器（按数组顺序收窄检出，框中心点判定、不 clamp）；
//   count/persist/regex = 判决规则（对过滤后结果评估，命中=违规）；
//   verdict 字段 ANY_NG/ALL_OK 在 v1 语义下等价，由壳层持有，字段为模板 C 预留。
#pragma once
#include <cfloat>
#include <climits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "core/frame.hpp"

struct JudgeRule {
    std::string type;                       // count|score|size|region|persist|regex
    std::string cls = "*";                  // "*" 通配
    int min = 0, max = INT_MAX;             // count 数量窗
    float thres = 0.f;                      // score 阈值
    float min_px = 0.f, max_px = FLT_MAX;   // size 面积窗
    std::vector<cv::Point2f> polygon;       // region
    bool inside = true;                     // region：true=保留域内
    int frames = 1;                         // persist 连续帧
    std::string pattern;                    // regex
};

struct JudgeVerdict {
    bool ng = false;
    nlohmann::json detail;                  // {"violations":[{rule,type,cls,actual}...]}
    std::vector<int> persist_counters;      // 与 persist 规则下标对齐（帧间状态，调用方持有）
};

struct JudgeCore {
    // 非法规则抛 std::runtime_error（what 含规则下标与键名）
    static std::vector<JudgeRule> ParseRules(const nlohmann::json& rules);

    // 过滤规则依序应用；判决规则跳过
    static std::vector<Detection> ApplyFilters(const std::vector<JudgeRule>& rules,
                                               const std::vector<Detection>& in);

    // 对过滤后的检出评估判决规则；persist_counters 由调用方跨帧持有
    static JudgeVerdict Evaluate(const std::vector<JudgeRule>& rules,
                                 const std::vector<Detection>& detections,
                                 std::vector<int>* persist_counters);

    static bool PointInPolygon(float px, float py, const std::vector<cv::Point2f>& poly);
};
