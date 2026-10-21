#include "stages/judge.hpp"

#include <cctype>

std::vector<JudgeRule> JudgeCore::ParseRules(const nlohmann::json& rules) {
    if (!rules.is_array()) throw std::runtime_error("judge.rules must be an array");
    std::vector<JudgeRule> out;
    for (size_t i = 0; i < rules.size(); ++i) {
        const auto& r = rules[i];
        if (!r.is_object() || !r.contains("type"))
            throw std::runtime_error("judge.rules[" + std::to_string(i) + "].type is required");
        JudgeRule j;
        j.type = r["type"].get<std::string>();
        j.cls = r.value("class", "*");
        if (j.type == "count") {
            j.min = r.value("min", 0);
            j.max = r.value("max", INT_MAX);
        } else if (j.type == "score") {
            j.thres = r.value("thres", 0.f);
        } else if (j.type == "size") {
            j.min_px = r.value("min_px", 0.f);
            j.max_px = r.value("max_px", FLT_MAX);
        } else if (j.type == "region") {
            if (!r.contains("polygon") || !r["polygon"].is_array() || r["polygon"].size() < 3)
                throw std::runtime_error("judge.rules[" + std::to_string(i) +
                                         "].polygon needs >=3 [x,y] points");
            for (const auto& pt : r["polygon"]) {
                if (!pt.is_array() || pt.size() < 2)
                    throw std::runtime_error("judge.rules[" + std::to_string(i) +
                                             "].polygon point must be [x,y]");
                j.polygon.emplace_back(pt[0].get<float>(), pt[1].get<float>());
            }
            j.inside = r.value("inside", true);
        } else if (j.type == "persist") {
            j.frames = r.value("frames", 1);
            if (j.frames < 1) j.frames = 1;
        } else if (j.type == "regex") {
            j.pattern = r.value("pattern", "");
            if (j.pattern.empty())
                throw std::runtime_error("judge.rules[" + std::to_string(i) + "].pattern is required");
        } else {
            throw std::runtime_error("judge.rules[" + std::to_string(i) +
                                     "].type unknown: " + j.type);
        }
        out.push_back(std::move(j));
    }
    return out;
}

bool JudgeCore::PointInPolygon(float px, float py, const std::vector<cv::Point2f>& poly) {
    bool inside = false;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
        const float xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
        if (((yi > py) != (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi))
            inside = !inside;
    }
    return inside;
}

std::vector<Detection> JudgeCore::ApplyFilters(const std::vector<JudgeRule>& rules,
                                               const std::vector<Detection>& in) {
    std::vector<Detection> cur = in;
    for (const JudgeRule& r : rules) {
        if (r.type == "score") {
            std::vector<Detection> next;
            for (const Detection& d : cur)
                if (d.score >= r.thres) next.push_back(d);
            cur.swap(next);
        } else if (r.type == "size") {
            std::vector<Detection> next;
            for (const Detection& d : cur) {
                const float area = (d.x2 - d.x1) * (d.y2 - d.y1);
                if (area >= r.min_px && area <= r.max_px) next.push_back(d);
            }
            cur.swap(next);
        } else if (r.type == "region") {
            std::vector<Detection> next;
            for (const Detection& d : cur) {
                const float cx = (d.x1 + d.x2) / 2.f, cy = (d.y1 + d.y2) / 2.f;
                const bool pin = PointInPolygon(cx, cy, r.polygon);
                if (pin == r.inside) next.push_back(d);
            }
            cur.swap(next);
        }
    }
    return cur;
}

JudgeVerdict JudgeCore::Evaluate(const std::vector<JudgeRule>& rules,
                                 const std::vector<Detection>& detections,
                                 std::vector<int>* persist_counters) {
    JudgeVerdict v;
    nlohmann::json violations = nlohmann::json::array();

    auto cls_match = [](const JudgeRule& r, const Detection& d) {
        return r.cls == "*" || d.label == r.cls;
    };

    for (size_t i = 0; i < rules.size(); ++i) {
        const JudgeRule& r = rules[i];
        if (r.type == "count") {
            int n = 0;
            for (const Detection& d : detections)
                if (cls_match(r, d)) ++n;
            if (n < r.min || n > r.max)
                violations.push_back({{"rule", i}, {"type", "count"},
                                      {"cls", r.cls}, {"actual", n}});
        } else if (r.type == "regex") {
            std::regex re;
            try {
                re.assign(r.pattern);
            } catch (const std::regex_error&) {
                throw std::runtime_error("judge.rules[" + std::to_string(i) +
                                         "].pattern invalid regex: " + r.pattern);
            }
            int bad = 0;
            for (const Detection& d : detections)
                if (cls_match(r, d) && !std::regex_search(d.label, re)) ++bad;
            if (bad > 0)
                violations.push_back({{"rule", i}, {"type", "regex"},
                                      {"cls", r.cls}, {"actual", bad}});
        } else if (r.type == "persist") {
            // 去抖状态机：连续 frames 帧有匹配检出计数 +1，断流清零；
            // 达阈值置违规并锁存（counter=-1），锁存后每帧 ng=true 直至流水线重建
            if (persist_counters->size() <= i) persist_counters->resize(i + 1, 0);
            int& c = (*persist_counters)[i];
            const bool hit = std::any_of(detections.begin(), detections.end(),
                                         [&](const Detection& d) { return cls_match(r, d); });
            if (c < 0) {                     // 已锁存
                v.ng = true;
            } else if (hit) {
                ++c;
                if (c >= r.frames) {
                    c = -1;
                    v.ng = true;
                    violations.push_back({{"rule", i}, {"type", "persist"},
                                          {"cls", r.cls}, {"actual", r.frames}});
                }
            } else {
                c = 0;
            }
        }
    }

    v.detail = {{"violations", violations}};
    v.ng = v.ng || !violations.empty();
    return v;
}
