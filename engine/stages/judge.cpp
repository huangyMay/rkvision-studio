#include "stages/judge.hpp"

#include <cctype>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/imgcodecs.hpp>

#include "core/registry.hpp"

REGISTER_STAGE("judge", JudgeStage)

static void MkdirP(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' && i > 0) mkdir(cur.c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

int JudgeStage::Init(const nlohmann::json& params, const StageContext& ctx) {
    if (!params.is_object()) {
        fprintf(stderr, "judge: params must be an object\n");
        return -1;
    }
    try {
        rules_ = JudgeCore::ParseRules(params.value("rules", nlohmann::json::array()));
    } catch (const std::exception& e) {
        fprintf(stderr, "judge: %s\n", e.what());
        return -1;
    }
    verdict_mode_ = params.value("verdict", "ANY_NG");
    if (verdict_mode_ != "ANY_NG" && verdict_mode_ != "ALL_OK") {
        fprintf(stderr, "judge: verdict must be ANY_NG or ALL_OK, got %s\n", verdict_mode_.c_str());
        return -1;
    }
    actions_.clear();
    if (params.contains("actions")) {
        if (!params["actions"].is_array()) {
            fprintf(stderr, "judge: actions must be an array\n");
            return -1;
        }
        for (const auto& a : params["actions"]) {
            if (!a.is_string()) {
                fprintf(stderr, "judge: actions entries must be strings\n");
                return -1;
            }
            std::string s = a.get<std::string>();
            if (s.rfind("plc_write:", 0) == 0 || s.rfind("http:", 0) == 0) {
                fprintf(stderr, "judge: action '%s' needs M-c (Modbus/HTTP), not supported in v1\n",
                        s.c_str());
                return -1;
            }
            if (s.rfind("snapshot:", 0) == 0) {
                std::string dir = s.substr(strlen("snapshot:"));
                if (!dir.empty() && dir[0] != '/' && !ctx.project_root.empty())
                    dir = ctx.project_root + "/" + dir;
                actions_.push_back("snapshot:" + dir);
            } else if (s.rfind("log:", 0) == 0) {
                actions_.push_back(s);
            } else {
                fprintf(stderr, "judge: unknown action '%s' (v1: log:/snapshot:)\n", s.c_str());
                return -1;
            }
        }
    }
    counters_.assign(rules_.size(), 0);
    return 0;
}

void JudgeStage::RunActions(const Frame& frame, bool ng) {
    for (const std::string& a : actions_) {
        if (a.rfind("log:", 0) == 0) {
            fprintf(stderr, "judge[%s] %s: %s\n", ng ? "NG" : "OK",
                    frame.image_name.c_str(), a.c_str() + strlen("log:"));
        } else if (a.rfind("snapshot:", 0) == 0) {
            const std::string dir = a.substr(strlen("snapshot:"));
            MkdirP(dir);
            const cv::Mat img = frame.orig_image.empty() ? frame.image : frame.orig_image;
            const std::string base = dir + "/" +
                (frame.image_name.empty() ? "frame" : frame.image_name) + "_ng.jpg";
            if (!cv::imwrite(base, img))
                fprintf(stderr, "judge: snapshot imwrite failed: %s\n", base.c_str());
        }
    }
}

bool JudgeStage::Process(Frame& frame) {
    const size_t before = frame.detections.size();
    frame.detections = JudgeCore::ApplyFilters(rules_, frame.detections);
    const JudgeVerdict v = JudgeCore::Evaluate(rules_, frame.detections, &counters_);
    frame.verdict = v.ng ? "NG" : "OK";
    frame.judge = v.detail;
    frame.judge["filtered"] = before - frame.detections.size();
    if (v.ng) RunActions(frame, true);
    return true;
}

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
            try {                          // Init 期预编译校验，非法 pattern 在此报错而非首帧 abort
                std::regex re;
                re.assign(j.pattern);
            } catch (const std::regex_error&) {
                throw std::runtime_error("judge.rules[" + std::to_string(i) +
                                         "].pattern invalid regex: " + j.pattern);
            }
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
    auto cls_match = [](const JudgeRule& r, const Detection& d) {
        return r.cls == "*" || d.label == r.cls;
    };
    for (const JudgeRule& r : rules) {
        if (r.type == "score") {
            std::vector<Detection> next;
            for (const Detection& d : cur)
                if (!cls_match(r, d) || d.score >= r.thres) next.push_back(d);
            cur.swap(next);
        } else if (r.type == "size") {
            std::vector<Detection> next;
            for (const Detection& d : cur) {
                const float area = (d.x2 - d.x1) * (d.y2 - d.y1);
                if (!cls_match(r, d) || (area >= r.min_px && area <= r.max_px)) next.push_back(d);
            }
            cur.swap(next);
        } else if (r.type == "region") {
            std::vector<Detection> next;
            for (const Detection& d : cur) {
                if (!cls_match(r, d)) { next.push_back(d); continue; }
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
