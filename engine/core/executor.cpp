#include "core/executor.hpp"

#include <chrono>

RunStats Execute(const std::vector<std::unique_ptr<IStage>>& chain, int max_frames) {
    RunStats stats;
    if (chain.empty()) return stats;

    using clock = std::chrono::steady_clock;
    double e2e_total_ms = 0.0;
    std::vector<double> frame_ms(chain.size(), 0.0);
    std::map<std::string, double> stage_total_ms;

    while (max_frames <= 0 || stats.frames < max_frames) {
        Frame frame;
        clock::time_point t0 = clock::now();

        bool completed = true;
        size_t i = 0;
        for (const auto& stage : chain) {
            clock::time_point ts = clock::now();
            bool more = stage->Process(frame);
            frame_ms[i] = std::chrono::duration<double, std::milli>(clock::now() - ts).count();
            frame.timing[stage->Id()] = frame_ms[i];
            ++i;
            if (!more) {   // source 耗尽（契约上只有 source 返回 false）
                completed = false;
                break;
            }
        }

        if (!completed) break;
        e2e_total_ms += std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        for (size_t k = 0; k < chain.size(); ++k) {
            stage_total_ms[chain[k]->Id()] += frame_ms[k];
        }
        ++stats.frames;
    }

    stats.e2e_avg_ms = stats.frames > 0 ? e2e_total_ms / stats.frames : 0.0;
    for (const auto& kv : stage_total_ms) {
        stats.avg_ms[kv.first] = stats.frames > 0 ? kv.second / stats.frames : 0.0;
    }
    return stats;
}
