// 同步单链执行器：source 产帧 → 依级 Process → 计时统计。
// 线程池/多模型流水是 M3 的事，M0 保持最简。
#pragma once
#include <map>
#include <string>
#include <vector>

#include "core/stage.hpp"

struct RunStats {
    int frames = 0;
    int ok_frames = 0;                      // verdict=="OK" 的帧（judge 在链上时）
    int ng_frames = 0;                      // verdict=="NG" 的帧
    std::map<std::string, double> avg_ms;   // 各 stage 的每帧平均耗时
    double e2e_avg_ms = 0.0;
};

// max_frames>0 时作为帧数上限（配合 loop 型 source）；<=0 表示跑到 source 耗尽
RunStats Execute(const std::vector<std::unique_ptr<IStage>>& chain, int max_frames);
