// RunProject：rvs_engine 的库层入口（main 只是 argv 薄壳，单测直接打这里）。
#pragma once
#include <string>

// 返回 0 成功；非 0 失败（2=spec 错、3=模型文件缺失、4=装配失败）。
// 成功时 summary_json 输出一行汇总（含 frames / e2e_avg_ms）。
int RunProject(const std::string& project_dir, std::string* summary_json);
