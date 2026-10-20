#include "core/run_project.hpp"

#include <cstdio>
#include <memory>
#include <vector>

#include "core/assembler.hpp"
#include "core/executor.hpp"
#include "core/spec.hpp"
#include "core/stage.hpp"

int RunProject(const std::string& project_dir, std::string* summary_json) {
    PipelineSpec spec;
    try {
        spec = load_spec(project_dir + "/pipeline.json");
    } catch (const std::exception& e) {
        fprintf(stderr, "rvs_engine: %s\n", e.what());
        return 2;
    }

    // 模型文件预检（板端 infer 才真正加载；host 桩也要求工程包自洽）
    for (const auto& kv : spec.models) {
        std::string p = kv.second.file;
        if (!p.empty() && p[0] != '/') p = project_dir + "/" + p;
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "rvs_engine: model file not found: %s (model key: %s)\n",
                    p.c_str(), kv.first.c_str());
            return 3;
        }
        fclose(f);
    }

    StageContext ctx;
    ctx.spec = &spec;
    ctx.project_root = project_dir;

    std::vector<std::unique_ptr<IStage>> chain;
    try {
        chain = Assemble(spec, ctx);
    } catch (const std::exception& e) {
        fprintf(stderr, "rvs_engine: %s\n", e.what());
        return 4;
    }

    RunStats stats = Execute(chain, 0);
    char buf[512];
    snprintf(buf, sizeof(buf), "{\"frames\":%d,\"e2e_avg_ms\":%.2f}", stats.frames, stats.e2e_avg_ms);
    if (summary_json) *summary_json = buf;
    printf("%s\n", buf);
    return 0;
}
