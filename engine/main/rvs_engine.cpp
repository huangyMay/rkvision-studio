// rvs_engine：读工程包、装配流水线、跑完、打一行汇总 JSON。
#include "core/run_project.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: rvs_engine <project_dir>\n");
        return 1;
    }
    return RunProject(argv[1], nullptr);
}
