// Task 5: archive sink 行为规格（NG 证据存档 + 保留天数清理）
#include "core/registry.hpp"
#include "stages/sink_archive.hpp"

#include <cstdio>
#include <fstream>
#include <system_error>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

int main() {
    if (system("rm -rf /tmp/rvs_test_archive && mkdir -p /tmp/rvs_test_archive") != 0) return 1;

    // ① 注册名：主名 archive + 设计稿 §3.1/§6 的别名 sink_archive
    CHECK(StageRegistry::Instance().Create("archive") != nullptr);
    CHECK(StageRegistry::Instance().Create("sink_archive") != nullptr);

    // ② NG 帧落 jpg+json；OK 帧不落
    {
        StageContext ctx;
        ctx.project_root = "/tmp/rvs_test_archive";
        ArchiveSink s;
        nlohmann::json p = {{"ng_dir", "ng_images"}};
        CHECK(s.Init(p, ctx) == 0);

        Frame f;
        f.image.create(32, 32, CV_8UC3);
        f.image_name = "bad";
        f.verdict = "NG";
        f.judge = nlohmann::json{{"violations", nlohmann::json::array({nlohmann::json{{"type", "count"}}})}};
        CHECK(s.Process(f) == true);
        std::ifstream jf("/tmp/rvs_test_archive/ng_images/bad_ng.json");
        CHECK(jf.good());
        nlohmann::json j = nlohmann::json::parse(jf);
        CHECK(j["violations"].size() == 1);              // frame.judge 全文
        std::ifstream imgf("/tmp/rvs_test_archive/ng_images/bad_ng.jpg");
        CHECK(imgf.good());

        Frame ok;
        ok.image.create(32, 32, CV_8UC3);
        ok.image_name = "good";
        ok.verdict = "OK";
        CHECK(s.Process(ok) == true);
        std::ifstream j2("/tmp/rvs_test_archive/ng_images/good_ng.json");
        CHECK(!j2.good());                               // OK 帧不落盘
    }

    // ③ keep_days：Init 清理过期文件（mtime 超 keep_days），新文件保留
    {
        if (system("mkdir -p /tmp/rvs_test_archive/ng2 && "
                   "echo x > /tmp/rvs_test_archive/ng2/old_ng.jpg && "
                   "touch -d '3 days ago' /tmp/rvs_test_archive/ng2/old_ng.jpg && "
                   "echo x > /tmp/rvs_test_archive/ng2/new_ng.jpg") != 0) return 1;
        StageContext ctx;
        ctx.project_root = "/tmp/rvs_test_archive";
        ArchiveSink s;
        nlohmann::json p = {{"ng_dir", "ng2"}, {"keep_days", 1}};
        CHECK(s.Init(p, ctx) == 0);
        std::ifstream oldf("/tmp/rvs_test_archive/ng2/old_ng.jpg");
        CHECK(!oldf.good());                             // 过期已清
        std::ifstream newf("/tmp/rvs_test_archive/ng2/new_ng.jpg");
        CHECK(newf.good());                              // 未过期保留
    }

    // ④ ng_dir 不可写：Process 仍返回 true，不阻断流水线（Review Focus 4）
    {
        if (system("rm -rf /tmp/rvs_test_archive/ro && mkdir -p /tmp/rvs_test_archive/ro && "
                   "chmod 555 /tmp/rvs_test_archive/ro") != 0) return 1;
        StageContext ctx;
        ctx.project_root = "/tmp/rvs_test_archive";
        ArchiveSink s;
        nlohmann::json p = {{"ng_dir", "ro"}};
        CHECK(s.Init(p, ctx) == 0);
        Frame f;
        f.image.create(32, 32, CV_8UC3);
        f.image_name = "x";
        f.verdict = "NG";
        CHECK(s.Process(f) == true);
        if (system("chmod 755 /tmp/rvs_test_archive/ro") != 0) return 1;
    }

    printf("test_archive: all checks passed\n");
    return 0;
}
