// Task 5: image_dir 源的行为规格
#include "core/frame.hpp"
#include "core/registry.hpp"
#include "core/stage.hpp"
#include "stages/source_image_dir.hpp"

#include <cstdio>
#include <fstream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while(0)

static void write_ppm(const std::string& path) {
    std::ofstream f(path.c_str());
    // 最小合法 PPM(P6): 8x6 全红
    f << "P6\n8 6\n255\n";
    for (int i = 0; i < 8 * 6; ++i) { f << (char)255 << (char)0 << (char)0; }
}

int main() {
    // 准备 tmp 目录：两张图（乱序文件名，验证排序产出）
    std::string dir = "/tmp/rvs_test_src";
    if (system("rm -rf /tmp/rvs_test_src && mkdir -p /tmp/rvs_test_src") != 0) return 1;
    write_ppm(dir + "/b_second.ppm");
    write_ppm(dir + "/a_first.ppm");

    StageContext ctx;
    nlohmann::json params;
    params["path"] = dir;

    // ① 依次 Process 两帧，第三次 false
    {
        ImageDirSource src;
        CHECK(src.Init(params, ctx) == 0);
        Frame f;
        CHECK(src.Process(f) == true);
        CHECK(f.image.cols == 8 && f.image.rows == 6);
        CHECK(f.orig_w == 8 && f.orig_h == 6);
        CHECK(!f.image_name.empty());
        std::string first = f.image_name;
        CHECK(src.Process(f) == true);
        CHECK(f.image_name != first);          // 两帧不同文件
        CHECK(src.Process(f) == false);        // 耗尽
    }

    // ② loop + max_frames：4 帧上限
    {
        ImageDirSource src;
        params["loop"] = true;
        params["max_frames"] = 4;
        CHECK(src.Init(params, ctx) == 0);
        Frame f;
        int got = 0;
        while (src.Process(f)) ++got;
        CHECK(got == 4);
    }

    // ③ 空目录 Init 报错（返回非 0）
    if (system("mkdir -p /tmp/rvs_test_src_empty") != 0) return 1;
    {
        ImageDirSource src;
        nlohmann::json p2;
        p2["path"] = "/tmp/rvs_test_src_empty";
        CHECK(src.Init(p2, ctx) != 0);
    }

    // ④ 注册名可查
    CHECK(StageRegistry::Instance().Create("image_dir") != nullptr);

    printf("test_source: all checks passed\n");
    return 0;
}
