// attrs_probe：加载 .rknn，打印输出张量量化参数（zp/scale）JSON。
// 用途：金标准回归与板上调试的一次性物料提取。
// 用法: attrs_probe <model.rknn> [n_outputs]
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rknn_api.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: attrs_probe <model.rknn> [n_outputs]\n");
        return 1;
    }
    int n_outputs = argc > 2 ? atoi(argv[2]) : 3;

    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { printf("cannot open %s\n", argv[1]); return 1; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void* model = malloc(sz);
    if (fread(model, 1, sz, fp) != (size_t)sz) { fclose(fp); free(model); return 1; }
    fclose(fp);

    rknn_context ctx;
    int ret = rknn_init(&ctx, model, sz, 0, NULL);
    free(model);
    if (ret < 0) { printf("rknn_init failed ret=%d\n", ret); return 1; }

    printf("{\"zps\":[");
    for (int i = 0; i < n_outputs; ++i) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret < 0) { printf("],\"error\":\"query %d failed ret=%d\"}\n", i, ret); return 1; }
        printf("%s%d", i ? "," : "", attr.zp);
    }
    printf("],\"scales\":[");
    for (int i = 0; i < n_outputs; ++i) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        printf("%s%.10g", i ? "," : "", attr.scale);
    }
    printf("]}\n");
    rknn_destroy(ctx);
    return 0;
}
