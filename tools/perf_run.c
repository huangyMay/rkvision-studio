/*
 * perf_run.c —— RKNN 板端性能剖析器（Studio 性能剖析页配套，2026-10-25）
 *
 * 分段计时一张模型的每帧成本，把"一条龙 inference"拆成三段税：
 *   set = rknn_inputs_set（归一化 RNE / 布局拷贝税）
 *   run = rknn_run（纯 NPU 执行——架构对比该看的数字）
 *   get = rknn_outputs_get（want_float=1 的 fp32 展开回传税）
 *
 * 输入用随机 uint8（NPU 卷积计算量与数据内容无关，性能测试无需真图）。
 * 预热 5 帧后计 N 帧 avg/min，单行 JSON 到 stdout。
 *
 * 用法: perf_run <model.rknn> <loops> <core_mask>
 *   core_mask: 0=auto 1=npu0 2=npu1 3=npu0+1 4=npu2 7=npu0+1+2（按板型支持）
 *
 * 编译:
 *   Android(3576/3588-android): NDK clang -I<rknn_api.h> -L<librknnrt dir> -lrknnrt
 *   Debian(3588):              gcc 同上（板上自编或 aarch64-linux-gnu-gcc）
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rknn_api.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double avg(const double *v, int n) { double s = 0; for (int i = 0; i < n; i++) s += v[i]; return s / n; }
static double mn(const double *v, int n) { double m = v[0]; for (int i = 1; i < n; i++) if (v[i] < m) m = v[i]; return m; }

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: perf_run <model.rknn> <loops> <core_mask>\n");
        return 1;
    }
    const char *model_path = argv[1];
    int loops = atoi(argv[2]);
    int core_mask = atoi(argv[3]);
    if (loops < 1) loops = 30;
    if (loops > 2000) loops = 2000;

    rknn_context ctx;
    int ret = rknn_init(&ctx, (void *)model_path, 0, 0, NULL);
    if (ret < 0) { printf("{\"ok\":false,\"stage\":\"init\",\"ret\":%d}\n", ret); return 1; }

    if (core_mask > 0) {
        ret = rknn_set_core_mask(ctx, (rknn_core_mask)core_mask);
        if (ret < 0) { printf("{\"ok\":false,\"stage\":\"set_core_mask\",\"ret\":%d}\n", ret); return 1; }
    }

    rknn_input_output_num io;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (ret < 0) { printf("{\"ok\":false,\"stage\":\"query_num\",\"ret\":%d}\n", ret); return 1; }
    rknn_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret < 0) { printf("{\"ok\":false,\"stage\":\"query_attr\",\"ret\":%d}\n", ret); return 1; }

    /* u8 输入：attr.size 即字节总数（n_elems×1）；fmt 沿用 attr.fmt，不猜维度顺序 */
    const uint32_t in_size = attr.size;
    unsigned char *data = (unsigned char *)malloc(in_size);
    unsigned int seed = 42;
    for (uint32_t i = 0; i < in_size; i++) { seed = seed * 1103515245 + 12345; data[i] = (unsigned char)(seed >> 16); }

    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.buf = data;
    input.size = in_size;
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = attr.fmt;

    rknn_output *outputs = (rknn_output *)calloc(io.n_output, sizeof(rknn_output));

    /* 预热 5 帧（库内部 lazy init / 缓存 / 频率爬升） */
    for (int i = 0; i < 5; i++) {
        rknn_inputs_set(ctx, 1, &input);
        if (rknn_run(ctx, NULL) < 0) { printf("{\"ok\":false,\"stage\":\"warmup_run\"}\n"); return 1; }
        if (rknn_outputs_get(ctx, io.n_output, outputs, NULL) < 0) { printf("{\"ok\":false,\"stage\":\"warmup_get\"}\n"); return 1; }
        rknn_outputs_release(ctx, io.n_output, outputs);
    }

    double *t_set = malloc(sizeof(double) * loops);
    double *t_run = malloc(sizeof(double) * loops);
    double *t_get = malloc(sizeof(double) * loops);
    for (int i = 0; i < loops; i++) {
        double t0 = now_ms();
        if (rknn_inputs_set(ctx, 1, &input) < 0) { printf("{\"ok\":false,\"stage\":\"set\",\"frame\":%d}\n", i); return 1; }
        double t1 = now_ms();
        if (rknn_run(ctx, NULL) < 0) { printf("{\"ok\":false,\"stage\":\"run\",\"frame\":%d}\n", i); return 1; }
        double t2 = now_ms();
        if (rknn_outputs_get(ctx, io.n_output, outputs, NULL) < 0) { printf("{\"ok\":false,\"stage\":\"get\",\"frame\":%d}\n", i); return 1; }
        double t3 = now_ms();
        rknn_outputs_release(ctx, io.n_output, outputs);
        t_set[i] = t1 - t0; t_run[i] = t2 - t1; t_get[i] = t3 - t2;
    }

    double set_a = avg(t_set, loops), run_a = avg(t_run, loops), get_a = avg(t_get, loops);
    double e2e = set_a + run_a + get_a;
    printf("{\"ok\":true,\"core_mask\":%d,\"frames\":%d,"
           "\"set_ms\":%.3f,\"run_ms\":%.3f,\"get_ms\":%.3f,\"e2e_ms\":%.3f,"
           "\"set_min\":%.3f,\"run_min\":%.3f,\"get_min\":%.3f,"
           "\"fps\":%.2f,\"n_in\":%d,\"n_out\":%d}\n",
           core_mask, loops,
           set_a, run_a, get_a, e2e,
           mn(t_set, loops), mn(t_run, loops), mn(t_get, loops),
           e2e > 0 ? 1000.0 / e2e : 0.0, io.n_input, io.n_output);

    free(t_set); free(t_run); free(t_get);
    free(outputs); free(data);
    rknn_destroy(ctx);
    return 0;
}
