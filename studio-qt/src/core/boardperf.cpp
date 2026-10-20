#include "boardperf.h"
#include "adbclient.h"

#include <QDir>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QJsonDocument>
#include <QJsonObject>
#include <algorithm>
#include <QtConcurrent/QtConcurrent>

namespace {
constexpr auto kRemoteProj = "/data/local/tmp/rvs/perf_test";
constexpr auto kRemoteRoot = "/data/local/tmp/rvs";

// 从引擎 stdout 抓 {"frames":N,"e2e_avg_ms":X}
double parseE2E(const QString &out, int *frames) {
    const int k = out.lastIndexOf(QStringLiteral("{\"frames\""));
    if (k < 0) return -1;
    const QJsonObject o = QJsonDocument::fromJson(out.mid(k).toUtf8()).object();
    if (o.isEmpty()) return -1;
    if (frames) *frames = o.value(QStringLiteral("frames")).toInt();
    return o.value(QStringLiteral("e2e_avg_ms")).toDouble();
}
} // namespace

BoardPerfRunner::BoardPerfRunner(QObject *parent) : QObject(parent) {}

bool BoardPerfRunner::running() const { return m_running; }

void BoardPerfRunner::run(const QString &serial, const QString &rknnLocal,
                          const QString &imageLocal, const QString &remoteRoot,
                          const QString &engineLocalDir, int runs) {
    if (m_running) return;
    m_running = true;

    auto *watcher = new QFutureWatcher<BoardPerf>(this);
    connect(watcher, &QFutureWatcher<BoardPerf>::finished, this, [this, watcher] {
        watcher->deleteLater();
        m_running = false;
        emit finished(watcher->result());
    });

    auto fn = [serial, rknnLocal, imageLocal, remoteRoot, engineLocalDir, runs]() -> BoardPerf {
        BoardPerf r;
        r.device = serial;
        r.model = QFileInfo(rknnLocal).fileName();
        r.runs = runs;
        AdbClient adb({}, serial);

        // 首次：推引擎+运行时（存在性检查避免每次 45MB）
        if (adb.shell(QStringLiteral("test -f %1/rvs_engine && echo ok").arg(remoteRoot))
                .trimmed() != QStringLiteral("ok")) {
            if (!adb.push(engineLocalDir + QStringLiteral("/rvs_engine"),
                          remoteRoot + QStringLiteral("/rvs_engine")))
                { r.error = QStringLiteral("推引擎失败"); return r; }
            if (!adb.push(engineLocalDir + QStringLiteral("/librknnrt.so"),
                          remoteRoot + QStringLiteral("/librknnrt.so")))
                { r.error = QStringLiteral("推运行时失败"); return r; }
            adb.shell(QStringLiteral("chmod 755 %1/rvs_engine").arg(remoteRoot));
        }

        // 双工程测量：full（含 infer）与 base（无 infer）→ 差值 = NPU 纯推理增量
        const QString projF = remoteRoot + QStringLiteral("/perf_test");
        const QString projB = remoteRoot + QStringLiteral("/perf_base");
        // 本地临时 pipeline 两份（full/base），base 无 models 段、无 inf 段
        const QString kFullJson = QStringLiteral(
            "{\"version\":1,\"name\":\"perf_test\",\"target\":{\"platform\":\"rk3576\",\"runtime\":\"android-native\",\"core_mask\":\"auto\"},"
            "\"models\":{\"det\":{\"family\":\"yolov5-det\",\"file\":\"models/model.rknn\",\"input\":{\"name\":\"\",\"shape\":[1,3,640,640],\"layout\":\"NHWC\",\"dtype\":\"uint8\"}}},"
            "\"pipeline\":[{\"id\":\"src\",\"plugin\":\"image_dir\",\"params\":{\"path\":\"assets/test_images\",\"loop\":false}},"
            "{\"id\":\"pre\",\"plugin\":\"letterbox\",\"params\":{\"size\":640,\"pad_value\":255}},"
            "{\"id\":\"inf\",\"plugin\":\"infer\",\"model\":\"det\"},"
            "{\"id\":\"out\",\"plugin\":\"image_out\",\"params\":{\"path\":\"output\",\"draw\":false,\"save_json\":false}}]}");
        const QString kBaseJson = QStringLiteral(
            "{\"version\":1,\"name\":\"perf_base\",\"target\":{\"platform\":\"rk3576\",\"runtime\":\"android-native\",\"core_mask\":\"auto\"},"
            "\"pipeline\":[{\"id\":\"src\",\"plugin\":\"image_dir\",\"params\":{\"path\":\"assets/test_images\",\"loop\":false}},"
            "{\"id\":\"pre\",\"plugin\":\"letterbox\",\"params\":{\"size\":640,\"pad_value\":255}},"
            "{\"id\":\"out\",\"plugin\":\"image_out\",\"params\":{\"path\":\"output\",\"draw\":false,\"save_json\":false}}]}");
        const QString tmpDir = QStringLiteral("/tmp/rvs_boardperf");
        QDir().mkpath(tmpDir);
        QFile f1(tmpDir + QStringLiteral("/pipeline.json"));  f1.open(QIODevice::WriteOnly); f1.write(kFullJson.toUtf8());
        QFile f2(tmpDir + QStringLiteral("/pipeline_base.json")); f2.open(QIODevice::WriteOnly); f2.write(kBaseJson.toUtf8());

        adb.shell(QStringLiteral("mkdir -p %1/models %1/assets/test_images %2/assets/test_images").arg(projF, projB));
        if (!adb.push(tmpDir + QStringLiteral("/pipeline.json"), projF + QStringLiteral("/pipeline.json")))
            { r.error = QStringLiteral("推 pipeline.json 失败"); return r; }
        if (!adb.push(tmpDir + QStringLiteral("/pipeline_base.json"), projB + QStringLiteral("/pipeline.json")))
            { r.error = QStringLiteral("推 pipeline_base.json 失败"); return r; }
        if (!adb.push(rknnLocal, projF + QStringLiteral("/models/model.rknn")))
            { r.error = QStringLiteral("推 .rknn 失败"); return r; }
        for (int i = 0; i < 10; ++i) {
            const QString name = QStringLiteral("perf_%1.jpg").arg(i, 2, 10, QLatin1Char('0'));
            if (!adb.push(imageLocal, projF + QStringLiteral("/assets/test_images/") + name))
                { r.error = QStringLiteral("推测试图失败(full)"); return r; }
            if (!adb.push(imageLocal, projB + QStringLiteral("/assets/test_images/") + name))
                { r.error = QStringLiteral("推测试图失败(base)"); return r; }
        }

        // 跑 N 轮，解析 e2e_avg_ms
        const QString cmdF = QStringLiteral(
            "cd %1 && LD_LIBRARY_PATH=%2 %2/rvs_engine .").arg(projF, remoteRoot);
        const QString cmdB = QStringLiteral(
            "cd %1 && LD_LIBRARY_PATH=%2 %2/rvs_engine .").arg(projB, remoteRoot);
        QList<double> fullMs, baseMs;
        for (int i = 0; i < runs; ++i) {
            const QString out = adb.shell(cmdF);
            int frames = 0;
            const double ms = parseE2E(out, &frames);
            if (ms < 0) {
                r.error = QStringLiteral("full 第 %1 轮引擎输出无法解析").arg(i + 1);
                r.rawOut = out.right(400);
                return r;
            }
            r.framesPerRun = qMax(r.framesPerRun, frames);
            fullMs.append(ms);
        }
        for (int i = 0; i < runs; ++i) {
            const QString out = adb.shell(cmdB);
            int frames = 0;
            const double ms = parseE2E(out, &frames);
            if (ms < 0) {
                r.error = QStringLiteral("base 第 %1 轮引擎输出无法解析").arg(i + 1);
                r.rawOut = out.right(400);
                return r;
            }
            baseMs.append(ms);
        }
        auto median = [](QList<double> v) {
            std::sort(v.begin(), v.end());
            return v.isEmpty() ? 0.0 : v[v.size() / 2];
        };
        r.e2eMs = fullMs;
        r.avgMs = median(fullMs);
        r.baseAvgMs = median(baseMs);
        r.inferMs = qMax(0.0, r.avgMs - r.baseAvgMs);
        if (!r.e2eMs.isEmpty()) {
            r.ok = true;
            double sum = 0, mn = 1e18;
            for (double v : r.e2eMs) { sum += v; mn = qMin(mn, v); }
            r.avgMs = sum / r.e2eMs.size();
            r.minMs = mn;
        }
        return r;
    };

    auto future = QtConcurrent::run(fn);
    watcher->setFuture(future);
}
