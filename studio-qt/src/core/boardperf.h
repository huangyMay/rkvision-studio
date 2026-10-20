#pragma once
// 板端 NPU 性能测试：推 .rknn+测试图到板上 perf_test 工程，跑引擎 N 轮，
// 解析 stdout 的 {"frames":N,"e2e_avg_ms":X}，返回真实 NPU 端到端耗时。
#include <QObject>
#include <QString>
#include <QList>

struct BoardPerf {
    bool ok = false;
    QString error;
    QString rawOut;                 // 解析失败时的引擎输出尾部（排障用）
    QString device;
    QString model;
    int runs = 0;                   // 引擎执行轮数
    int framesPerRun = 0;
    QList<double> e2eMs;            // 每轮 full 端到端平均耗时（含推理）
    double avgMs = 0;
    double minMs = 0;
    double baseAvgMs = 0;           // base（无推理）端到端平均
    double inferMs = 0;             // 推理增量 = full - base（真实 NPU 每帧）
};

Q_DECLARE_METATYPE(BoardPerf)

class BoardPerfRunner : public QObject {
    Q_OBJECT
public:
    explicit BoardPerfRunner(QObject *parent = nullptr);
    bool running() const;
    // remoteRoot = 板上部署根（/data/local/tmp/rvs）；engineLocalDir = PC 侧引擎二进制目录
    void run(const QString &serial, const QString &rknnLocal,
             const QString &imageLocal, const QString &remoteRoot,
             const QString &engineLocalDir, int runs = 5);

signals:
    void finished(const BoardPerf &result);

private:
    bool m_running = false;
};
