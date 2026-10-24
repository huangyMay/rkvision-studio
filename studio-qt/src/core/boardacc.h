// 板端量化精度 runner：调 WSL 侧 tools/rknn_acc.py（accuracy_analysis，需真板+网络/USB adb）。
// 契约：stdout 末行 JSON {"ok":..,"out_dir":..,"layers":[{name,sim_entire_cos,...}],...}
#pragma once
#include <QObject>
#include <QString>

class QProcess;

class BoardAccRunner : public QObject {
    Q_OBJECT
public:
    explicit BoardAccRunner(QObject *parent = nullptr);
    bool running() const { return m_proc != nullptr; }

    // 所有路径参数须为 WSL 形态（/home/.. 或 /mnt/c/..）；scriptWsl = tools/rknn_acc.py 绝对路径
    void start(const QString &distro, const QString &scriptWsl, const QString &onnxWsl,
               const QString &datasetWsl, const QString &device,
               const QString &meanstd, const QString &outDirWsl);
    void stop();

signals:
    void finished(bool ok, const QString &summaryJson, const QString &error);
    void logLine(const QString &text, const QString &color);

private:
    QProcess *m_proc = nullptr;
    QString m_out;
};
