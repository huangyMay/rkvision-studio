// 精度对比 runner：调 WSL 侧 tools/acc_compare.py（四级漏斗对比，板在线时含④）。
// 契约：stdout 末行 JSON {"ok":..,"stages":{..},"tensor_cosine":{..},"match":{..}}
#pragma once
#include <QObject>
#include <QString>

class QProcess;

class AccCompareRunner : public QObject {
    Q_OBJECT
public:
    explicit AccCompareRunner(QObject *parent = nullptr);
    bool running() const { return m_proc != nullptr; }

    // 所有路径须为 WSL 形态；device 为空 = 跳过④（纯 PC 对比）
    void start(const QString &distro, const QString &scriptWsl,
               const QString &onnxWsl, const QString &imageWsl,
               const QString &device, const QString &meanstd,
               const QString &datasetWsl);
    void stop();

signals:
    void finished(bool ok, const QString &summaryJson, const QString &error);
    void logLine(const QString &text, const QString &color);

private:
    QProcess *m_proc = nullptr;
    QString m_out;
};
