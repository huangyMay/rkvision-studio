// ONNX 逐层体检 runner：调 WSL 侧 tools/onnx_probe.py（图手术暴露全部中间输出，纯 PC）。
// 契约：stdout 末行 JSON {"ok":..,"layers":[..],"sensitive":[..],"table":"..."}
#pragma once
#include <QObject>
#include <QString>

class QProcess;

class OnnxProbeRunner : public QObject {
    Q_OBJECT
public:
    explicit OnnxProbeRunner(QObject *parent = nullptr);
    bool running() const { return m_proc != nullptr; }

    // 路径参数须为 WSL 形态；scriptWsl = tools/onnx_probe.py 绝对路径
    void start(const QString &distro, const QString &scriptWsl,
               const QString &onnxWsl, const QString &imageWsl,
               const QString &outDirWsl);
    void stop();

signals:
    void finished(bool ok, const QString &summaryJson, const QString &error);
    void logLine(const QString &text, const QString &color);

private:
    QProcess *m_proc = nullptr;
    QString m_out;
};
