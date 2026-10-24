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

    // 路径参数须为 WSL 形态；device 为空 = 跳过④（纯 PC 对比）；dumpDir = 标注图落盘目录
    void start(const QString &distro, const QString &scriptWsl,
               const QString &onnxWsl, const QString &imageWsl,
               const QString &device, const QString &meanstd,
               const QString &datasetWsl, const QString &dumpDir);
    void stop();

signals:
    void finished(bool ok, const QString &summaryJson, const QString &error);
    void logLine(const QString &text, const QString &color);
    // 流式：每完成一级发一条（json 含 stage/stages/match/tensor_cosine[/image]）
    void stageUpdate(const QString &json);
    // 流式：逐层余弦就绪（accuracy_analysis 完成）
    void layersUpdate(const QString &layersJson);

private:
    QProcess *m_proc = nullptr;
    QString m_out;
};
