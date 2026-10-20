#pragma once
// ONNX 模型分析服务层：QProcess 调 WSL python3 (tools/onnx_analyze.py)，
// 回传结构统计/参数量/MACs/逐层耗时。与 OnnxRunner 同模式。
#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>

struct OnnxIo {
    QString name;
    QStringList shape;
    QString dtype;
};
struct OnnxLayerTime {
    QString name;
    QString op;
    double ms = 0;
};
struct OnnxOpCount {
    QString op;
    int count = 0;
};
struct OnnxAnalysis {
    bool ok = false;
    QString error;
    QString model;
    double fileMb = 0;
    QVector<OnnxIo> inputs, outputs;
    qint64 params = -1;                 // -1 = 未知
    double macs = -1;                   // -1 = shape 推断失败未估算
    QVector<OnnxOpCount> opHist;
    int runs = 0;
    double avgTotalMs = 0;
    QVector<OnnxLayerTime> topLayers;
};

Q_DECLARE_METATYPE(OnnxAnalysis)

class OnnxAnalyzer : public QObject {
    Q_OBJECT
public:
    explicit OnnxAnalyzer(QObject *parent = nullptr);
    bool running() const;
    void run(const QString &modelPath, int runs = 20);

signals:
    void finished(const OnnxAnalysis &result);

private:
    class QProcess *m_proc = nullptr;
    class QTimer *m_watchdog = nullptr;
};
