#pragma once
// RKNN 模拟推理服务层：QProcess 调 WSL python3 (tools/rknn_sim.py)。
// toolkit2 不支持 load_rknn 后模拟——后端是 load_onnx→config→build→模拟，
// 因此输入是 ONNX+转换配置（meanstd），副产品 .rknn 可直接上板。
#include <QObject>
#include <QString>

#include "onnxrunner.h"   // 复用 OnnxResult

class RknnSimRunner : public QObject {
    Q_OBJECT
public:
    explicit RknnSimRunner(QObject *parent = nullptr);
    bool running() const;
    void run(const QString &imagePath, const QString &onnxPath,
             double conf, double iou, const QString &meanstd,
             const QString &saveRknn = QString());

signals:
    void finished(const OnnxResult &result);

private:
    class QProcess *m_proc = nullptr;
    class QTimer *m_watchdog = nullptr;
    QString m_imagePath;
};
