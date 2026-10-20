#pragma once
// ONNX 本地推理服务层：QProcess 调 WSL python3 (tools/onnx_infer.py)，
// JSON 回传检测结果与耗时分解。与 AdbClient 同模式（外部进程，无 UI 依赖）。
#include <QObject>
#include <QString>
#include <QVector>

struct OnnxDetection {
    double x1 = 0, y1 = 0, x2 = 0, y2 = 0, conf = 0;
    int cls = 0;
};

struct OnnxResult {
    bool ok = false;
    QString error;
    QString model, image;              // 文件名
    QString imagePath;                 // 原始（Windows）路径，供 UI 加载画框
    int imageW = 0, imageH = 0;
    int inputSize = 0, numClasses = 0;
    double confTh = 0.25, iouTh = 0.45;    // 本次推理所用阈值（细节条回显）
    bool simulator = false;                // true = toolkit2 模拟器（耗时≠板端）
    QString meanstd;                       // 模拟模式：转换期 mean/std 配置
    bool rknnSaved = false;                // 模拟模式：build 产出的 .rknn 已存盘
    double imageReadMs = 0, loadMs = 0, preMs = 0, inferMs = 0, postMs = 0, totalMs = 0;
    QVector<OnnxDetection> detections;
    QStringList classNames;            // <模型>.names 侧车文件（可选）
};

Q_DECLARE_METATYPE(OnnxResult)

class OnnxRunner : public QObject {
    Q_OBJECT
public:
    explicit OnnxRunner(QObject *parent = nullptr);
    bool running() const;
    void run(const QString &imagePath, const QString &modelPath,
             double conf = 0.25, double iou = 0.45);
    void cancel();

    static QString toWslPath(const QString &winPath);   // 盘符/UNC → WSL POSIX 路径

signals:
    void finished(const OnnxResult &result);

private:
    QStringList loadClassNames(const QString &modelPath) const;

    class QProcess *m_proc = nullptr;
    class QTimer *m_watchdog = nullptr;
    QString m_imagePath;
    double m_conf = 0.25, m_iou = 0.45;
};
