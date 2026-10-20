#pragma once
// PC 实时推理流：常驻 WSL python worker（tools/onnx_stream.py），
// stdin 收 JPEG 帧 → onnxruntime 推理 → stdout 回检测 JSON（u32LE 帧协议）。
// 实时语义：worker 上一帧未回时不排队，新帧直接丢弃（只推最新）。
#include <QObject>
#include <QProcess>
#include <QByteArray>
#include <QJsonObject>
#include <QVector>

#include "onnxrunner.h"   // OnnxDetection（信号参数需完整类型）

class InferStream : public QObject {
    Q_OBJECT
public:
    explicit InferStream(QObject *parent = nullptr);
    bool ready() const { return m_ready; }
    bool busy() const { return m_busy; }
    void start(const QString &modelPath, double conf, double iou);
    void stop();
    bool submitFrame(const QByteArray &jpeg);    // 返回 false=忙被丢弃
    void submitCommand(const QString &json);     // 旁路命令（如逐层剖析），不受 busy 限制

signals:
    void readyChanged(bool ready);
    void result(const QVector<OnnxDetection> &dets, int imgW, int imgH, double inferMs);
    void workerText(const QString &text);        // model_info/stats/prof 格式化文本

private slots:
    void onReadyRead();
    void onError(QProcess::ProcessError);

private:
    void dispatch(const QJsonObject &o);         // 按 type 分派消息
    QProcess *m_proc = nullptr;
    QByteArray m_buf;
    bool m_ready = false;
    bool m_busy = false;
    bool m_dying = false;
    int m_frameCnt = 0;
};
