#include "inferstream.h"
#include "onnxrunner.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QVector>
#include <QtEndian>
#include <QtGlobal>

namespace {
// WSL 侧常驻流脚本（tools/onnx_stream.py；RVS_TOOLS_DIR 由 CMake 注入）
const QString kStreamScript =
    OnnxRunner::toWslPath(QStringLiteral(RVS_TOOLS_DIR "/onnx_stream.py"));
} // namespace

InferStream::InferStream(QObject *parent) : QObject(parent) {}

void InferStream::start(const QString &modelPath, double conf, double iou) {
    stop();
    m_dying = false;
    m_ready = false;
    m_busy = false;
    m_buf.clear();

    m_proc = new QProcess(this);
    connect(m_proc, &QProcess::errorOccurred, this, &InferStream::onError);
    connect(m_proc, &QProcess::readyReadStandardOutput, this, &InferStream::onReadyRead);
    const QStringList args = {
        QStringLiteral("-d"), QStringLiteral(RVS_WSL_DISTRO), QStringLiteral("--"),
        QStringLiteral("python3"), kStreamScript,
        OnnxRunner::toWslPath(modelPath), QString::number(conf), QString::number(iou),
    };
    m_proc->start(QStringLiteral("wsl.exe"), args);
}

void InferStream::stop() {
    if (!m_proc) return;
    m_dying = true;
    m_proc->kill();
    m_proc->deleteLater();
    m_proc = nullptr;
    m_ready = false;
    m_busy = false;
    m_buf.clear();
    emit readyChanged(false);
}

bool InferStream::submitFrame(const QByteArray &jpeg) {
    if (!m_ready || m_busy || !m_proc) return false;
    const quint32 n = quint32(jpeg.size());
    const QByteArray hdr(reinterpret_cast<const char *>(&n), 4);
    m_proc->write(hdr);
    m_proc->write(jpeg);
    m_busy = true;
    return true;
}

void InferStream::submitCommand(const QString &json) {
    if (!m_ready || !m_proc) return;
    const QByteArray body = QStringLiteral("CMD:").toUtf8() + json.toUtf8();
    const quint32 n = quint32(body.size());
    const QByteArray hdr(reinterpret_cast<const char *>(&n), 4);
    m_proc->write(hdr);
    m_proc->write(body);
}

void InferStream::onError(QProcess::ProcessError) {
    if (m_dying) return;
    m_ready = false;
    emit readyChanged(false);
}

void InferStream::onReadyRead() {
    if (!m_proc) return;
    m_buf += m_proc->readAllStandardOutput();

    // 纯二进制分帧：4B 小端长度 + JSON；hello 消息即就绪握手
    for (;;) {
        if (m_buf.size() < 4) return;
        const quint32 n = qFromUnaligned<quint32>(
            reinterpret_cast<const uchar *>(m_buf.constData()));
        if (n > 64 * 1024 * 1024) {                      // 长度异常，防脏数据卡死
            m_buf.clear();
            return;
        }
        if (m_buf.size() < int(4 + n)) return;
        const QByteArray payload = m_buf.mid(4, int(n));
        m_buf.remove(0, 4 + int(n));
        m_busy = false;

        dispatch(QJsonDocument::fromJson(payload).object());
    }
}

void InferStream::dispatch(const QJsonObject &o) {
    const QString type = o.value(QStringLiteral("type")).toString();

    if (type == QStringLiteral("hello")) {              // 就绪握手
        m_ready = true;
        emit readyChanged(true);
        return;
    }

    if (type == QStringLiteral("model_info")) {
        QString ops;
        int shown = 0;
        for (const auto &v : o.value(QStringLiteral("op_hist")).toArray()) {
            const QJsonObject d = v.toObject();
            if (++shown > 6) { ops += QStringLiteral("…"); break; }
            ops += QStringLiteral("%1×%2  ").arg(d.value(QStringLiteral("op")).toString())
                       .arg(d.value(QStringLiteral("count")).toInt());
        }
        emit workerText(QStringLiteral(
            "[模型信息] 文件 %1MB · 参数 %2\n[算子] %3")
            .arg(o.value(QStringLiteral("file_mb")).toDouble(), 0, 'f', 1)
            .arg(o.value(QStringLiteral("params")).toInteger())
            .arg(ops));
        return;
    }
    if (type == QStringLiteral("stats")) {
        emit workerText(QStringLiteral(
            "[统计/%1帧] 平均 %2ms · 最快 %3ms · 最慢 %4ms · P95 %5ms · 推理FPS %6")
            .arg(o.value(QStringLiteral("frames")).toInt())
            .arg(o.value(QStringLiteral("avg_ms")).toDouble(), 0, 'f', 1)
            .arg(o.value(QStringLiteral("min_ms")).toDouble(), 0, 'f', 1)
            .arg(o.value(QStringLiteral("max_ms")).toDouble(), 0, 'f', 1)
            .arg(o.value(QStringLiteral("p95_ms")).toDouble(), 0, 'f', 1)
            .arg(o.value(QStringLiteral("inf_fps")).toDouble(), 0, 'f', 1));
        return;
    }
    if (type == QStringLiteral("prof")) {
        const QJsonArray layers = o.value(QStringLiteral("layers")).toArray();
        QString text = QStringLiteral("[逐层TOP%1] · 平均 %2ms/帧")
                           .arg(layers.size())
                           .arg(o.value(QStringLiteral("avg_ms")).toDouble(), 0, 'f', 1);
        if (!o.value(QStringLiteral("error")).toString().isEmpty())
            text = QStringLiteral("[逐层剖析失败] %1").arg(o.value(QStringLiteral("error")).toString());
        int i = 0;
        for (const auto &v : layers) {
            const QJsonObject d = v.toObject();
            text += QStringLiteral("\n%1. %2  %3  %4ms")
                        .arg(++i, 2).arg(d.value(QStringLiteral("name")).toString())
                        .arg(d.value(QStringLiteral("op")).toString())
                        .arg(d.value(QStringLiteral("ms")).toDouble(), 0, 'f', 2);
        }
        emit workerText(text);
        return;
    }

    // 帧结果（叠加数据）+ 节流的信息行（每 25 帧）
    if (!o.value(QStringLiteral("ok")).toBool()) return;
    ++m_frameCnt;
    if (m_frameCnt % 25 == 0) {
        emit workerText(QStringLiteral(
            "[帧#%1] 预处理 %2ms · 推理 %3ms · 后处理 %4ms · 检出 %5 · top %6")
            .arg(o.value(QStringLiteral("frame")).toInt())
            .arg(o.value(QStringLiteral("pre_ms")).toDouble(), 0, 'f', 1)
            .arg(o.value(QStringLiteral("infer_ms")).toDouble(), 0, 'f', 1)
            .arg(o.value(QStringLiteral("post_ms")).toDouble(), 0, 'f', 1)
            .arg(o.value(QStringLiteral("n")).toInt())
            .arg(o.value(QStringLiteral("top")).toString()));
    }
    QVector<OnnxDetection> dets;
    for (const auto &v : o.value(QStringLiteral("detections")).toArray()) {
        const QJsonObject d = v.toObject();
        OnnxDetection det;
        det.x1 = d.value(QStringLiteral("x1")).toDouble();
        det.y1 = d.value(QStringLiteral("y1")).toDouble();
        det.x2 = d.value(QStringLiteral("x2")).toDouble();
        det.y2 = d.value(QStringLiteral("y2")).toDouble();
        det.conf = d.value(QStringLiteral("conf")).toDouble();
        det.cls = d.value(QStringLiteral("cls")).toInt();
        dets.append(det);
    }
    emit result(dets, o.value(QStringLiteral("w")).toInt(),
                o.value(QStringLiteral("h")).toInt(),
                o.value(QStringLiteral("infer_ms")).toDouble());
}
