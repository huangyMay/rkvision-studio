#include "rknnsimrunner.h"
#include "onnxrunner.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QTimer>

namespace {
// WSL 侧模拟脚本（tools/rknn_sim.py；RVS_TOOLS_DIR 由 CMake 注入）
const QString kSimScript =
    OnnxRunner::toWslPath(QStringLiteral(RVS_TOOLS_DIR "/rknn_sim.py"));
constexpr int kTimeoutMs = 180000;      // 含 toolkit2 build 编译，单次可达分钟级
} // namespace

RknnSimRunner::RknnSimRunner(QObject *parent) : QObject(parent) {
    m_watchdog = new QTimer(this);
    m_watchdog->setSingleShot(true);
    connect(m_watchdog, &QTimer::timeout, this, [this] {
        if (m_proc) {
            m_proc->kill();
            m_proc->waitForFinished(3000);
        }
    });
}

bool RknnSimRunner::running() const { return m_proc != nullptr; }

void RknnSimRunner::run(const QString &imagePath, const QString &onnxPath,
                        double conf, double iou, const QString &meanstd,
                        const QString &saveRknn) {
    if (m_proc) return;
    m_imagePath = imagePath;

    m_proc = new QProcess(this);
    connect(m_proc, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        if (m_watchdog->isActive()) m_watchdog->stop();
        OnnxResult r;
        r.ok = false;
        r.simulator = true;
        r.error = QStringLiteral("无法启动 wsl.exe: %1").arg(m_proc->errorString());
        emit finished(r);
        m_proc->deleteLater();
        m_proc = nullptr;
    });
    connect(m_proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this](int, QProcess::ExitStatus) {
        if (m_watchdog->isActive()) m_watchdog->stop();
        const QString out = QString::fromUtf8(m_proc->readAllStandardOutput());
        OnnxResult r;
        r.imagePath = m_imagePath;
        r.simulator = true;

        QString line;
        const QStringList lines = out.split('\n');
        for (auto it = lines.crbegin(); it != lines.crend(); ++it) {
            const QString t = it->trimmed();
            if (!t.isEmpty()) { line = t; break; }
        }
        const QJsonObject o = QJsonDocument::fromJson(line.toUtf8()).object();
        if (o.isEmpty() || !o.value(QStringLiteral("ok")).toBool()) {
            r.ok = false;
            r.error = o.value(QStringLiteral("error")).toString(
                QStringLiteral("模拟脚本无有效输出"));
        } else {
            r.ok = true;
            r.model = o.value(QStringLiteral("model")).toString();
            r.image = o.value(QStringLiteral("image")).toString();
            r.imageW = o.value(QStringLiteral("image_w")).toInt();
            r.imageH = o.value(QStringLiteral("image_h")).toInt();
            r.inputSize = o.value(QStringLiteral("input_size")).toInt();
            r.numClasses = o.value(QStringLiteral("num_classes")).toInt();
            r.meanstd = o.value(QStringLiteral("meanstd")).toString();
            r.rknnSaved = o.value(QStringLiteral("rknn_saved")).toBool();
            const QJsonObject t = o.value(QStringLiteral("timings")).toObject();
            r.imageReadMs = t.value(QStringLiteral("image_read_ms")).toDouble();
            r.loadMs = t.value(QStringLiteral("build_ms")).toDouble();   // build 即装载
            r.preMs = t.value(QStringLiteral("preprocess_ms")).toDouble();
            r.inferMs = t.value(QStringLiteral("infer_ms")).toDouble();
            r.postMs = t.value(QStringLiteral("postprocess_ms")).toDouble();
            r.totalMs = t.value(QStringLiteral("total_ms")).toDouble();
            for (const auto &v : o.value(QStringLiteral("detections")).toArray()) {
                const QJsonObject d = v.toObject();
                OnnxDetection det;
                det.x1 = d.value(QStringLiteral("x1")).toDouble();
                det.y1 = d.value(QStringLiteral("y1")).toDouble();
                det.x2 = d.value(QStringLiteral("x2")).toDouble();
                det.y2 = d.value(QStringLiteral("y2")).toDouble();
                det.conf = d.value(QStringLiteral("conf")).toDouble();
                det.cls = d.value(QStringLiteral("cls")).toInt();
                r.detections.append(det);
            }
        }
        emit finished(r);
        m_proc->deleteLater();
        m_proc = nullptr;
    });

    QStringList args = {
        QStringLiteral("-d"), QStringLiteral(RVS_WSL_DISTRO), QStringLiteral("--"),
        QStringLiteral("python3"), kSimScript,
        OnnxRunner::toWslPath(imagePath), OnnxRunner::toWslPath(onnxPath),
        QString::number(conf), QString::number(iou), meanstd,
    };
    if (!saveRknn.isEmpty())
        args << OnnxRunner::toWslPath(saveRknn);
    m_proc->start(QStringLiteral("wsl.exe"), args);
    m_watchdog->start(kTimeoutMs);
}
