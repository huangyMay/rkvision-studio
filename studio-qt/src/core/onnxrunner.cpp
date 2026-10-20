#include "onnxrunner.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QRegularExpression>
#include <QTimer>

namespace {
// WSL 侧推理脚本路径（tools/onnx_infer.py；RVS_TOOLS_DIR 由 CMake 注入，
// 经 toWslPath 转 POSIX，跨机器可移植）
const QString kInferScript =
    OnnxRunner::toWslPath(QStringLiteral(RVS_TOOLS_DIR "/onnx_infer.py"));
constexpr int kTimeoutMs = 60000;
} // namespace

OnnxRunner::OnnxRunner(QObject *parent) : QObject(parent) {
    m_watchdog = new QTimer(this);
    m_watchdog->setSingleShot(true);
    connect(m_watchdog, &QTimer::timeout, this, [this] {
        if (m_proc) {
            m_proc->kill();
            m_proc->waitForFinished(3000);
        }
    });
}

bool OnnxRunner::running() const { return m_proc != nullptr; }

QString OnnxRunner::toWslPath(const QString &winPath) {
    // UNC wsl 路径：\\wsl.localhost\<发行版>\home\... 或 \\wsl$\<发行版>\... -> /home/...
    static const QRegularExpression unc(
        QStringLiteral("^[\\\\/]{2}wsl(?:\\.localhost|\\$)[\\\\/]([^\\\\/]+)[\\\\/](.*)$"));
    const auto mu = unc.match(winPath);
    if (mu.hasMatch()) return QStringLiteral("/") + mu.captured(2).replace('\\', '/');
    // 盘符：C:\a\b.jpg 或 D:/x/y.onnx -> /mnt/<drive>/...
    static const QRegularExpression re(QStringLiteral("^([A-Za-z]):[/\\\\](.*)$"));
    const auto m = re.match(winPath);
    if (!m.hasMatch()) return winPath;                 // 已是 POSIX 路径则原样
    const QString rest = m.captured(2).replace('\\', '/');
    return QStringLiteral("/mnt/%1/%2").arg(m.captured(1).toLower(), rest);
}

QStringList OnnxRunner::loadClassNames(const QString &modelPath) const {
    // 侧车文件：同名 .names / .txt，每行一个类名
    const QFileInfo fi(modelPath);
    for (const QString &ext : {QStringLiteral("names"), QStringLiteral("txt")}) {
        QFile f(fi.dir().filePath(fi.completeBaseName() + "." + ext));
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) continue;
        QStringList names;
        while (!f.atEnd())
            names << QString::fromUtf8(f.readLine()).trimmed();
        if (!names.isEmpty()) return names;
    }
    return {};
}

void OnnxRunner::run(const QString &imagePath, const QString &modelPath,
                     double conf, double iou) {
    if (m_proc) return;
    m_imagePath = imagePath;
    m_conf = conf;
    m_iou = iou;

    m_proc = new QProcess(this);
    connect(m_proc, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        if (m_watchdog->isActive()) m_watchdog->stop();
        OnnxResult r;
        r.ok = false;
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
        r.confTh = m_conf;
        r.iouTh = m_iou;

        // 取 stdout 最后一个非空行（JSON）
        QString line;
        const QStringList lines = out.split('\n');
        for (auto it = lines.crbegin(); it != lines.crend(); ++it) {
            const QString t = it->trimmed();
            if (!t.isEmpty()) { line = t; break; }
        }
        const QJsonDocument doc = QJsonDocument::fromJson(line.toUtf8());
        const QJsonObject o = doc.object();
        if (o.isEmpty() || !o.value(QStringLiteral("ok")).toBool()) {
            r.ok = false;
            r.error = o.value(QStringLiteral("error")).toString(
                QStringLiteral("推理脚本无有效输出"));
        } else {
            r.ok = true;
            r.model = o.value(QStringLiteral("model")).toString();
            r.image = o.value(QStringLiteral("image")).toString();
            r.imageW = o.value(QStringLiteral("image_w")).toInt();
            r.imageH = o.value(QStringLiteral("image_h")).toInt();
            r.inputSize = o.value(QStringLiteral("input_size")).toInt();
            r.numClasses = o.value(QStringLiteral("num_classes")).toInt();
            const QJsonObject t = o.value(QStringLiteral("timings")).toObject();
            r.imageReadMs = t.value(QStringLiteral("image_read_ms")).toDouble();
            r.loadMs = t.value(QStringLiteral("load_ms")).toDouble();
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

    const QStringList args = {
        QStringLiteral("-d"), QStringLiteral(RVS_WSL_DISTRO), QStringLiteral("--"),
        QStringLiteral("python3"), kInferScript,
        toWslPath(imagePath), toWslPath(modelPath),
        QString::number(conf), QString::number(iou),
    };
    m_proc->start(QStringLiteral("wsl.exe"), args);
    m_watchdog->start(kTimeoutMs);
}

void OnnxRunner::cancel() {
    if (m_watchdog->isActive()) m_watchdog->stop();
    if (m_proc) {
        m_proc->kill();
        m_proc->deleteLater();
        m_proc = nullptr;
    }
}
