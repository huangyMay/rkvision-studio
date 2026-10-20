#include "onnxanalyzer.h"
#include "onnxrunner.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QTimer>

namespace {
// WSL 侧分析脚本（tools/onnx_analyze.py；RVS_TOOLS_DIR 由 CMake 注入）
const QString kAnalyzeScript =
    OnnxRunner::toWslPath(QStringLiteral(RVS_TOOLS_DIR "/onnx_analyze.py"));
constexpr int kTimeoutMs = 120000;      // 首次 shape 推断+profiling 较慢
} // namespace

OnnxAnalyzer::OnnxAnalyzer(QObject *parent) : QObject(parent) {
    m_watchdog = new QTimer(this);
    m_watchdog->setSingleShot(true);
    connect(m_watchdog, &QTimer::timeout, this, [this] {
        if (m_proc) {
            m_proc->kill();
            m_proc->waitForFinished(3000);
        }
    });
}

bool OnnxAnalyzer::running() const { return m_proc != nullptr; }

void OnnxAnalyzer::run(const QString &modelPath, int runs) {
    if (m_proc) return;

    m_proc = new QProcess(this);
    connect(m_proc, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        if (m_watchdog->isActive()) m_watchdog->stop();
        OnnxAnalysis r;
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
        OnnxAnalysis r;

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
                QStringLiteral("分析脚本无有效输出"));
        } else {
            r.ok = true;
            r.model = o.value(QStringLiteral("model")).toString();
            r.fileMb = o.value(QStringLiteral("file_mb")).toDouble();
            r.params = o.value(QStringLiteral("params")).toInteger(-1);
            r.macs = o.value(QStringLiteral("macs")).toDouble(-1);
            r.runs = o.value(QStringLiteral("runs")).toInt();
            r.avgTotalMs = o.value(QStringLiteral("avg_total_ms")).toDouble();
            for (const auto &v : o.value(QStringLiteral("inputs")).toArray()) {
                const QJsonObject d = v.toObject();
                OnnxIo io;
                io.name = d.value(QStringLiteral("name")).toString();
                io.dtype = d.value(QStringLiteral("dtype")).toString();
                for (const auto &s : d.value(QStringLiteral("shape")).toArray())
                    io.shape << s.toVariant().toString();
                r.inputs.append(io);
            }
            for (const auto &v : o.value(QStringLiteral("outputs")).toArray()) {
                const QJsonObject d = v.toObject();
                OnnxIo io;
                io.name = d.value(QStringLiteral("name")).toString();
                io.dtype = d.value(QStringLiteral("dtype")).toString();
                for (const auto &s : d.value(QStringLiteral("shape")).toArray())
                    io.shape << s.toVariant().toString();
                r.outputs.append(io);
            }
            for (const auto &v : o.value(QStringLiteral("op_hist")).toArray()) {
                const QJsonObject d = v.toObject();
                r.opHist.append({d.value(QStringLiteral("op")).toString(),
                                 d.value(QStringLiteral("count")).toInt()});
            }
            for (const auto &v : o.value(QStringLiteral("top_layers")).toArray()) {
                const QJsonObject d = v.toObject();
                r.topLayers.append({d.value(QStringLiteral("name")).toString(),
                                    d.value(QStringLiteral("op")).toString(),
                                    d.value(QStringLiteral("ms")).toDouble()});
            }
        }
        emit finished(r);
        m_proc->deleteLater();
        m_proc = nullptr;
    });

    const QStringList args = {
        QStringLiteral("-d"), QStringLiteral(RVS_WSL_DISTRO), QStringLiteral("--"),
        QStringLiteral("python3"), kAnalyzeScript,
        OnnxRunner::toWslPath(modelPath), QString::number(runs),
    };
    m_proc->start(QStringLiteral("wsl.exe"), args);
    m_watchdog->start(kTimeoutMs);
}
