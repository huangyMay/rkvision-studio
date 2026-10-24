#include "core/onnxprobe.h"

#include <QProcess>
#include <QTimer>

namespace {
constexpr int kTimeoutMs = 3 * 60 * 1000;   // 纯 PC 单次推理+统计，3 分钟兜底
}

OnnxProbeRunner::OnnxProbeRunner(QObject *parent) : QObject(parent) {}

void OnnxProbeRunner::start(const QString &distro, const QString &scriptWsl,
                            const QString &onnxWsl, const QString &imageWsl,
                            const QString &outDirWsl) {
    stop();
    m_out.clear();
    m_proc = new QProcess(this);
    m_proc->setProgram(QStringLiteral("wsl.exe"));
    m_proc->setArguments({QStringLiteral("-d"), distro, QStringLiteral("--"),
                          QStringLiteral("python3"), QStringLiteral("-u"),
                          scriptWsl, onnxWsl, imageWsl, outDirWsl});
    connect(m_proc, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        emit logLine(QStringLiteral("逐层体检: 进程错误（WSL python3/onnxruntime 可用？）"), "#ff6b6b");
    });
    connect(m_proc, &QProcess::readyReadStandardOutput, this, [this] {
        m_out += QString::fromUtf8(m_proc->readAllStandardOutput());
    });
    connect(m_proc, &QProcess::finished, this,
            [this](int, QProcess::ExitStatus) {
        QString json, err;
        const QStringList lines = m_out.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
        for (const QString &l : lines) {
            const QString t = l.trimmed();
            if (t.startsWith(QStringLiteral("{\"ok\""))) json = t;
        }
        bool ok = !json.isEmpty() && json.contains(QStringLiteral("\"ok\": true"));
        if (!json.isEmpty() && !ok) {
            const int i = json.indexOf(QStringLiteral("\"error\": \""));
            if (i >= 0) err = json.mid(i + 10, json.indexOf('"', i + 10) - i - 10);
        }
        if (json.isEmpty())
            err = QStringLiteral("onnx_probe.py 无 JSON 输出（WSL python3/onnxruntime 就绪？）");
        emit finished(ok, json, err);
        m_proc->deleteLater();
        m_proc = nullptr;
    });

    QTimer::singleShot(kTimeoutMs, m_proc, [this] {
        if (m_proc) {
            m_proc->kill();
            emit finished(false, QString(),
                          QStringLiteral("逐层体检超时（3 分钟）"));
        }
    });

    m_proc->start(QIODevice::ReadOnly);
}

void OnnxProbeRunner::stop() {
    if (m_proc) {
        m_proc->disconnect(this);
        m_proc->kill();
        m_proc->deleteLater();
        m_proc = nullptr;
    }
}
