#include "core/boardacc.h"

#include <QProcess>
#include <QTimer>

namespace {
constexpr int kTimeoutMs = 15 * 60 * 1000;   // 量化构建+双端逐层推理实测可达数分钟
}

BoardAccRunner::BoardAccRunner(QObject *parent) : QObject(parent) {}

void BoardAccRunner::start(const QString &distro, const QString &scriptWsl,
                           const QString &onnxWsl, const QString &datasetWsl,
                           const QString &device, const QString &meanstd,
                           const QString &outDirWsl) {
    stop();
    m_out.clear();
    m_proc = new QProcess(this);
    m_proc->setProgram(QStringLiteral("wsl.exe"));
    m_proc->setArguments({QStringLiteral("-d"), distro, QStringLiteral("--"),
                          QStringLiteral("python3"), QStringLiteral("-u"),
                          scriptWsl, onnxWsl, datasetWsl, device, meanstd, outDirWsl});
    connect(m_proc, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        emit logLine(QStringLiteral("量化精度: 进程错误（WSL/python3 可用？）"), "#ff6b6b");
    });
    connect(m_proc, &QProcess::readyReadStandardOutput, this, [this] {
        m_out += QString::fromUtf8(m_proc->readAllStandardOutput());
        const QStringList ls = m_out.split(QLatin1Char('\n'));
        for (const QString &l : ls)
            if (l.trimmed().startsWith(QStringLiteral("{\"ok\"")))
                emit logLine(l.trimmed().left(200), "#8ab4f8");
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
            err = QStringLiteral("rknn_acc.py 无 JSON 输出（WSL python3/toolkit2 就绪？）");
        emit finished(ok, json, err);
        m_proc->deleteLater();
        m_proc = nullptr;
    });

    // 超时看门狗：量化构建+逐层推理实测 ~7 分钟，15 分钟兜底
    QTimer::singleShot(kTimeoutMs, m_proc, [this] {
        if (m_proc) {
            m_proc->kill();
            emit finished(false, QString(),
                          QStringLiteral("accuracy_analysis 超时（15 分钟）——查网络 adb 保活与 rknn_server"));
        }
    });

    m_proc->start(QIODevice::ReadOnly);
}

void BoardAccRunner::stop() {
    if (m_proc) {
        m_proc->disconnect(this);
        m_proc->kill();
        m_proc->deleteLater();
        m_proc = nullptr;
    }
}
