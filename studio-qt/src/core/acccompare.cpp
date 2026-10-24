#include "core/acccompare.h"

#include <QProcess>
#include <QTimer>

namespace {
constexpr int kTimeoutMs = 12 * 60 * 1000;   // fp16+int8 双构建+四级推理实测 ~5 分钟
}

AccCompareRunner::AccCompareRunner(QObject *parent) : QObject(parent) {}

void AccCompareRunner::start(const QString &distro, const QString &scriptWsl,
                             const QString &onnxWsl, const QString &imageWsl,
                             const QString &device, const QString &meanstd,
                             const QString &datasetWsl, const QString &dumpDir) {
    stop();
    m_out.clear();
    m_finalJson.clear();
    m_proc = new QProcess(this);
    m_proc->setProgram(QStringLiteral("wsl.exe"));
    QStringList args{QStringLiteral("-d"), distro, QStringLiteral("--"),
                     QStringLiteral("python3"), QStringLiteral("-u"), scriptWsl,
                     onnxWsl, imageWsl};
    if (!device.isEmpty()) args << device;
    args << QStringLiteral("0.25") << QStringLiteral("0.45") << meanstd << datasetWsl
         << dumpDir;
    m_proc->setArguments(args);
    connect(m_proc, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        emit logLine(QStringLiteral("精度对比: 进程错误（WSL python3/toolkit2 可用？）"), "#ff6b6b");
    });
    connect(m_proc, &QProcess::readyReadStandardOutput, this, [this] {
        m_out += QString::fromUtf8(m_proc->readAllStandardOutput());
        // 流式解析：STAGE {...} / LAYERS [...] 行即时上抛，UI 渐进渲染
        QStringList lines = m_out.split(QLatin1Char('\n'));
        m_out = lines.takeLast();                 // 半行留在缓冲
        for (const QString &l : lines) {
            const QString t = l.trimmed();
            if (t.startsWith(QStringLiteral("STAGE ")))
                emit stageUpdate(t.mid(6));
            else if (t.startsWith(QStringLiteral("LAYERS ")))
                emit layersUpdate(t.mid(7));
            else if (t.startsWith(QStringLiteral("LOG ")))
                emit flowLog(t.mid(4));
            else if (t.startsWith(QStringLiteral("{\"ok\"")))
                m_finalJson = t;                  // 最终报告行（finished 用）
        }
    });
    connect(m_proc, &QProcess::finished, this,
            [this](int, QProcess::ExitStatus) {
        // 最终 JSON 在 readyRead 里已捕获（流式消费模式下 m_out 只剩半行）
        QString err;
        QString json = m_finalJson;
        // 半行兜底：final JSON 可能与后续输出粘连在 m_out 尾部
        if (json.isEmpty() && m_out.contains(QStringLiteral("{\"ok\""))) {
            const int i = m_out.indexOf(QStringLiteral("{\"ok\""));
            const QString t = m_out.mid(i).trimmed();
            if (t.startsWith(QStringLiteral("{\"ok\""))) json = t;
        }
        bool ok = !json.isEmpty() && json.contains(QStringLiteral("\"ok\": true"));
        if (!json.isEmpty() && !ok) {
            const int i = json.indexOf(QStringLiteral("\"error\": \""));
            if (i >= 0) err = json.mid(i + 10, json.indexOf('"', i + 10) - i - 10);
        }
        if (json.isEmpty())
            err = QStringLiteral("acc_compare.py 无 JSON 输出（WSL python3/toolkit2 就绪？）");
        emit finished(ok, json, err);
        m_proc->deleteLater();
        m_proc = nullptr;
    });

    QTimer::singleShot(kTimeoutMs, m_proc, [this] {
        if (m_proc) {
            m_proc->kill();
            emit finished(false, QString(),
                          QStringLiteral("精度对比超时（12 分钟）——查 WSL toolkit2 与网络 adb"));
        }
    });

    m_proc->start(QIODevice::ReadOnly);
}

void AccCompareRunner::stop() {
    if (m_proc) {
        m_proc->disconnect(this);
        m_proc->kill();
        m_proc->deleteLater();
        m_proc = nullptr;
    }
}
