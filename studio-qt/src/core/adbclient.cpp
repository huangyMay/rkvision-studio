#include "adbclient.h"
#include <QRegularExpression>

AdbClient::AdbClient(const QString &adbPath, const QString &serial)
    : m_bin(adbPath.isEmpty() ? QStringLiteral("adb") : adbPath)
    , m_serial(serial) {}

QString AdbClient::run(const QStringList &args, int timeoutMs) const {
    QStringList full = args;
    if (!m_serial.isEmpty()) full << QStringLiteral("-s") << m_serial;
    QProcess p;
    p.start(m_bin, full);
    if (!p.waitForStarted(3000)) return {};
    if (!p.waitForFinished(timeoutMs)) { p.kill(); p.waitForFinished(); return {}; }
    return QString::fromUtf8(p.readAllStandardOutput());
}

QStringList AdbClient::devices() const {
    QStringList out;
    const QString text = run({QStringLiteral("devices")});
    for (const QString &line : text.split(QLatin1Char('\n'))) {
        const QString t = line.trimmed();
        if (t.endsWith(QStringLiteral("device"))) {
            out << t.section(QLatin1Char('\t'), 0, 0);
        }
    }
    return out;
}

QString AdbClient::shell(const QString &cmd) const {
    return run({QStringLiteral("shell"), cmd}).trimmed();
}

bool AdbClient::push(const QString &local, const QString &remote, QString *err) const {
    QStringList full;
    if (!m_serial.isEmpty()) full << QStringLiteral("-s") << m_serial;
    full << QStringLiteral("push") << local << remote;
    QProcess p;
    p.start(m_bin, full);
    if (!p.waitForStarted(3000)) { if (err) *err = QStringLiteral("adb 无法启动"); return false; }
    if (!p.waitForFinished(600000)) { p.kill(); if (err) *err = QStringLiteral("push 超时"); return false; }
    if (p.exitCode() != 0) { if (err) *err = QString::fromUtf8(p.readAllStandardError()); return false; }
    return true;
}

QString AdbClient::forward(int localPort, const QString &remote) const {
    return run({QStringLiteral("forward"), QString::number(localPort), remote});
}

void AdbClient::removeForward(int localPort) const {
    run({QStringLiteral("forward"), QStringLiteral("--remove"), QString::number(localPort)});
}
