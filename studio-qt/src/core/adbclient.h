#pragma once
// adb.exe 通道：同步封装（本地命令毫秒级，阻塞可接受）
#include <QString>
#include <QStringList>
#include <QProcess>

class AdbClient {
public:
    explicit AdbClient(const QString &adbPath = {}, const QString &serial = {});

    QStringList devices() const;                 // 在线序列号列表
    QString shell(const QString &cmd) const;     // 返回 stdout
    bool push(const QString &local, const QString &remote, QString *err = nullptr) const;
    QString forward(int localPort, const QString &remote) const;
    void removeForward(int localPort) const;
    bool hasDevice() const { return !m_serial.isEmpty(); }

    const QString &serial() const { return m_serial; }
    const QString &bin() const { return m_bin; }

private:
    QString run(const QStringList &args, int timeoutMs = 15000) const;
    QString m_bin;
    QString m_serial;
};
