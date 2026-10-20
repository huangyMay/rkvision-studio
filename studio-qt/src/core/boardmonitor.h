#pragma once
// 板端状态监视器：经 adb 采样 CPU 负载（/proc/stat 差分）、内存、NPU 负载（debugfs，
// 需 adb root）、SoC 温度。单飞轮询（未返回时跳过），NPU 缺数据时自动尝试一次 adb root。
#include <QObject>
#include <QProcess>
#include <QSet>
#include <QString>

class BoardMonitor : public QObject {
    Q_OBJECT
public:
    struct Sample {
        bool ok = false;
        double cpuPct = -1;                      // -1 = 首帧无差分
        double memUsedMb = 0, memTotalMb = 0;
        bool npuValid = false;
        double npuCore0 = -1, npuCore1 = -1;
        double tempC = -1;
    };

    explicit BoardMonitor(QObject *parent = nullptr);
    void poll(const QString &serial);            // 异步；忙/空串时跳过

signals:
    void stats(const BoardMonitor::Sample &s);

private slots:
    void onFinished();

private:
    QProcess *m_proc = nullptr;
    QString m_serial;
    quint64 m_prevIdle = 0, m_prevTotal = 0;
    bool m_hasPrev = false;
    QSet<QString> m_rootTried;
};
