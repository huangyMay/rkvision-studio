#include "boardmonitor.h"

#include <QRegularExpression>

namespace {
// 单次 adb shell 采集全部指标（一行 CPU + 两行内存 + NPU 负载 + 温度）
const char *kSampleCmd =
    "head -1 /proc/stat; "
    "grep -E 'MemTotal|MemAvailable' /proc/meminfo; "
    "cat /sys/kernel/debug/rknpu/load 2>/dev/null; "
    "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null";

const QRegularExpression kCpuRe(QStringLiteral("^cpu\\s+((?:\\d+\\s*)+)$"));
const QRegularExpression kMemRe(QStringLiteral("^(MemTotal|MemAvailable):\\s+(\\d+) kB"));
const QRegularExpression kNpuDualRe(QStringLiteral("Core0:\\s*(\\d+)%.*,\\s*Core1:\\s*(\\d+)%"));
const QRegularExpression kNpuSingleRe(QStringLiteral("NPU load:\\s*(\\d+)%"));
} // namespace

BoardMonitor::BoardMonitor(QObject *parent) : QObject(parent) {}

void BoardMonitor::poll(const QString &serial) {
    if (serial.isEmpty() || m_proc) return;
    if (serial != m_serial) {                    // 换板：CPU 差分基线作废
        m_serial = serial;
        m_hasPrev = false;
    }

    m_proc = new QProcess(this);
    connect(m_proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &BoardMonitor::onFinished);
    m_proc->start(QStringLiteral("adb"),
                  {QStringLiteral("-s"), serial, QStringLiteral("shell"),
                   QLatin1String(kSampleCmd)});
}

void BoardMonitor::onFinished() {
    const QString out = QString::fromUtf8(m_proc->readAllStandardOutput());
    m_proc->deleteLater();
    m_proc = nullptr;

    Sample s;
    quint64 idle = 0, total = 0;
    for (const QString &ln : out.split('\n')) {
        const QString t = ln.trimmed();
        const auto mc = kCpuRe.match(t);
        if (mc.hasMatch()) {
            const QStringList f = mc.captured(1).split(QRegularExpression("\\s+"),
                                                       Qt::SkipEmptyParts);
            quint64 tot = 0, idl = 0;
            for (int i = 0; i < f.size(); ++i) {
                const quint64 v = f[i].toULongLong();
                tot += v;
                if (i == 3 || i == 4) idl += v;      // idle + iowait
            }
            idle = idl; total = tot;
            continue;
        }
        const auto mm = kMemRe.match(t);
        if (mm.hasMatch()) {
            if (mm.captured(1) == QStringLiteral("MemTotal"))
                s.memTotalMb = mm.captured(2).toDouble() / 1024.0;
            else
                s.memUsedMb = mm.captured(2).toDouble() / 1024.0;
            continue;
        }
        const auto mn = kNpuDualRe.match(t);
        if (mn.hasMatch()) {
            s.npuValid = true;
            s.npuCore0 = mn.captured(1).toDouble();
            s.npuCore1 = mn.captured(2).toDouble();
            continue;
        }
        const auto ms1 = kNpuSingleRe.match(t);
        if (ms1.hasMatch()) {                        // 单核 NPU（3566 等）
            s.npuValid = true;
            s.npuCore0 = ms1.captured(1).toDouble();
            s.npuCore1 = -1;
            continue;
        }
        bool okNum = false;
        const double mv = t.toDouble(&okNum);
        if (okNum && mv > 10000) s.tempC = mv / 1000.0;   // 毫摄氏度
    }

    if (total > m_prevTotal && m_hasPrev) {
        s.cpuPct = 100.0 * (1.0 - double(idle - m_prevIdle) / double(total - m_prevTotal));
    }
    m_prevIdle = idle; m_prevTotal = total; m_hasPrev = true;

    s.memUsedMb = qMax(0.0, s.memTotalMb - s.memUsedMb);      // Available → used
    s.ok = s.memTotalMb > 0;

    // NPU 负载在 debugfs 下需要 root——每板尝试一次 adb root（adbd 重启会清 reverse，
    // 重挂 reverse 由使用方在下次部署/预览时自然恢复）
    if (s.ok && !s.npuValid && !m_rootTried.contains(m_serial)) {
        m_rootTried.insert(m_serial);
        QProcess::startDetached(QStringLiteral("adb"),
                                {QStringLiteral("-s"), m_serial, QStringLiteral("root")});
    }
    emit stats(s);
}
