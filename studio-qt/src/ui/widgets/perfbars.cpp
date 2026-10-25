#include "ui/widgets/perfbars.h"

#include <QPainter>
#include <QtMath>

namespace {
constexpr int kRowH = 40;         // 每行高
constexpr int kLabelW = 96;       // 左侧 mask 标签宽
constexpr int kLegendH = 26;      // 头部图例高
constexpr int kBarH = 18;         // 色条高（行内垂直居中）

const QColor kColSet(240, 173, 78);    // 归一化税：橙
const QColor kColRun(92, 184, 92);     // 纯 NPU：绿（主角）
const QColor kColGet(91, 192, 222);    // 回传税：蓝
const QColor kColText(200, 200, 200);
const QColor kColFail(231, 76, 60);
} // namespace

PerfBars::PerfBars(QWidget *parent) : QWidget(parent)
{
    setMinimumHeight(kLegendH + kRowH);
}

QSize PerfBars::minimumSizeHint() const
{
    return QSize(420, kLegendH + qMax(1, m_rows.size()) * kRowH + 8);
}

void PerfBars::setRows(const QVector<PerfRow> &rows)
{
    m_rows = rows;
    setMinimumHeight(kLegendH + qMax(1, m_rows.size()) * kRowH + 8);
    updateGeometry();
    update();
}

static QString maskName(int mask)
{
    switch (mask) {
    case 0: return QStringLiteral("auto");
    case 1: return QStringLiteral("0x1 核0");
    case 2: return QStringLiteral("0x2 核1");
    case 3: return QStringLiteral("0x3 核0+1");
    case 4: return QStringLiteral("0x4 核2");
    case 7: return QStringLiteral("0x7 三核");
    default: return QStringLiteral("0x%1").arg(mask, 0, 16);
    }
}

void PerfBars::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const int w = width();

    // 头部图例
    p.setPen(kColText);
    QFont f = p.font();
    f.setPointSizeF(f.pointSizeF() * 0.92);
    p.setFont(f);
    int lx = kLabelW;
    const struct { QColor c; const char *t; } legend[] = {
        {kColSet, "set 归一化"}, {kColRun, "run 纯NPU"}, {kColGet, "get 回传"},
    };
    for (const auto &l : legend) {
        p.setBrush(l.c);
        p.setPen(Qt::NoPen);
        p.drawRoundedRect(QRectF(lx, 6, 12, 12), 3, 3);
        p.setPen(kColText);
        p.drawText(QPoint(lx + 17, 16), QString::fromUtf8(l.t));
        lx += 17 + p.fontMetrics().horizontalAdvance(QString::fromUtf8(l.t)) + 18;
    }

    // 每行三段条（e2e 最大值归一；失败行画红字原因）
    double maxE2E = 0;
    for (const auto &r : m_rows)
        if (r.ok) maxE2E = qMax(maxE2E, r.setMs + r.runMs + r.getMs);
    const double barW = w - kLabelW - 14;
    for (int i = 0; i < m_rows.size(); i++) {
        const PerfRow &r = m_rows[i];
        const int y = kLegendH + i * kRowH;
        p.setPen(kColText);
        p.drawText(QRect(4, y, kLabelW - 10, kRowH), Qt::AlignVCenter, maskName(r.mask));
        if (!r.ok) {
            p.setPen(kColFail);
            p.drawText(QRect(kLabelW, y, w - kLabelW - 8, kRowH), Qt::AlignVCenter,
                       r.note.isEmpty() ? QStringLiteral("失败") : r.note);
            continue;
        }
        double x = kLabelW;
        const struct { double v; QColor c; } segs[] = {
            {r.setMs, kColSet}, {r.runMs, kColRun}, {r.getMs, kColGet},
        };
        for (const auto &s : segs) {
            const double sw = maxE2E > 0 ? s.v / maxE2E * barW : 0;
            if (sw > 0.5) {
                p.setPen(Qt::NoPen);
                p.setBrush(s.c);
                p.drawRoundedRect(QRectF(x, y + (kRowH - kBarH) / 2.0, sw, kBarH), 3, 3);
            }
            // 数值：条够宽画在条上，否则画在条尾外
            p.setPen(kColText);
            const QString t = QString::number(s.v, 'f', 1);
            const int tw = p.fontMetrics().horizontalAdvance(t);
            const double tc = x + sw / 2;
            if (tw + 8 < sw)
                p.drawText(QRectF(x, y + (kRowH - kBarH) / 2.0, sw, kBarH),
                           Qt::AlignCenter, t);
            else
                p.drawText(QPoint(qMin(qRound(x + sw + 3), w - tw - 2),
                                  y + kRowH / 2 + p.fontMetrics().height() / 2 - 2), t);
            x += sw;
        }
        // 行尾 e2e 汇总（右对齐覆盖防溢出：条区已留 14px）
        const double e2e = r.setMs + r.runMs + r.getMs;
        const QString et = QStringLiteral("  e2e %1ms").arg(e2e, 0, 'f', 1);
        p.setPen(QColor(160, 160, 160));
        p.drawText(QPoint(w - p.fontMetrics().horizontalAdvance(et) - 4, y + 14), et);
    }
}
