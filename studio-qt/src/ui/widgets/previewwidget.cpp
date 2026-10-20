#include "previewwidget.h"
#include <QPainter>
#include <QPainterPath>

PreviewWidget::PreviewWidget(QWidget *parent) : QWidget(parent) {
    setMinimumHeight(320);
    setStyleSheet("background:#0a0e14; border-radius:10px;");
}

void PreviewWidget::setFrame(const QImage &img) {
    m_frame = img;
    update();
}

void PreviewWidget::setPlaceholder(const QString &text) {
    m_placeholder = text;
    update();
}

void PreviewWidget::setStatus(const QString &text) {
    m_status = text;
    update();
}

void PreviewWidget::setOverlay(const QVector<QPair<QRectF, QString>> &items) {
    m_overlay = items;
    update();
}

void PreviewWidget::clearOverlay() {
    if (!m_overlay.isEmpty()) {
        m_overlay.clear();
        update();
    }
}

void PreviewWidget::resizeEvent(QResizeEvent *) { update(); }

void PreviewWidget::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.fillRect(rect(), QColor(0x0a, 0x0e, 0x14));

    const QRect r = rect().adjusted(12, 12, -12, -12);

    if (m_frame.isNull()) {
        p.setPen(QColor(0x3a, 0x46, 0x5a));
        p.setFont(QFont("Consolas", 15, QFont::Bold));
        p.drawText(r, Qt::AlignCenter, m_placeholder);
    } else {
        // 等比缩放居中
        QImage scaled = m_frame.scaled(r.size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        QRect dst((width() - scaled.width()) / 2, (height() - scaled.height()) / 2,
                  scaled.width(), scaled.height());
        p.drawImage(dst, scaled);

        // ---- 检测框叠加（归一化坐标 → 绘制区映射）----
        if (!m_overlay.isEmpty()) {
            p.setFont(QFont("Consolas", 9, QFont::Bold));
            for (const auto &ov : m_overlay) {
                const QRectF box(dst.x() + ov.first.x() * dst.width(),
                                 dst.y() + ov.first.y() * dst.height(),
                                 ov.first.width() * dst.width(),
                                 ov.first.height() * dst.height());
                p.setBrush(QColor(0x3e, 0xcf, 0x72, 30));
                p.setPen(QPen(QColor(0x3e, 0xcf, 0x72), 2));
                p.drawRect(box);
                if (!ov.second.isEmpty()) {
                    const QRect tb = QFontMetrics(p.font()).tightBoundingRect(ov.second);
                    p.setBrush(QColor(0x3e, 0xcf, 0x72));
                    p.setPen(Qt::NoPen);
                    p.drawRect(QRectF(box.x(), box.y() - tb.height() - 6,
                                      tb.width() + 8, tb.height() + 5));
                    p.setPen(QColor(10, 14, 20));
                    p.drawText(QPointF(box.x() + 4, box.y() - 5), ov.second);
                }
            }
            p.setBrush(Qt::NoBrush);
        }
    }

    // ---- HUD 四角 L 形角标（青色发光）----
    p.setRenderHint(QPainter::Antialiasing, true);
    QPen pen(QColor(0x00, 0xe5, 0xff, 0xd0), 2);
    pen.setCapStyle(Qt::FlatCap);
    p.setPen(pen);
    p.setBrush(Qt::NoBrush);
    const int L = 26, M = 6;
    const QRect h = rect().adjusted(M, M, -M, -M);
    p.drawLine(h.left(), h.top() + L, h.left(), h.top());
    p.drawLine(h.left(), h.top(), h.left() + L, h.top());
    p.drawLine(h.right() - L, h.top(), h.right(), h.top());
    p.drawLine(h.right(), h.top(), h.right(), h.top() + L);
    p.drawLine(h.left(), h.bottom() - L, h.left(), h.bottom());
    p.drawLine(h.left(), h.bottom(), h.left() + L, h.bottom());
    p.drawLine(h.right() - L, h.bottom(), h.right(), h.bottom());
    p.drawLine(h.right(), h.bottom() - L, h.right(), h.bottom());

    // ---- 半透明扫描线（氛围）----
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(0x00, 0xe5, 0xff, 10));
    const int scanY = (int)((time(nullptr) * 60) % height());
    p.drawRect(0, scanY, width(), 2);

    // ---- 左上 REC 指示 + 右下状态字 ----
    p.setPen(QColor(0xff, 0x5c, 0x5c));
    p.setFont(QFont("Consolas", 9, QFont::Bold));
    p.drawText(rect().adjusted(16, 10, 0, 0), Qt::AlignLeft | Qt::AlignTop,
               QStringLiteral("● LIVE"));
    if (!m_status.isEmpty()) {
        p.setPen(QColor(0x00, 0xe5, 0xff, 0xc0));
        p.drawText(rect().adjusted(0, 0, -16, -10), Qt::AlignRight | Qt::AlignBottom, m_status);
    }
}
