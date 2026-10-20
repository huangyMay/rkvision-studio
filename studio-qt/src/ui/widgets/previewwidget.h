#pragma once
// 预览大屏：QImage 帧 + HUD 角标/扫描线/状态字 + 检测框叠加，科技感核心件
#include <QImage>
#include <QLabel>
#include <QPair>
#include <QRectF>
#include <QString>
#include <QVector>
#include <QWidget>

class PreviewWidget : public QWidget {
    Q_OBJECT
public:
    explicit PreviewWidget(QWidget *parent = nullptr);
    void setFrame(const QImage &img);
    void setPlaceholder(const QString &text);
    void setStatus(const QString &text);       // 右下角状态字（如 fps / 分辨率）
    // 检测框叠加：rect 为归一化坐标（0~1，相对当前帧），label 可空
    void setOverlay(const QVector<QPair<QRectF, QString>> &items);
    void clearOverlay();

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    QImage m_frame;
    QString m_placeholder = QStringLiteral("NO SIGNAL");
    QString m_status;
    QVector<QPair<QRectF, QString>> m_overlay;
};
