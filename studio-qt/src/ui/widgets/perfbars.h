#pragma once

#include <QString>
#include <QVector>
#include <QWidget>

// 性能剖析瀑布图：每个 core_mask 一行，行内三段色条 = set/run/get 分段耗时。
// run（纯 NPU）是主角用饱和绿；set/get 是税用弱色。宽度按各行 e2e 最大值归一。
struct PerfRow {
    int mask = 0;
    double setMs = 0, runMs = 0, getMs = 0;
    bool ok = false;
    QString note;                 // 失败原因 / 备注（三核切分、不支持等）
};

class PerfBars : public QWidget {
    Q_OBJECT
public:
    explicit PerfBars(QWidget *parent = nullptr);
    void setRows(const QVector<PerfRow> &rows);
    QSize minimumSizeHint() const override;

protected:
    void paintEvent(QPaintEvent *) override;

private:
    QVector<PerfRow> m_rows;
};
