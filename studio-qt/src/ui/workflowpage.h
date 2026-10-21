// WORKFLOW / 工作流 页：分段表单卡片 → WorkflowGen 生成 pipeline.json（设计稿 §5）。
// 本页只做表单采集与校验提示；保存复用 ProjectRegistry 目录扫描，部署复用 onDeploy() 代码路径。
#pragma once
#include <QWidget>
#include "model/workflowgen.h"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QSpinBox;
class QTableWidget;

class WorkflowPage : public QWidget {
    Q_OBJECT
public:
    explicit WorkflowPage(const QString &projectsRoot, QWidget *parent = nullptr);

signals:
    void projectSaved(const QString &name);
    void deployRequested(const QString &name);
    void logLine(const QString &text, const QString &color);

private:
    void onSave(bool deploy);
    WorkflowForm collect() const;               // 规则行非法抛 QString

    QString projectsRoot_;
    QLineEdit *name_ = nullptr;
    QComboBox *srcType_ = nullptr;
    QLineEdit *srcPath_ = nullptr, *device_ = nullptr;
    QSpinBox *width_ = nullptr, *height_ = nullptr;
    QSpinBox *size_ = nullptr, *pad_ = nullptr;
    QLineEdit *model_ = nullptr;
    QComboBox *family_ = nullptr;
    QLineEdit *classes_ = nullptr;
    QDoubleSpinBox *conf_ = nullptr, *iou_ = nullptr;
    QTableWidget *rules_ = nullptr;
    QCheckBox *ngArchive_ = nullptr;
    QLineEdit *ngDir_ = nullptr;
    QCheckBox *imageOut_ = nullptr, *mjpegEn_ = nullptr;
    QSpinBox *port_ = nullptr;
};
