#include "ui/workflowpage.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

namespace {
void makeCardWrap(QVBoxLayout *target, QVBoxLayout **layout, const QString &title) {
    auto *card = new QFrame;
    card->setObjectName("Card");
    auto *v = new QVBoxLayout(card);
    v->setContentsMargins(14, 12, 14, 14);
    auto *t = new QLabel(title);
    t->setObjectName("CardTitle");
    v->addWidget(t);
    target->addWidget(card);
    *layout = v;
}
} // namespace

WorkflowPage::WorkflowPage(const QString &projectsRoot, QWidget *parent)
    : QWidget(parent), projectsRoot_(projectsRoot) {
    auto *root = new QVBoxLayout(this);
    root->setContentsMargins(14, 14, 14, 14);
    root->setSpacing(12);

    auto *cols = new QHBoxLayout;
    root->addLayout(cols, 1);
    auto *left = new QVBoxLayout;
    auto *right = new QVBoxLayout;
    left->setSpacing(12); right->setSpacing(12);
    cols->addLayout(left, 1);
    cols->addLayout(right, 1);

    // ---- 左列 ----
    QVBoxLayout *nameL, *srcL, *preL, *outL;
    makeCardWrap(left, &nameL, tr("PROJECT / 工程名"));
    name_ = new QLineEdit;
    name_->setPlaceholderText(tr("my_defect_scan"));
    nameL->addWidget(name_);

    makeCardWrap(left, &srcL, tr("SOURCE / 信号源"));
    srcType_ = new QComboBox;
    srcType_->addItem(tr("图片目录 (image_dir)"));
    srcType_->addItem(tr("USB 摄像头 (v4l2)"));
    srcPath_ = new QLineEdit(QStringLiteral("assets/test_images"));
    auto *srcBrowse = new QPushButton(tr("…"));
    srcBrowse->setMaximumWidth(36);
    device_ = new QLineEdit(QStringLiteral("/dev/video0"));
    width_ = new QSpinBox; width_->setRange(160, 3840); width_->setValue(1280);
    height_ = new QSpinBox; height_->setRange(120, 2160); height_->setValue(720);
    auto *srcRow1 = new QHBoxLayout;
    srcRow1->addWidget(srcType_, 1);
    srcL->addLayout(srcRow1);
    auto *srcRow2 = new QHBoxLayout;
    srcRow2->addWidget(srcPath_, 1); srcRow2->addWidget(srcBrowse);
    srcRow2->addWidget(device_); srcRow2->addWidget(width_);
    srcRow2->addWidget(new QLabel("x")); srcRow2->addWidget(height_);
    srcL->addLayout(srcRow2);
    device_->hide(); width_->hide(); height_->hide();

    makeCardWrap(left, &preL, tr("PREPROCESS / 预处理"));
    auto *preRow = new QHBoxLayout;
    preRow->addWidget(new QLabel(tr("尺寸")));
    size_ = new QSpinBox; size_->setRange(96, 1280); size_->setValue(640);
    preRow->addWidget(size_);
    preRow->addWidget(new QLabel(tr("pad 值")));
    pad_ = new QSpinBox; pad_->setRange(0, 255); pad_->setValue(255);
    preRow->addWidget(pad_); preRow->addStretch(1);
    preL->addLayout(preRow);

    makeCardWrap(left, &outL, tr("OUTPUT / 输出"));
    imageOut_ = new QCheckBox(tr("image_out（结果图 + JSON 存 output/）"));
    imageOut_->setChecked(true);
    mjpegEn_ = new QCheckBox(tr("MJPEG 预览流"));
    port_ = new QSpinBox; port_->setRange(1024, 65535); port_->setValue(8000);
    auto *outRow = new QHBoxLayout;
    outRow->addWidget(mjpegEn_); outRow->addWidget(port_); outRow->addStretch(1);
    outL->addWidget(imageOut_);
    outL->addLayout(outRow);

    // ---- 右列 ----
    QVBoxLayout *mdlL, *postL, *jdgL;
    makeCardWrap(right, &mdlL, tr("MODEL / 模型与家族"));
    model_ = new QLineEdit;
    model_->setPlaceholderText(tr("点 … 选择 .rknn"));
    auto *mdlBrowse = new QPushButton(tr("…"));
    mdlBrowse->setMaximumWidth(36);
    family_ = new QComboBox;
    family_->addItem(QStringLiteral("yolov5-det"));
    family_->addItem(QStringLiteral("yolov8-det"));
    classes_ = new QLineEdit(QStringLiteral("head,helmet"));
    classes_->setPlaceholderText(tr("类别逗号分隔，可留空"));
    auto *mdlRow1 = new QHBoxLayout;
    mdlRow1->addWidget(model_, 1); mdlRow1->addWidget(mdlBrowse);
    auto *mdlRow2 = new QHBoxLayout;
    mdlRow2->addWidget(family_); mdlRow2->addWidget(classes_, 1);
    mdlL->addLayout(mdlRow1); mdlL->addLayout(mdlRow2);

    makeCardWrap(right, &postL, tr("POST / 后处理"));
    auto *postRow = new QHBoxLayout;
    postRow->addWidget(new QLabel("conf"));
    conf_ = new QDoubleSpinBox; conf_->setRange(0.05, 0.95); conf_->setSingleStep(0.05);
    conf_->setValue(0.25);
    postRow->addWidget(conf_);
    postRow->addWidget(new QLabel("IoU"));
    iou_ = new QDoubleSpinBox; iou_->setRange(0.10, 0.90); iou_->setSingleStep(0.05);
    iou_->setValue(0.45);
    postRow->addWidget(iou_); postRow->addStretch(1);
    postL->addLayout(postRow);

    makeCardWrap(right, &jdgL, tr("JUDGE / 判决规则（可选，留空不判决）"));
    rules_ = new QTableWidget(0, 4);
    rules_->setHorizontalHeaderLabels({tr("类型"), tr("类别"), tr("参数1"), tr("参数2")});
    rules_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    rules_->setMinimumHeight(150);
    jdgL->addWidget(rules_);
    auto hint = new QLabel(tr(
        "count: min/max | score: thres | size: min_px/max_px | "
        "region: \"x,y;x,y;x,y\" + inside/outside | persist: frames | regex: pattern"));
    hint->setWordWrap(true);
    jdgL->addWidget(hint);
    auto *jdgRow = new QHBoxLayout;
    auto *addRule = new QPushButton(tr("＋ 规则"));
    auto *delRule = new QPushButton(tr("－ 删除"));
    ngArchive_ = new QCheckBox(tr("NG 存档"));
    ngArchive_->setChecked(true);
    ngDir_ = new QLineEdit(QStringLiteral("ng_images"));
    jdgRow->addWidget(addRule); jdgRow->addWidget(delRule);
    jdgRow->addWidget(ngArchive_); jdgRow->addWidget(ngDir_, 1);
    jdgL->addLayout(jdgRow);

    // ---- 底部按钮 ----
    auto *btns = new QHBoxLayout;
    auto *saveBtn = new QPushButton(tr("✔ 生成并保存工程"));
    saveBtn->setObjectName("Primary");
    auto *depBtn = new QPushButton(tr("⚡ 保存并部署到板"));
    btns->addWidget(saveBtn); btns->addWidget(depBtn); btns->addStretch(1);
    root->addLayout(btns);

    // ---- 接线 ----
    connect(srcType_, &QComboBox::currentIndexChanged, this, [this, srcBrowse](int i) {
        const bool v4l2 = (i == 1);
        srcPath_->setVisible(!v4l2); srcBrowse->setVisible(!v4l2);
        device_->setVisible(v4l2); width_->setVisible(v4l2); height_->setVisible(v4l2);
    });
    connect(srcBrowse, &QPushButton::clicked, this, [this] {
        const QString d = QFileDialog::getExistingDirectory(this, tr("图片目录"));
        if (!d.isEmpty()) srcPath_->setText(d);
    });
    connect(mdlBrowse, &QPushButton::clicked, this, [this] {
        const QString f = QFileDialog::getOpenFileName(this, tr("选择 RKNN 模型"),
                                                       QString(), tr("RKNN 模型 (*.rknn)"));
        if (!f.isEmpty()) model_->setText(f);
    });
    connect(addRule, &QPushButton::clicked, this, [this] {
        const int row = rules_->rowCount();
        rules_->insertRow(row);
        auto *type = new QComboBox;
        for (const char *t : {"count", "score", "size", "region", "persist", "regex"})
            type->addItem(QString::fromLatin1(t));
        rules_->setCellWidget(row, 0, type);
        rules_->setItem(row, 1, new QTableWidgetItem(QStringLiteral("*")));
    });
    connect(delRule, &QPushButton::clicked, this, [this] {
        const auto rows = rules_->selectionModel()->selectedRows();
        for (const auto &idx : rows) rules_->removeRow(idx.row());
    });
    connect(saveBtn, &QPushButton::clicked, this, [this] { onSave(false); });
    connect(depBtn, &QPushButton::clicked, this, [this] { onSave(true); });
}

void WorkflowPage::onSave(bool deploy) {
    WorkflowForm f;
    try {
        f = collect();
    } catch (const QString &msg) {
        QMessageBox::warning(this, tr("WORKFLOW"), msg);
        return;
    }
    QString err;
    const QString dir = WorkflowGen::save(f, projectsRoot_, &err);
    if (dir.isEmpty()) {
        QMessageBox::warning(this, tr("WORKFLOW"), err);
        return;
    }
    emit logLine(tr("WORKFLOW 已生成: %1").arg(dir), "#3ecf72");
    emit projectSaved(f.projectName);
    if (deploy) emit deployRequested(f.projectName);
}

WorkflowForm WorkflowPage::collect() const {
    WorkflowForm f;
    f.projectName = name_->text().trimmed();
    f.sourceV4l2 = (srcType_->currentIndex() == 1);
    f.sourcePath = srcPath_->text().trimmed();
    f.device = device_->text().trimmed();
    f.width = width_->value(); f.height = height_->value();
    f.size = size_->value(); f.padValue = pad_->value();
    f.modelFile = model_->text().trimmed();
    f.family = family_->currentText();
    f.classes = classes_->text().trimmed();
    f.conf = conf_->value(); f.iou = iou_->value();
    f.ngArchive = ngArchive_->isChecked();
    f.ngDir = ngDir_->text().trimmed();
    f.imageOut = imageOut_->isChecked();
    f.mjpeg = mjpegEn_->isChecked();
    f.mjpegPort = port_->value();

    for (int i = 0; i < rules_->rowCount(); ++i) {
        auto *type = qobject_cast<QComboBox *>(rules_->cellWidget(i, 0));
        if (!type) continue;
        WorkflowRule r;
        r.type = type->currentText();
        r.cls = rules_->item(i, 1) ? rules_->item(i, 1)->text().trimmed() : QStringLiteral("*");
        if (r.cls.isEmpty()) r.cls = QStringLiteral("*");
        r.param1 = rules_->item(i, 2) ? rules_->item(i, 2)->text().trimmed() : QString();
        r.param2 = rules_->item(i, 3) ? rules_->item(i, 3)->text().trimmed() : QString();
        if (r.param1.isEmpty() && r.type != QLatin1String("region"))
            continue;                          // 空行跳过（region 的参数在 param1，不适用）
        f.rules.append(r);
    }
    return f;
}
