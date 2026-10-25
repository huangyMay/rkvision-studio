#include "mainwindow.h"
#include "core/adbclient.h"
#include "core/deployer.h"
#include "core/mjpegclient.h"
#include "core/boardmonitor.h"
#include "core/acccompare.h"
#include "core/boardperf.h"
#include "core/inferstream.h"
#include "core/onnxanalyzer.h"
#include "core/onnxprobe.h"
#include "core/onnxrunner.h"
#include "core/projectregistry.h"
#include "core/rknnsimrunner.h"
#include "ui/workflowpage.h"
#include "ui/widgets/previewwidget.h"

#include <QButtonGroup>
#include <QBuffer>
#include <QCheckBox>
#include <QColor>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QLineEdit>
#include <QMediaPlayer>
#include <QPainter>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QMessageBox>
#include <QProcess>
#include <QScrollArea>
#include <QStandardPaths>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTextStream>
#include <QUrl>
#include <QVideoSink>
#include <QVideoFrame>
#include <QVideoWidget>
#include <QDialog>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QStyle>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>

namespace {
constexpr int kPreviewPort = 18080;   // adb forward 本地端口 → 板上 :8000
const char *kProjectsRoot = PROJECTS_ROOT;

QString resolveProjectsRoot();     // 定义在 buildUi 之后（同一匿名命名空间）
} // namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    buildUi();
    m_mjpeg = new MjpegClient(this);
    connect(m_mjpeg, &MjpegClient::frame, this, [this](const QImage &img) {
        m_preview->setFrame(img);
        ++m_frames;
        const qint64 now = QDateTime::currentMSecsSinceEpoch();
        if (now - m_lastFpsMs >= 1000) {
            m_preview->setStatus(QStringLiteral("%1 fps · PC 推理 %2ms")
                                     .arg(m_frames).arg(m_liveInferMs, 0, 'f', 0));
            m_frames = 0;
            m_lastFpsMs = now;
        }
    });
    connect(m_mjpeg, &MjpegClient::frameJpeg, this, [this](const QByteArray &jpg) {
        if (m_ckLiveInfer->isChecked()) m_istream->submitFrame(jpg);   // 忙时自动丢帧
    });
    connect(m_mjpeg, &MjpegClient::stateChanged, this, [this](const QString &s) {
        m_preview->setPlaceholder(s);
        appendLog(s, "#8b97a8");
    });
    m_onnx = new OnnxRunner(this);
    connect(m_onnx, &OnnxRunner::finished, this, [this](const OnnxResult &r) {
        m_btnInfer->setEnabled(true);
        if (!r.ok) {
            m_inferStats->setText(QStringLiteral("✗ %1").arg(r.error));
            appendLog(QStringLiteral("本地推理失败: %1").arg(r.error), "#ff6b6b");
            return;
        }
        drawInferResult(r);
        appendLog(QStringLiteral("本地推理完成: %1 · %2 框 · %3 ms")
                      .arg(r.model).arg(r.detections.size()).arg(r.totalMs, 0, 'f', 0),
                  "#3ecf72");
    });
    m_analyzer = new OnnxAnalyzer(this);
    m_probe = new OnnxProbeRunner(this);
    connect(m_probe, &OnnxProbeRunner::logLine, this, &MainWindow::appendLog);
    connect(m_probe, &OnnxProbeRunner::finished, this, [this](bool ok, const QString &json,
                                                              const QString &err) {
        m_btnProbe->setEnabled(true);
        if (!ok) {
            appendLog(QStringLiteral("逐层体检失败: %1").arg(err), "#ff6b6b");
            QMessageBox::warning(this, QStringLiteral("逐层体检"), err);
            return;
        }
        const QJsonDocument doc = QJsonDocument::fromJson(json.toUtf8());
        const QJsonObject o = doc.object();
        const QJsonArray sens = o.value("sensitive").toArray();
        QString sensStr;
        for (const QJsonValue &v : sens) {
            const QJsonObject s = v.toObject();
            sensStr += QStringLiteral("%1（outlier ×%2）<br>")
                           .arg(s.value("name").toString().toHtmlEscaped())
                           .arg(s.value("outlier").toDouble(), 0, 'f', 1);
        }
        if (sensStr.isEmpty())
            sensStr = QStringLiteral("无——逐层激活分布温和，int8 量化风险低");
        const QJsonArray layers = o.value("layers").toArray();

        QDialog dlg(this);
        dlg.setWindowTitle(QStringLiteral("逐层体检 — %1").arg(o.value("model").toString()));
        dlg.resize(980, 680);
        auto *v = new QVBoxLayout(&dlg);
        auto *sum = new QLabel(QStringLiteral(
            "<b>%1</b> · 输入 %2×%3 · 推理 <b>%4 ms</b> · 层数 <b>%5</b><br>"
            "<b>量化敏感层（outlier 比 > 10，int8 按 max 定 scale 会压掉小值分辨率）</b>：<br>%6")
            .arg(o.value("model").toString().toHtmlEscaped())
            .arg(o.value("input_hw").toArray().at(0).toInt())
            .arg(o.value("input_hw").toArray().at(1).toInt())
            .arg(o.value("total_ms").toDouble(), 0, 'f', 0)
            .arg(o.value("n_layers").toInt())
            .arg(sensStr));
        sum->setWordWrap(true);
        v->addWidget(sum);
        auto *te = new QTextEdit;
        te->setReadOnly(true);
        te->setPlainText(o.value("table").toString());
        QFont mono(QStringLiteral("Consolas"));
        mono.setStyleHint(QFont::Monospace);
        te->setFont(mono);
        v->addWidget(te, 1);
        auto *hint = new QLabel(
            QStringLiteral("完整报告（markdown）: %1").arg(o.value("report").toString().toHtmlEscaped()));
        v->addWidget(hint);
        dlg.exec();
    });
    connect(m_analyzer, &OnnxAnalyzer::finished, this, [this](const OnnxAnalysis &r) {
        m_btnAnalyze->setEnabled(true);
        if (!r.ok) {
            appendLog(QStringLiteral("模型分析失败: %1").arg(r.error), "#ff6b6b");
            return;
        }
        appendLog(QStringLiteral("模型分析完成: %1 · 参数 %2 · 总耗时 %3ms")
                      .arg(r.model).arg(r.params).arg(r.avgTotalMs, 0, 'f', 1), "#3ecf72");
        showAnalysis(r);
    });
    m_rknn = new RknnSimRunner(this);
    m_compare = new AccCompareRunner(this);
    connect(m_compare, &AccCompareRunner::logLine, this, &MainWindow::appendLog);
    connect(m_compare, &AccCompareRunner::flowLog, this, [this](const QString &msg) {
        const QString ts = QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss"));
        m_cmpLog->append(QStringLiteral("<span style='color:#5a6a80'>%1</span> %2").arg(ts, msg.toHtmlEscaped()));
    });
    // 流式：每级完成即时路由到对应图片格（不等全跑完）
    connect(m_compare, &AccCompareRunner::stageUpdate, this, [this](const QString &json) {
        const QJsonObject o = QJsonDocument::fromJson(json.toUtf8()).object();
        const QString st = o.value("stage").toString();
        const QString img = o.value("image").toString();
        if (!img.isEmpty()) {
            // 脚本回传 WSL 路径（/mnt/c/.. 或 /tmp/..），转 Qt(Windows) 可读形态
            QString p = img;
            if (p.startsWith(QStringLiteral("/mnt/")) && p.size() > 7)
                p = p.mid(5, 1).toUpper() + ":/" + p.mid(7);   // /mnt/c/.. -> C:/..（mid(7) 起是盘符后路径）
            else if (p.startsWith(QStringLiteral("/")))
                p = QStringLiteral("//wsl.localhost/Ubuntu-22.04") + p;
            const QImage im(p);
            if (!im.isNull()) {
                PreviewWidget *dst = nullptr;
                if (st == QStringLiteral("1_onnx")) dst = m_imgOnnx;
                else if (st == QStringLiteral("2_sim_fp16")) dst = m_imgSimFp16;
                else if (st == QStringLiteral("3_sim_int8")) dst = m_imgSimInt8;
                else if (st == QStringLiteral("4_sim_int16")) dst = m_imgSimInt16;
                else if (st == QStringLiteral("5_board_fp16")) dst = m_imgBdFp16;
                else if (st == QStringLiteral("6_board_int8")) dst = m_imgBdInt8;
                else if (st == QStringLiteral("7_board_int16")) dst = m_imgBdInt16;
                if (dst) dst->setImage(im);
            }
        }
        // 该级检测数即时进信息面板（操作流），失败/跳过也透明呈现
        const QJsonObject sts = o.value("stages").toObject().value(st).toObject();
        if (sts.contains(QStringLiteral("n")))
            m_cmpLog->append(QStringLiteral("<b>%1</b> 完成 · 检出 %2 框")
                                 .arg(st.toHtmlEscaped()).arg(sts.value("n").toInt()));
        else if (sts.contains(QStringLiteral("error")))
            m_cmpLog->append(QStringLiteral("<span style='color:#ff6b6b'>%1 失败: %2</span>")
                                 .arg(st.toHtmlEscaped(),
                                      sts.value("error").toString().toHtmlEscaped()));
        else if (sts.contains(QStringLiteral("skipped")))
            m_cmpLog->append(QStringLiteral("%1 —— 未连板，跳过").arg(st.toHtmlEscaped()));
        // 数据面板同步点亮：该通道页签的总体指标立即刷新
        refreshChanLabel(chanIndex(st), o);
    });
    connect(m_compare, &AccCompareRunner::layersUpdate, this, [this](const QString &json) {
        // {"ch": 通道键, "kind": "sim|rt", "layers": [...]} —— 填进对应通道页签
        const QJsonObject o = QJsonDocument::fromJson(json.toUtf8()).object();
        const int idx = chanIndex(o.value(QStringLiteral("ch")).toString());
        const QJsonArray layers = o.value(QStringLiteral("layers")).toArray();
        if (idx < 0 || layers.isEmpty()) return;
        fillLayerTable(m_chanTable[idx], layers);
        // 逐层摘要追加到该通道总体指标下（保留已有行，幂等替换）
        const bool sim = layers.first().toObject().contains(QStringLiteral("sim_entire_cos"));
        const QString cosk = sim ? QStringLiteral("sim_single_cos")
                                 : QStringLiteral("rt_single_cos");
        double mn = 2.0;
        int lows = 0;
        for (const QJsonValue &v : layers) {
            const double ss = v.toObject().value(cosk).toDouble(1.0);
            mn = qMin(mn, ss);
            if (ss < 0.99) ++lows;
        }
        QString base = m_chanLabel[idx]->text();
        base.remove(QRegularExpression(QStringLiteral("^逐层: .*$"),
                                       QRegularExpression::MultilineOption));
        m_chanLabel[idx]->setText(base.trimmed() + QStringLiteral("\n逐层: %1 层 · 最低 %2 %3 · 低于0.99 %4 层")
                                      .arg(layers.size()).arg(sim ? "sim" : "rt")
                                      .arg(mn, 0, 'f', 5).arg(lows));
    });
    connect(m_compare, &AccCompareRunner::finished, this, [this](bool ok, const QString &json,
                                                                 const QString &err) {
        m_btnCompare->setEnabled(true);
        if (!ok) {
            appendLog(QStringLiteral("精度对比失败: %1").arg(err), "#ff6b6b");
            QMessageBox::warning(this, QStringLiteral("精度对比"), err);
            return;
        }
        const QJsonObject o = QJsonDocument::fromJson(json.toUtf8()).object();
        updateCompareSummary(o);                  // 漏斗摘要进信息面板（不在对比面板显示）
        m_lastCompareJson = json;                 // 供 💾 导出
        m_cmpDumpDir = o.value("dump_dir").toString();
        appendLog(QStringLiteral("四级对比完成（%1 s）——逐层表 %2")
                      .arg(o.value("total_ms").toDouble() / 1000, 0, 'f', 0)
                      .arg(o.contains(QStringLiteral("layers"))
                               ? QStringLiteral("已填充") : QStringLiteral("见「🎯量化精度」")),
                  "#3ecf72");
    });
    connect(m_rknn, &RknnSimRunner::finished, this, [this](const OnnxResult &r) {
        m_btnRkSim->setEnabled(true);
        if (!r.ok) {
            appendLog(QStringLiteral("转换/模拟失败: %1").arg(r.error), "#ff6b6b");
            return;
        }
        appendLog(QStringLiteral("转换+模拟完成(×86, 非板端性能): %1 · %2 框 · build %3ms · 推理 %4ms%5")
                      .arg(r.model).arg(r.detections.size())
                      .arg(r.loadMs, 0, 'f', 0).arg(r.inferMs, 0, 'f', 0)
                      .arg(r.rknnSaved ? QStringLiteral(" · ✅ .rknn 已存盘")
                                       : QStringLiteral(" · ⚠ 未填存盘路径，仅模拟")),
                  "#3ecf72");
    });
    m_boardPerf = new BoardPerfRunner(this);
    connect(m_boardPerf, &BoardPerfRunner::finished, this, [this](const BoardPerf &r) {
        m_btnBoardPerf->setEnabled(true);
        if (!r.ok) {
            appendLog(QStringLiteral("板端性能失败: %1 %2").arg(r.error, r.rawOut), "#ff6b6b");
            return;
        }
        const QString runsStr = r.e2eMs.isEmpty()
            ? QString() : QStringLiteral("（%1 框×%2 轮： %3）")
                .arg(r.framesPerRun).arg(r.e2eMs.size())
                .arg([&]{ QString s; for (double v : r.e2eMs) s += QStringLiteral("%1 ").arg(v, 0, 'f', 1); return s; }());
        appendLog(QStringLiteral("📡 板端 NPU 真测: 推理增量 %1ms/帧 · full e2e %2ms · base %3ms · %4 · min %5ms %6")
                      .arg(r.inferMs, 0, 'f', 1).arg(r.avgMs, 0, 'f', 1).arg(r.baseAvgMs, 0, 'f', 1)
                      .arg(r.model).arg(r.minMs, 0, 'f', 1).arg(runsStr),
                  "#3ecf72");
    });
    refreshStatus();
}

void MainWindow::buildUi() {
    setWindowTitle(QStringLiteral("RKVision Studio"));
    resize(1280, 800);

    auto *central = new QWidget;
    setCentralWidget(central);
    auto *root = new QVBoxLayout(central);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    // ---- 顶栏 ----
    auto *header = new QFrame;
    header->setObjectName("Header");
    header->setFixedHeight(56);
    auto *hl = new QHBoxLayout(header);
    hl->setContentsMargins(18, 0, 18, 0);
    auto *title = new QLabel(QStringLiteral("RVS · RKVISION STUDIO"));
    title->setObjectName("AppTitle");
    auto *sub = new QLabel(QStringLiteral("// EDGE VISION WORKBENCH"));
    sub->setObjectName("AppSub");
    m_devicePill = new QLabel(QStringLiteral("NO DEVICE"));
    m_devicePill->setObjectName("Pill");
    m_devicePill->setProperty("class", "PillIdle");
    m_serialBox = new QComboBox;
    m_serialBox->setFixedWidth(190);
    auto *btnRefresh = new QPushButton(QStringLiteral("⟳"));
    btnRefresh->setFixedWidth(44);
    hl->addWidget(title);
    hl->addWidget(sub);
    hl->addStretch();
    hl->addWidget(m_devicePill);
    hl->addWidget(m_serialBox);
    hl->addWidget(btnRefresh);
    root->addWidget(header);

    // ---- 页面导航栏：LIVE 实时预览 / ONNX LAB 本地推理 两个工作页 ----
    auto *nav = new QFrame;
    nav->setObjectName("NavBar");
    nav->setFixedHeight(42);
    auto *nl = new QHBoxLayout(nav);
    nl->setContentsMargins(18, 0, 18, 0);
    nl->setSpacing(8);
    auto *navLive = new QPushButton(QStringLiteral("◉ LIVE / 实时预览"));
    navLive->setObjectName("NavTab");
    navLive->setCheckable(true);
    navLive->setChecked(true);
    auto *navLab = new QPushButton(QStringLiteral("🧪 模型实验室 / 精度分析"));
    navLab->setObjectName("NavTab");
    navLab->setCheckable(true);
    auto *navWorkflow = new QPushButton(QStringLiteral("🛠 WORKFLOW / 工作流"));
    navWorkflow->setObjectName("NavTab");
    navWorkflow->setCheckable(true);
    m_navLive = navLive;
    m_navLab = navLab;
    m_navWorkflow = navWorkflow;
    auto *navGroup = new QButtonGroup(this);
    navGroup->setExclusive(true);
    navGroup->addButton(navLive);
    navGroup->addButton(navLab);
    navGroup->addButton(navWorkflow);
    nl->addWidget(navLive);
    nl->addWidget(navLab);
    nl->addWidget(navWorkflow);
    nl->addStretch();
    root->addWidget(nav);

    m_views = new QStackedWidget;
    root->addWidget(m_views, 1);

    // ---- 页 0：LIVE 实时预览（工程/信号源/日志 + 预览大屏） ----
    auto *pageLive = new QWidget;
    m_views->addWidget(pageLive);
    auto *body = new QHBoxLayout(pageLive);
    body->setContentsMargins(14, 14, 14, 14);
    body->setSpacing(14);

    auto *left = new QVBoxLayout;
    left->setSpacing(12);
    // 小屏/高缩放：左列内容超出窗口高度时出滚动条，不再挤压重叠
    auto *leftScroll = new QScrollArea;
    leftScroll->setWidgetResizable(true);
    leftScroll->setFrameShape(QFrame::NoFrame);
    leftScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto *leftBody = new QWidget;
    leftBody->setLayout(left);
    leftScroll->setWidget(leftBody);
    leftScroll->setFixedWidth(372);
    body->addWidget(leftScroll, 0);
    body->setStretch(0, 0);
    body->setStretch(1, 1);

    auto makeCard = [](QVBoxLayout *target, QVBoxLayout **layout, const QString &title) {
        auto *card = new QFrame;
        card->setObjectName("Card");
        auto *v = new QVBoxLayout(card);
        v->setContentsMargins(14, 12, 14, 14);
        auto *t = new QLabel(title);
        t->setObjectName("CardTitle");
        v->addWidget(t);
        target->addWidget(card);
        *layout = v;
        return card;
    };
    QVBoxLayout *projL, *prevL, *infoL, *logL;
    makeCard(left, &projL, QStringLiteral("PROJECT / 工程"));
    makeCard(left, &prevL, QStringLiteral("SOURCE / 信号源与预览"));
    makeCard(left, &infoL, QStringLiteral("INFER INFO / 推理信息"));
    makeCard(left, &logL, QStringLiteral("LOG / 日志"));

    // 工程卡
    m_projectBox = new QComboBox;
    projL->addWidget(m_projectBox);
    m_btnDeploy = new QPushButton(QStringLiteral("① 部署到板"));
    m_btnDeploy->setObjectName("Primary");
    m_btnRun = new QPushButton(QStringLiteral("② 运行一遍"));
    projL->addWidget(m_btnDeploy);
    projL->addWidget(m_btnRun);

    // 预览卡
    m_sourceBox = new QComboBox;
    m_sourceBox->addItem(QStringLiteral("引擎 MJPEG（USB/forward）"));
    m_sourceBox->addItem(QStringLiteral("相机 MJPEG（网络直连）"));
    m_sourceBox->addItem(QStringLiteral("相机 RTSP → mediamtx"));
    prevL->addWidget(m_sourceBox);
    m_rtspUrl = new QLineEdit(QStringLiteral("rtsp://127.0.0.1:8554/hdmiin"));
    prevL->addWidget(m_rtspUrl);
    m_btnPreview = new QPushButton(QStringLiteral("▶ 启动预览"));
    m_btnPreview->setObjectName("Primary");
    m_btnStop = new QPushButton(QStringLiteral("■ 停止"));
    m_btnStop->setObjectName("Danger");
    prevL->addWidget(m_btnPreview);
    prevL->addWidget(m_btnStop);
    // PC 实时推理：预览流抽帧送 ONNX Runtime，检测框叠加在大屏上
    m_ckLiveInfer = new QCheckBox(QStringLiteral("⚡ PC 实时推理（预览叠加检测框）"));
    prevL->addWidget(m_ckLiveInfer);
    auto *liveRow = new QHBoxLayout;
    liveRow->setSpacing(6);
    m_liveModelPath = new QLineEdit;
    m_liveModelPath->setPlaceholderText(QStringLiteral("ONNX 模型（实时推理用）"));
    liveRow->addWidget(m_liveModelPath, 1);
    auto *liveBrowse = new QPushButton(QStringLiteral("…"));
    liveBrowse->setFixedWidth(34);
    liveRow->addWidget(liveBrowse);
    prevL->addLayout(liveRow);
    connect(liveBrowse, &QPushButton::clicked, this, [this] {
        const QString f = QFileDialog::getOpenFileName(this, QStringLiteral("选择 ONNX 模型"),
                                                       QString(), QStringLiteral("ONNX 模型 (*.onnx)"));
        if (!f.isEmpty()) m_liveModelPath->setText(f);
    });
    connect(m_sourceBox, &QComboBox::currentIndexChanged, this, [this](int i) {
        m_rtspUrl->setEnabled(i == 1);
    });
    m_rtspUrl->setEnabled(false);

    // 推理信息卡：模型静态信息 / 每帧分解 / 滚动统计 / 逐层剖析（PC 实时推理开启后刷新）
    m_inferInfo = new QTextEdit;
    m_inferInfo->setObjectName("Log");
    m_inferInfo->setReadOnly(true);
    m_inferInfo->setFixedHeight(150);
    m_inferInfo->setPlaceholderText(QStringLiteral("勾选「PC 实时推理」后显示模型信息与逐帧/统计指标"));
    infoL->addWidget(m_inferInfo);
    auto *infoBtnRow = new QHBoxLayout;
    infoBtnRow->setSpacing(6);
    auto *btnProf = new QPushButton(QStringLiteral("📊 逐层剖析（20帧）"));
    infoBtnRow->addWidget(btnProf);
    infoBtnRow->addStretch();
    infoL->addLayout(infoBtnRow);
    connect(btnProf, &QPushButton::clicked, this, [this] {
        if (!m_istream->ready()) {
            appendLog(QStringLiteral("逐层剖析：请先开启 PC 实时推理"), "#ff6b6b");
            return;
        }
        m_inferInfo->append(QStringLiteral("逐层剖析中（独立 session 跑 20 帧）…"));
        m_istream->submitCommand(QStringLiteral("{\"cmd\":\"profile\",\"runs\":20}"));
    });

    // 日志卡（置底伸展）
    m_log = new QTextEdit;
    m_log->setObjectName("Log");
    m_log->setReadOnly(true);
    logL->addWidget(m_log, 1);
    left->addStretch();

    // 页 0 右侧预览大屏：MJPEG 与 RTSP(QVideoSink 解码) 都走 setImage
    auto *right = new QVBoxLayout;
    right->setSpacing(8);
    body->addLayout(right, 1);
    m_preview = new PreviewWidget;
    m_preview->setPlaceholder(QStringLiteral("NO SIGNAL — 启动预览或等待信号源"));
    right->addWidget(m_preview, 1);
    // 板端状态条：CPU/内存/NPU/温度，2 秒轮询
    auto *bStatsBar = new QFrame;
    bStatsBar->setObjectName("StatsBar");
    auto *bStatsL = new QHBoxLayout(bStatsBar);
    bStatsL->setContentsMargins(12, 6, 12, 6);
    m_boardStats = new QLabel(QStringLiteral("BOARD — 等待设备"));
    m_boardStats->setObjectName("StatsText");
    bStatsL->addWidget(m_boardStats);
    right->addWidget(bStatsBar);
    m_bmon = new BoardMonitor(this);
    connect(m_bmon, &BoardMonitor::stats, this, [this](const BoardMonitor::Sample &s) {
        if (!s.ok) { m_boardStats->setText(QStringLiteral("BOARD — 采样失败")); return; }
        QString npu = s.npuValid
            ? (s.npuCore1 >= 0 ? QStringLiteral("NPU %1%+%2%").arg(s.npuCore0, 0, 'f', 0).arg(s.npuCore1, 0, 'f', 0)
                               : QStringLiteral("NPU %1%").arg(s.npuCore0, 0, 'f', 0))
            : QStringLiteral("NPU n/a(需root)");
        const QString cpu = s.cpuPct >= 0 ? QStringLiteral("%1%").arg(s.cpuPct, 0, 'f', 0)
                                          : QStringLiteral("--");
        m_boardStats->setText(QStringLiteral(
            "CPU %1 · 内存 %2/%3G (%4%) · %5 · %6°C")
            .arg(cpu)
            .arg(s.memUsedMb / 1024.0, 0, 'f', 1)
            .arg(s.memTotalMb / 1024.0, 0, 'f', 1)
            .arg(s.memTotalMb > 0 ? 100.0 * s.memUsedMb / s.memTotalMb : 0.0, 0, 'f', 0)
            .arg(npu)
            .arg(s.tempC, 0, 'f', 1));
    });
    auto *bmonTimer = new QTimer(this);
    connect(bmonTimer, &QTimer::timeout, this, [this] {
        const QString serial = m_serialBox->currentText();
        if (!serial.isEmpty() && serial != QStringLiteral("(无设备)"))
            m_bmon->poll(serial);
    });
    bmonTimer->start(2000);
    // PC 实时推理流
    m_istream = new InferStream(this);
    connect(m_istream, &InferStream::workerText, this, [this](const QString &t) {
        m_inferInfo->append(t);
    });
    connect(m_istream, &InferStream::readyChanged, this, [this](bool ready) {
        if (ready) appendLog(QStringLiteral("PC 实时推理 worker 就绪"), "#3ecf72");
    });
    connect(m_istream, &InferStream::result, this,
            [this](const QVector<OnnxDetection> &dets, int w, int h, double ms) {
        QVector<QPair<QRectF, QString>> overlay;
        overlay.reserve(dets.size());
        for (const OnnxDetection &d : dets) {
            overlay.append({ QRectF(d.x1 / w, d.y1 / h,
                                    (d.x2 - d.x1) / w, (d.y2 - d.y1) / h),
                             QStringLiteral("cls%1 %2").arg(d.cls).arg(d.conf, 0, 'f', 2) });
        }
        m_preview->setOverlay(overlay);
        m_liveInferMs = ms;                    // 状态行由各 fps 计数器统一显示
    });
    connect(m_ckLiveInfer, &QCheckBox::toggled, this, [this](bool on) {
        if (!on) {
            m_istream->stop();
            m_preview->clearOverlay();
            appendLog(QStringLiteral("PC 实时推理已关闭"));
            return;
        }
        QString model = m_liveModelPath->text().trimmed();
        // 免粘贴：未填时自动尝试 exe 旁 samples\yolov8n.onnx
        if (model.isEmpty()) {
            const QString def = QCoreApplication::applicationDirPath()
                                + QStringLiteral("/samples/yolov8n.onnx");
            if (QFileInfo::exists(def)) {
                model = QDir::toNativeSeparators(def);
                m_liveModelPath->setText(model);
                appendLog(QStringLiteral("未选模型，已自动使用 %1").arg(model), "#8b97a8");
            }
        }
        if (model.isEmpty() || !QFileInfo::exists(model)) {
            appendLog(QStringLiteral("PC 实时推理：模型路径无效——点 … 选择 .onnx"), "#ff6b6b");
            m_ckLiveInfer->setChecked(false);
            return;
        }
        m_istream->start(model, 0.25, 0.45);
        appendLog(QStringLiteral("PC 实时推理开启: %1（预览流抽帧→ONNX Runtime）")
                      .arg(QFileInfo(model).fileName()), "#3ecf72");
    });
    m_player = new QMediaPlayer(this);
    m_sink = new QVideoSink(this);
    m_player->setVideoSink(m_sink);
    connect(m_sink, &QVideoSink::videoFrameChanged, this, [this](const QVideoFrame &f) {
        if (f.isValid()) {
            const QImage img = f.toImage();
            m_preview->setFrame(img);
            ++m_frames;
            const qint64 now = QDateTime::currentMSecsSinceEpoch();
            if (now - m_lastFpsMs >= 1000) {
                m_preview->setStatus(QStringLiteral("RTSP %1 fps · PC 推理 %2ms")
                                         .arg(m_frames).arg(m_liveInferMs, 0, 'f', 0));
                m_frames = 0;
                m_lastFpsMs = now;
            }
            // PC 实时推理：RTSP 解码帧同样接入（编码 JPEG 送 worker，忙时自动丢弃）
            if (m_ckLiveInfer->isChecked() && m_istream->ready() && !m_istream->busy()) {
                QByteArray jpg;
                QBuffer buf(&jpg);
                buf.open(QIODevice::WriteOnly);
                img.save(&buf, "JPEG", 80);
                m_istream->submitFrame(jpg);
            }
        }
    });
    connect(m_player, &QMediaPlayer::errorOccurred, this,
            [this](QMediaPlayer::Error, const QString &s) {
                appendLog(QStringLiteral("RTSP 播放错误: %1").arg(s), "#ff6b6b");
            });
    connect(m_player, &QMediaPlayer::mediaStatusChanged, this,
            [this](QMediaPlayer::MediaStatus st) {
                QString s = QStringLiteral("未知");
                switch (st) {
                case QMediaPlayer::LoadingMedia: s = "加载中"; break;
                case QMediaPlayer::LoadedMedia: s = "已加载"; break;
                case QMediaPlayer::StalledMedia: s = "缓冲停滞"; break;
                case QMediaPlayer::BufferingMedia: s = "缓冲中"; break;
                case QMediaPlayer::BufferedMedia: s = "缓冲完成"; break;
                case QMediaPlayer::EndOfMedia: s = "结束"; break;
                case QMediaPlayer::InvalidMedia: s = "无效媒体"; break;
                default: break;
                }
                appendLog(QStringLiteral("QMediaPlayer: %1").arg(s));
            });

    // ---- 页 1：ONNX LAB 本地推理（模型/图片 + 独立大屏 + 推理细节条） ----
    // ---- 页 1：模型实验室（合并 ONNX LAB + RKNN LAB，内部分段切换）----
    auto *pageLab = new QWidget;
    m_views->addWidget(pageLab);                     // m_views index 1
    auto *labOuter = new QVBoxLayout(pageLab);
    labOuter->setContentsMargins(0, 0, 0, 0);
    labOuter->setSpacing(0);
    auto *segRow = new QHBoxLayout;
    segRow->setContentsMargins(14, 10, 14, 0);
    segRow->setSpacing(8);
    m_segInfer = new QPushButton(QStringLiteral("⚡ ONNX 基线 / 本地推理"));
    m_segInfer->setObjectName("NavTab");
    m_segInfer->setCheckable(true);
    m_segInfer->setChecked(true);
    m_segSim = new QPushButton(QStringLiteral("🧩 模拟量化 / 板端验证"));
    m_segSim->setObjectName("NavTab");
    m_segSim->setCheckable(true);
    auto *segGroup = new QButtonGroup(this);
    segGroup->setExclusive(true);
    segGroup->addButton(m_segInfer);
    segGroup->addButton(m_segSim);
    segRow->addWidget(m_segInfer);
    segRow->addWidget(m_segSim);
    segRow->addStretch();
    labOuter->addLayout(segRow);
    m_labStack = new QStackedWidget;
    labOuter->addWidget(m_labStack, 1);

    auto *pageInfer = new QWidget;
    m_labStack->addWidget(pageInfer);                // labStack index 0
    auto *ibody = new QHBoxLayout(pageInfer);
    ibody->setContentsMargins(14, 14, 14, 14);
    ibody->setSpacing(14);

    auto *ileft = new QVBoxLayout;
    ileft->setSpacing(12);
    ibody->addLayout(ileft, 0);
    ibody->setStretch(0, 0);
    ibody->setStretch(1, 1);

    QVBoxLayout *modelL;
    makeCard(ileft, &modelL, QStringLiteral("MODEL / 本地 ONNX 验证"));
    // 选模型 + 选图片 → WSL onnxruntime 推理 → 本页大屏画框
    auto modelRow = [this, modelL](QLineEdit **edit, const QString &hint,
                                   const QString &filter) {
        auto *row = new QHBoxLayout;
        row->setSpacing(6);
        *edit = new QLineEdit;
        (*edit)->setPlaceholderText(hint);
        auto *btn = new QPushButton(QStringLiteral("…"));
        btn->setFixedWidth(34);
        row->addWidget(*edit, 1);
        row->addWidget(btn);
        modelL->addLayout(row);
        connect(btn, &QPushButton::clicked, this, [this, edit, filter] {
            const QString f = QFileDialog::getOpenFileName(this, QString(), QString(), filter);
            if (!f.isEmpty()) (*edit)->setText(f);
        });
    };
    modelRow(&m_modelPath, QStringLiteral("模型 .onnx 路径"),
             QStringLiteral("ONNX 模型 (*.onnx)"));
    modelRow(&m_imagePath, QStringLiteral("图片 .jpg/.png 路径"),
             QStringLiteral("图片 (*.jpg *.jpeg *.png *.bmp)"));
    // 阈值参数：置信度 / NMS IoU
    auto *paramRow = new QHBoxLayout;
    paramRow->setSpacing(6);
    auto makeSpin = [](QDoubleSpinBox **spin, double val) {
        *spin = new QDoubleSpinBox;
        (*spin)->setRange(0.05, 0.95);
        (*spin)->setSingleStep(0.05);
        (*spin)->setDecimals(2);
        (*spin)->setValue(val);
        (*spin)->setFixedWidth(64);
        (*spin)->setAlignment(Qt::AlignCenter);
    };
    paramRow->addWidget(new QLabel(QStringLiteral("置信度")));
    makeSpin(&m_confSpin, 0.25);
    paramRow->addWidget(m_confSpin);
    paramRow->addStretch();
    paramRow->addWidget(new QLabel(QStringLiteral("NMS IoU")));
    makeSpin(&m_iouSpin, 0.45);
    paramRow->addWidget(m_iouSpin);
    modelL->addLayout(paramRow);
    m_btnInfer = new QPushButton(QStringLiteral("⚡ 本地推理"));
    m_btnInfer->setObjectName("Primary");
    modelL->addWidget(m_btnInfer);
    connect(m_btnInfer, &QPushButton::clicked, this, &MainWindow::onInfer);
    // 模型分析 / Netron / 逐层体检 一行三钮
    auto *toolRow = new QHBoxLayout;
    toolRow->setSpacing(8);
    m_btnAnalyze = new QPushButton(QStringLiteral("📊 模型分析"));
    m_btnProbe = new QPushButton(QStringLiteral("🧪 逐层体检"));
    m_btnNetron = new QPushButton(QStringLiteral("🔍 Netron"));
    toolRow->addWidget(m_btnAnalyze, 1);
    toolRow->addWidget(m_btnProbe, 1);
    toolRow->addWidget(m_btnNetron, 1);
    modelL->addLayout(toolRow);
    connect(m_btnAnalyze, &QPushButton::clicked, this, &MainWindow::onAnalyze);
    connect(m_btnProbe, &QPushButton::clicked, this, &MainWindow::onProbe);
    connect(m_btnNetron, &QPushButton::clicked, this, &MainWindow::onNetronView);
    ileft->addStretch();

    auto *iright = new QVBoxLayout;
    iright->setSpacing(8);
    ibody->addLayout(iright, 1);
    m_inferPreview = new PreviewWidget;
    m_inferPreview->setPlaceholder(QStringLiteral("ONNX LAB — 选择模型与图片后本地推理"));
    iright->addWidget(m_inferPreview, 1);
    auto *statsBar = new QFrame;
    statsBar->setObjectName("StatsBar");
    auto *statsL = new QHBoxLayout(statsBar);
    statsL->setContentsMargins(12, 6, 12, 6);
    m_inferStats = new QLabel(QStringLiteral("ONNX READY — 选择模型与图片后本地推理"));
    m_inferStats->setObjectName("StatsText");
    statsL->addWidget(m_inferStats);
    iright->addWidget(statsBar);

    connect(navLive, &QPushButton::clicked, this, [this] { m_views->setCurrentIndex(0); });
    connect(m_navLab, &QPushButton::clicked, this, [this] { m_views->setCurrentIndex(1); });
    connect(m_navWorkflow, &QPushButton::clicked, this, [this] { m_views->setCurrentIndex(2); });
    connect(m_segInfer, &QPushButton::clicked, this, [this] { m_labStack->setCurrentIndex(0); });
    connect(m_segSim, &QPushButton::clicked, this, [this] { m_labStack->setCurrentIndex(1); });

    // ---- 页 2：RKNN LAB 模拟推理（toolkit2 模拟器：ONNX+转换配置 → build → 模拟） ----
    auto *pageRknn = new QWidget;
    m_labStack->addWidget(pageRknn);                 // labStack index 1
    auto *rbody = new QHBoxLayout(pageRknn);
    rbody->setContentsMargins(14, 14, 14, 14);
    rbody->setSpacing(14);

    auto *rleft = new QVBoxLayout;
    rleft->setSpacing(12);
    rbody->addLayout(rleft, 0);
    rbody->setStretch(0, 0);
    rbody->setStretch(1, 1);

    QVBoxLayout *rkL;
    makeCard(rleft, &rkL, QStringLiteral("RKNN / 模拟推理（toolkit2 simulator）"));
    auto rkRow = [this, rkL](QLineEdit **edit, const QString &hint,
                             const QString &filter) {
        auto *row = new QHBoxLayout;
        row->setSpacing(6);
        *edit = new QLineEdit;
        (*edit)->setPlaceholderText(hint);
        auto *btn = new QPushButton(QStringLiteral("…"));
        btn->setFixedWidth(34);
        row->addWidget(*edit, 1);
        row->addWidget(btn);
        rkL->addLayout(row);
        connect(btn, &QPushButton::clicked, this, [this, edit, filter] {
            const QString f = QFileDialog::getOpenFileName(this, QString(), QString(), filter);
            if (!f.isEmpty()) (*edit)->setText(f);
        });
    };
    rkRow(&m_rkModelPath, QStringLiteral("模型 .onnx 路径（模拟输入=ONNX+转换配置）"),
          QStringLiteral("ONNX 模型 (*.onnx)"));
    rkRow(&m_rkImagePath, QStringLiteral("图片 .jpg/.png 路径"),
          QStringLiteral("图片 (*.jpg *.jpeg *.png *.bmp)"));
    auto *msRow = new QHBoxLayout;
    msRow->setSpacing(6);
    msRow->addWidget(new QLabel(QStringLiteral("转换配置")));
    m_rkMeanStd = new QLineEdit(QStringLiteral("0,0,0/255,255,255"));
    m_rkMeanStd->setToolTip(QStringLiteral("mean_values/std_values（与转换脚本一致，/ 分隔）；none/none = 无归一化"));
    msRow->addWidget(m_rkMeanStd, 1);
    rkL->addLayout(msRow);
    auto *saveRow = new QHBoxLayout;
    saveRow->setSpacing(6);
    auto *saveLabel = new QLabel(QStringLiteral("存/导入"));
    saveLabel->setToolTip(QStringLiteral("模拟时=build 产物的保存路径；板端性能时=被真测的 .rknn（外部现成的也行）"));
    saveRow->addWidget(saveLabel);
    m_rkSavePath = new QLineEdit;
    m_rkSavePath->setPlaceholderText(QStringLiteral(".rknn 路径（build 存盘 或 导入外部现成的）"));
    saveRow->addWidget(m_rkSavePath, 1);
    m_btnRkSim = new QPushButton(QStringLiteral("💾 转换导出"));
    m_btnRkSim->setToolTip(QStringLiteral("ONNX→RKNN 转换并按上面转换配置存盘 .rknn（附带 x86 模拟自检）"));
    saveRow->addWidget(m_btnRkSim);
    auto *btnRkBrowse = new QPushButton(QStringLiteral("…"));
    btnRkBrowse->setFixedWidth(34);
    saveRow->addWidget(btnRkBrowse);
    rkL->addLayout(saveRow);
    connect(m_btnRkSim, &QPushButton::clicked, this, &MainWindow::onRknnSim);
    connect(btnRkBrowse, &QPushButton::clicked, this, [this] {
        const QString f = QFileDialog::getOpenFileName(this, QStringLiteral("选择 .rknn 模型"),
                                                       QString(), QStringLiteral("RKNN 模型 (*.rknn)"));
        if (!f.isEmpty()) m_rkSavePath->setText(f);
    });
    auto *rkParamRow = new QHBoxLayout;
    rkParamRow->setSpacing(6);
    auto mkSpin = [](QDoubleSpinBox **spin, double val) {
        *spin = new QDoubleSpinBox;
        (*spin)->setRange(0.05, 0.95);
        (*spin)->setSingleStep(0.05);
        (*spin)->setDecimals(2);
        (*spin)->setValue(val);
        (*spin)->setFixedWidth(64);
        (*spin)->setAlignment(Qt::AlignCenter);
    };
    rkParamRow->addWidget(new QLabel(QStringLiteral("置信度")));
    mkSpin(&m_rkConfSpin, 0.25);
    rkParamRow->addWidget(m_rkConfSpin);
    rkParamRow->addStretch();
    rkParamRow->addWidget(new QLabel(QStringLiteral("NMS IoU")));
    mkSpin(&m_rkIouSpin, 0.45);
    rkParamRow->addWidget(m_rkIouSpin);
    rkL->addLayout(rkParamRow);
    // ⚖️ 精度对比：七通道全量（ONNX + 模拟/板上 × FP16/int8/int16 + 逐层），同行的板地址供其连板
    auto *cmpRow = new QHBoxLayout;
    m_btnCompare = new QPushButton(QStringLiteral("⚖️ 精度对比（七通道全量：模拟/板上 × FP16/int8/int16 + 逐层）"));
    m_btnCompare->setObjectName("Primary");
    m_accDevice = new QLineEdit(QStringLiteral("192.168.3.163:5555"));
    m_accDevice->setToolTip(QStringLiteral("板端网络 adb 地址（IP:端口）——⚖️ 板上通道与 📡 板端性能共用"));
    cmpRow->addWidget(m_btnCompare, 1);
    cmpRow->addWidget(m_accDevice);
    rkL->addLayout(cmpRow);
    connect(m_btnCompare, &QPushButton::clicked, this, &MainWindow::onAccCompare);
    m_btnBoardPerf = new QPushButton(QStringLiteral("📡 板端性能（NPU 真测，需板在线，用「存/导入」里的 .rknn）"));
    rkL->addWidget(m_btnBoardPerf);
    connect(m_btnBoardPerf, &QPushButton::clicked, this, &MainWindow::onBoardPerf);
    rleft->addStretch();

    auto *rright = new QVBoxLayout;
    rright->setSpacing(8);
    rbody->addLayout(rright, 1);

    // ---- 精度对比面板（⚖️ 运行/结果就地显示在本页右大区，流式填充） ----
    auto *pageCompareView = new QWidget;              // 右大区=精度对比面板（⚖️ 流式填充）
    rright->addWidget(pageCompareView, 1);
    auto *cvLayout = new QVBoxLayout(pageCompareView);
    cvLayout->setContentsMargins(4, 4, 4, 4);
    cvLayout->setSpacing(6);

    // 切换行+导出固定在面板顶部（不随内容滚动消失）
    auto *cvTail = new QHBoxLayout;
    auto *infoBtn = new QPushButton(QStringLiteral("📋 信息面板"));
    infoBtn->setCheckable(true);
    infoBtn->setChecked(true);
    auto *picBtn = new QPushButton(QStringLiteral("🖼 图片面板"));
    picBtn->setCheckable(true);
    auto *dataBtn = new QPushButton(QStringLiteral("📊 数据面板"));
    dataBtn->setCheckable(true);
    auto *viewGroup = new QButtonGroup(this);
    viewGroup->setExclusive(true);
    viewGroup->addButton(infoBtn);
    viewGroup->addButton(picBtn);
    viewGroup->addButton(dataBtn);
    connect(infoBtn, &QPushButton::clicked, this, [this] { m_cmpViewStack->setCurrentIndex(0); });
    connect(picBtn, &QPushButton::clicked, this, [this] { m_cmpViewStack->setCurrentIndex(1); });
    connect(dataBtn, &QPushButton::clicked, this, [this] { m_cmpViewStack->setCurrentIndex(2); });
    m_btnExport = new QPushButton(QStringLiteral("💾 导出报告"));
    cvTail->addWidget(infoBtn);
    cvTail->addWidget(picBtn);
    cvTail->addWidget(dataBtn);
    cvTail->addStretch(1);
    cvTail->addWidget(m_btnExport);
    cvLayout->addLayout(cvTail);

    // 内容区整体套 QScrollArea：任何屏幕/任何内容量都不丢按钮（固定窗口+滚动条）
    m_cmpViewStack = new QStackedWidget;
    auto *cmpScroll = new QScrollArea;
    cmpScroll->setWidgetResizable(true);
    cmpScroll->setFrameShape(QFrame::NoFrame);
    cmpScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    cmpScroll->setWidget(m_cmpViewStack);
    cvLayout->addWidget(cmpScroll, 1);

    // 视图 0：信息面板（操作流日志 + 摘要，默认显示）
    auto *infoPanel = new QWidget;
    auto *iv = new QVBoxLayout(infoPanel);
    iv->setContentsMargins(0, 0, 0, 0);
    iv->setSpacing(6);
    m_cmpLog = new QTextEdit;
    m_cmpLog->setReadOnly(true);
    m_cmpLog->setFont(QFont(QStringLiteral("Consolas")));
    m_cmpLog->setPlaceholderText(QStringLiteral(
        "操作流将在此实时显示：识别模型 → FP16/int8 转换 → 四级推理 → 检出结果…"));
    iv->addWidget(m_cmpLog, 1);
    m_cmpViewStack->addWidget(infoPanel);             // m_cmpViewStack index 0

    // 视图 0：图片面板（2×2）
    auto *imgPanel = new QWidget;
    auto *imgGrid = new QGridLayout(imgPanel);
    imgGrid->setContentsMargins(0, 0, 0, 0);
    imgGrid->setSpacing(6);
    auto mkCell = [](QGridLayout *g, PreviewWidget **w, const QString &ph,
                     int r, int c, const QString &tag) {
        auto *col = new QVBoxLayout;
        auto *t = new QLabel(tag);
        t->setObjectName("CardTitle");
        col->addWidget(t);
        *w = new PreviewWidget;
        (*w)->setPlaceholder(ph);
        col->addWidget(*w, 1);
        g->addLayout(col, r, c);
    };
    mkCell(imgGrid, &m_imgOnnx,     QStringLiteral("点 ⚖️ 后生成"), 0, 0,
           QStringLiteral("① ONNX 基线（本地）"));
    mkCell(imgGrid, &m_imgSimFp16,  QStringLiteral("③ 构建中…"), 0, 1,
           QStringLiteral("② FP16 模拟"));
    mkCell(imgGrid, &m_imgSimInt8,  QStringLiteral("点 ⚖️ 后生成"), 0, 2,
           QStringLiteral("③ int8 模拟"));
    mkCell(imgGrid, &m_imgSimInt16, QStringLiteral("点 ⚖️ 后生成"), 0, 3,
           QStringLiteral("④ int16 模拟"));
    mkCell(imgGrid, &m_imgBdFp16,   QStringLiteral("板离线/未跑"), 1, 0,
           QStringLiteral("⑤ 板上 FP16"));
    mkCell(imgGrid, &m_imgBdInt8,   QStringLiteral("板离线/未跑"), 1, 1,
           QStringLiteral("⑥ 板上 int8"));
    mkCell(imgGrid, &m_imgBdInt16,  QStringLiteral("板离线/未跑"), 1, 2,
           QStringLiteral("⑦ 板上 int16"));
    m_cmpViewStack->addWidget(imgPanel);              // m_cmpViewStack index 1

    // 视图 2：数据面板（七通道页签——与图片面板一一对应：每通道总体指标 + 自己的逐层表）
    auto *dataPanel = new QWidget;
    auto *dv = new QVBoxLayout(dataPanel);
    dv->setContentsMargins(0, 0, 0, 0);
    dv->setSpacing(6);
    auto *chanTabs = new QTabWidget;
    for (int i = 0; i < 7; ++i) {
        auto *page = new QWidget;
        auto *pl = new QVBoxLayout(page);
        pl->setContentsMargins(0, 4, 0, 0);
        pl->setSpacing(4);
        m_chanLabel[i] = new QLabel(QStringLiteral("排队中…"));
        m_chanLabel[i]->setWordWrap(true);
        m_chanLabel[i]->setTextInteractionFlags(Qt::TextSelectableByMouse);
        pl->addWidget(m_chanLabel[i]);
        m_chanTable[i] = new QTableWidget(0, 3);
        m_chanTable[i]->setHorizontalHeaderLabels(
            {QStringLiteral("层"), QStringLiteral("entire cos"), QStringLiteral("single cos")});
        m_chanTable[i]->setAlternatingRowColors(true);
        pl->addWidget(m_chanTable[i], 1);
        chanTabs->addTab(page, chanName(i));
    }
    dv->addWidget(chanTabs, 1);
    m_cmpViewStack->addWidget(dataPanel);             // m_cmpViewStack index 2

    connect(m_btnExport, &QPushButton::clicked, this, &MainWindow::onExportCompare);

    // ---- 信号 ----
    connect(btnRefresh, &QPushButton::clicked, this, &MainWindow::refreshStatus);
    connect(m_btnDeploy, &QPushButton::clicked, this, &MainWindow::onDeploy);
    connect(m_btnRun, &QPushButton::clicked, this, &MainWindow::onRun);
    connect(m_btnPreview, &QPushButton::clicked, this, &MainWindow::onPreviewStart);
    connect(m_btnStop, &QPushButton::clicked, this, &MainWindow::onPreviewStop);

    auto *statusTimer = new QTimer(this);
    connect(statusTimer, &QTimer::timeout, this, &MainWindow::refreshStatus);
    statusTimer->start(5000);

    // ---- 页 3：WORKFLOW 工作流配置器（表单 → pipeline.json → 部署链）----
    // 最后 addWidget：页面栈顺序 LIVE/ONNX/RKNN/WORKFLOW
    m_workflow = new WorkflowPage(resolveProjectsRoot());
    m_views->addWidget(m_workflow);
    connect(m_workflow, &WorkflowPage::projectSaved, this, [this](const QString &name) {
        refreshStatus();                            // 重新扫描 projects/，新工程进下拉
        const int pi = m_projectBox->findText(name);
        if (pi >= 0) m_projectBox->setCurrentIndex(pi);
        appendLog(QStringLiteral("工程已保存并选中: %1（去 LIVE 页可运行）").arg(name), "#3ecf72");
    });
    connect(m_workflow, &WorkflowPage::deployRequested, this, [this](const QString &name) {
        m_views->setCurrentIndex(0);
        m_navLive->setChecked(true);
        autoDeploy(name);
    });
    connect(m_workflow, &WorkflowPage::logLine, this, &MainWindow::appendLog);
}

AdbClient MainWindow::makeAdb() const {
    return AdbClient({}, m_serialBox->currentText() == QStringLiteral("(无设备)")
                                ? QString() : m_serialBox->currentText());
}

namespace {
// 工程根解析：exe 旁 projects/ 优先 → 环境变量 RVS_PROJECTS → 编译期路径
QString resolveProjectsRoot() {
    const QString beside = QCoreApplication::applicationDirPath() + "/projects";
    if (QDir(beside).exists()) return beside;
    const QString env = qEnvironmentVariable("RVS_PROJECTS");
    if (!env.isEmpty() && QDir(env).exists()) return env;
    return QString::fromUtf8(PROJECTS_ROOT);
}
} // namespace

void MainWindow::appendLog(const QString &msg, const QString &color) {
    const QString ts = QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss"));
    m_log->append(QString("<span style='color:#5a6a80'>%1</span> "
                          "<span style='color:%2'>%3</span>").arg(ts, color, msg));
}

void MainWindow::refreshStatus() {
    AdbClient adb;   // 不带 serial：仅枚举
    const QStringList devs = adb.devices();
    const ProjectRegistry reg(resolveProjectsRoot());

    const QString cur = m_serialBox->currentText();
    m_serialBox->clear();
    if (devs.isEmpty()) m_serialBox->addItem(QStringLiteral("(无设备)"));
    else for (const QString &d : devs) m_serialBox->addItem(d);

    const QString curProj = m_projectBox->currentText();
    m_projectBox->clear();
    const QStringList projects = reg.names();
    if (projects.isEmpty()) m_projectBox->addItem(QStringLiteral("(无工程)"));
    else for (const QString &p : projects) m_projectBox->addItem(p);

    // 恢复选择
    const int di = m_serialBox->findText(cur);
    if (di >= 0) m_serialBox->setCurrentIndex(di);
    const int pi = m_projectBox->findText(curProj);
    if (pi >= 0) m_projectBox->setCurrentIndex(pi);

    if (devs.isEmpty()) {
        m_devicePill->setText(QStringLiteral("● NO DEVICE"));
        m_devicePill->setObjectName("PillBad");
    } else {
        m_devicePill->setText(QStringLiteral("● ONLINE ×%1").arg(devs.size()));
        m_devicePill->setObjectName("PillOk");
    }
    m_devicePill->style()->unpolish(m_devicePill);
    m_devicePill->style()->polish(m_devicePill);
    const bool hasBoard = !devs.isEmpty();
    m_btnDeploy->setEnabled(hasBoard);
    m_btnRun->setEnabled(hasBoard);
    m_btnPreview->setEnabled(hasBoard);
}

void MainWindow::onDeploy() {
    m_btnDeploy->setEnabled(false);
    appendLog(QStringLiteral("部署中…"));
    const QString err = Deployer::deploy(makeAdb(), m_projectBox->currentText(),
                                         resolveProjectsRoot(),
                                         QStringLiteral("/data/local/tmp/rvs"));
    if (err.isEmpty()) appendLog(QStringLiteral("部署完成"), "#3ecf72");
    else appendLog(QStringLiteral("部署失败: %1").arg(err), "#ff6b6b");
    m_btnDeploy->setEnabled(true);
}

void MainWindow::onRun() {
    m_btnRun->setEnabled(false);
    appendLog(QStringLiteral("运行中…"));
    AdbClient adb = makeAdb();
    const QString summary = adb.shell(QStringLiteral(
        "cd /data/local/tmp/rvs/%1 && LD_LIBRARY_PATH=/data/local/tmp/rvs "
        "/data/local/tmp/rvs/rvs_engine .").arg(m_projectBox->currentText()));
    appendLog(QStringLiteral("结果: %1").arg(summary), "#3ecf72");
    m_btnRun->setEnabled(true);
}

void MainWindow::onPreviewStart() {
    m_mjpeg->stop();
    m_player->stop();
    m_player->setSource(QUrl());

    // 按 URL scheme 自动路由——杜绝源类型与地址不匹配
    const QString urlText = m_rtspUrl->text().trimmed();
    if (urlText.startsWith(QStringLiteral("rtsp://"))) {
        // RTSP 模式：rvs-cam 推流 → mediamtx 标准流（走网线/WiFi）
        if (urlText.isEmpty()) { appendLog(QStringLiteral("RTSP 地址为空"), "#ff6b6b"); return; }
        m_player->setSource(QUrl(urlText));
        m_player->play();
        appendLog(QStringLiteral("RTSP 预览: %1").arg(urlText), "#3ecf72");
        return;
    }
    if (urlText.startsWith(QStringLiteral("http://"))
            || urlText.startsWith(QStringLiteral("https://"))) {
        // 相机 MJPEG 模式（网络直连）：rvs-cam 内嵌服务 http://<板IP>:8080/stream
        const QUrl url = QUrl::fromUserInput(urlText);
        if (url.host().isEmpty()) {
            appendLog(QStringLiteral("URL 无效（例 http://192.168.1.10:8080/stream）"), "#ff6b6b");
            return;
        }
        m_preview->setPlaceholder(QStringLiteral("CONNECTING…"));
        m_mjpeg->start(url.host(), url.port(80), url.path().isEmpty() ? "/stream" : url.path());
        appendLog(QStringLiteral("MJPEG(网络) 预览: %1").arg(url.toString()), "#3ecf72");
        return;
    }
    // 引擎 MJPEG（USB/forward）：无 scheme 时视为板上引擎流
    AdbClient adb = makeAdb();
    adb.removeForward(kPreviewPort);
    adb.forward(kPreviewPort, QStringLiteral("tcp:8000"));
    m_preview->setPlaceholder(QStringLiteral("CONNECTING…"));
    m_mjpeg->start(QStringLiteral("127.0.0.1"), kPreviewPort, QStringLiteral("/stream"));
    appendLog(QStringLiteral("MJPEG 预览已启动 :%1/stream").arg(kPreviewPort), "#3ecf72");
}

void MainWindow::onPreviewStop() {
    m_player->stop();
    m_player->setSource(QUrl());
    m_mjpeg->stop();
    makeAdb().removeForward(kPreviewPort);
    m_preview->setFrame(QImage());
    m_preview->setPlaceholder(QStringLiteral("已停止"));
    appendLog(QStringLiteral("预览已停止"));
}

void MainWindow::onInfer() {
    if (m_onnx->running()) return;
    const QString model = m_modelPath->text().trimmed();
    const QString image = m_imagePath->text().trimmed();
    if (model.isEmpty() || !QFileInfo::exists(model)) {
        appendLog(QStringLiteral("模型路径无效: %1").arg(model), "#ff6b6b");
        return;
    }
    if (image.isEmpty() || !QFileInfo::exists(image)) {
        appendLog(QStringLiteral("图片路径无效: %1").arg(image), "#ff6b6b");
        return;
    }
    m_btnInfer->setEnabled(false);
    m_inferStats->setText(QStringLiteral("推理中… %1").arg(QFileInfo(model).fileName()));
    appendLog(QStringLiteral("本地推理: %1 ← %2 (conf=%3 iou=%4)")
                  .arg(QFileInfo(model).fileName(), QFileInfo(image).fileName())
                  .arg(m_confSpin->value(), 0, 'f', 2).arg(m_iouSpin->value(), 0, 'f', 2),
              "#8b97a8");
    m_onnx->run(image, model, m_confSpin->value(), m_iouSpin->value());
}

void MainWindow::drawInferResult(const OnnxResult &r) {
    const QImage img = annotateImage(r);
    if (img.isNull()) {
        m_inferStats->setText(QStringLiteral("✗ 无法加载图片 %1").arg(r.imagePath));
        return;
    }
    m_inferPreview->setFrame(img);
    m_inferPreview->setStatus(QStringLiteral("ONNX %1 框").arg(r.detections.size()));
    m_inferStats->setText(QStringLiteral(
        "模型 %1 · 类别数 %2 · 输入 %3×%3 · 图片 %4×%5  |  载入 %6ms · 预处理 %7ms · 推理 %8ms · 后处理 %9ms · 合计 %10ms  |  conf=%11 iou=%12  |  检出 %13 框")
        .arg(r.model).arg(r.numClasses).arg(r.inputSize)
        .arg(r.imageW).arg(r.imageH)
        .arg(r.loadMs, 0, 'f', 0).arg(r.preMs, 0, 'f', 1).arg(r.inferMs, 0, 'f', 0)
        .arg(r.postMs, 0, 'f', 1).arg(r.totalMs, 0, 'f', 0)
        .arg(r.confTh, 0, 'f', 2).arg(r.iouTh, 0, 'f', 2)
        .arg(r.detections.size()));
}

QImage MainWindow::annotateImage(const OnnxResult &r) {
    QImage img(r.imagePath);
    if (img.isNull()) return img;
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing);
    QFont f = p.font();
    f.setPixelSize(qMax(14, img.width() / 90));
    p.setFont(f);
    for (const OnnxDetection &d : r.detections) {
        const QRectF box(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
        p.setPen(QPen(QColor(0x3e, 0xcf, 0x72), qMax(2, img.width() / 500)));
        p.drawRect(box);
        const QString label = r.classNames.isEmpty()
            ? QStringLiteral("cls%1 %2").arg(d.cls).arg(d.conf, 0, 'f', 2)
            : QStringLiteral("%1 %2").arg(r.classNames.value(d.cls,
                                                          QString::number(d.cls))).arg(d.conf, 0, 'f', 2);
        const QPointF tp(d.x1, qMax(f.pixelSize() * 1.1, d.y1 - 4));
        const QFontMetrics fm(f);
        const QRect br = fm.tightBoundingRect(label);
        p.fillRect(QRectF(tp.x() - 3, tp.y() - fm.ascent() - 2, br.width() + 6, fm.height() + 3),
                   QColor(0x3e, 0xcf, 0x72));
        p.setPen(QColor(10, 14, 20));
        p.drawText(tp, label);
    }
    p.end();
    return img;
}

void MainWindow::autoRunInfer(const QString &image, const QString &model) {
    m_views->setCurrentIndex(1);               // 回归：切到模型实验室页
    m_labStack->setCurrentIndex(0);            // 回归：实验室子页=ONNX 基线
    m_segInfer->setChecked(true);
    m_imagePath->setText(image);
    m_modelPath->setText(model);
    onInfer();
}

void MainWindow::onAnalyze() {
    if (m_analyzer->running()) return;
    const QString model = m_modelPath->text().trimmed();
    if (model.isEmpty() || !QFileInfo::exists(model)) {
        appendLog(QStringLiteral("模型路径无效: %1").arg(model), "#ff6b6b");
        return;
    }
    m_btnAnalyze->setEnabled(false);
    appendLog(QStringLiteral("模型分析中: %1（含 shape 推断与逐层 profiling，稍慢）")
                  .arg(QFileInfo(model).fileName()), "#8b97a8");
    m_analyzer->run(model);
}

void MainWindow::showAnalysis(const OnnxAnalysis &a) {
    // 动态字段逐个转义，<br> 等结构标签最后拼，避免被 toHtmlEscaped 打成字面量
    QString inStr, outStr;
    for (const OnnxIo &io : a.inputs)
        inStr += QStringLiteral("%1 %2 [%3]<br>")
                     .arg(io.name.toHtmlEscaped(), io.dtype.toHtmlEscaped(),
                          io.shape.join(QStringLiteral("×")).toHtmlEscaped());
    for (const OnnxIo &io : a.outputs)
        outStr += QStringLiteral("%1 [%2]<br>")
                     .arg(io.name.toHtmlEscaped(),
                          io.shape.join(QStringLiteral("×")).toHtmlEscaped());
    QString ops;
    for (const OnnxOpCount &o : a.opHist)                   // 全量列出，不截断
        ops += QStringLiteral("%1×%2  ").arg(o.op.toHtmlEscaped()).arg(o.count);
    const auto fmtBig = [](double v) {
        if (v >= 1e9) return QStringLiteral("%1 G").arg(v / 1e9, 0, 'f', 2);
        if (v >= 1e6) return QStringLiteral("%1 M").arg(v / 1e6, 0, 'f', 2);
        if (v >= 1e3) return QStringLiteral("%1 K").arg(v / 1e3, 0, 'f', 1);
        return QStringLiteral("%1").arg(v, 0, 'f', 0);
    };
    const QString paramsStr = a.params >= 0 ? fmtBig(a.params) : QStringLiteral("未知");
    const QString macsStr = a.macs >= 0
        ? fmtBig(a.macs) + QStringLiteral("MACs（%1 GFLOPs）").arg(a.macs * 2 / 1e9, 0, 'f', 1)
        : QStringLiteral("未估算");

    QDialog dlg(this);
    dlg.setWindowTitle(QStringLiteral("模型分析 — %1").arg(a.model));
    dlg.resize(780, 520);
    auto *v = new QVBoxLayout(&dlg);
    auto *sum = new QLabel(QStringLiteral(
        "<b>%1</b> · %2 MB<br>"
        "参数量 <b>%3</b> · 计算量 <b>%4</b> · CPU 平均 <b>%5 ms</b>/次（%6 次取均值）<br>"
        "<b>输入</b>：%7<b>输出</b>：%8"
        "<b>算子统计</b>：%9")
        .arg(a.model.toHtmlEscaped()).arg(a.fileMb, 0, 'f', 1).arg(paramsStr, macsStr)
        .arg(a.avgTotalMs, 0, 'f', 1).arg(a.runs)
        .arg(inStr, outStr, ops));
    sum->setWordWrap(true);
    sum->setTextFormat(Qt::RichText);
    v->addWidget(sum);

    auto *table = new QTableWidget(a.topLayers.size(), 3);
    table->setHorizontalHeaderLabels({QStringLiteral("层（节点）"), QStringLiteral("算子"),
                                      QStringLiteral("平均耗时 ms")});
    table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    table->verticalHeader()->setVisible(false);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    for (int i = 0; i < a.topLayers.size(); ++i) {
        const OnnxLayerTime &l = a.topLayers[i];
        table->setItem(i, 0, new QTableWidgetItem(l.name));
        table->setItem(i, 1, new QTableWidgetItem(l.op));
        table->setItem(i, 2, new QTableWidgetItem(QString::number(l.ms, 'f', 3)));
    }
    v->addWidget(table, 1);
    auto *close = new QPushButton(QStringLiteral("关闭"));
    connect(close, &QPushButton::clicked, &dlg, &QDialog::accept);
    v->addWidget(close, 0, Qt::AlignRight);
    dlg.exec();
}

void MainWindow::onProbe() {
    if (m_probe->running()) return;
    const QString onnx = m_modelPath->text().trimmed();
    if (onnx.isEmpty() || !QFileInfo::exists(onnx) || !onnx.endsWith(QStringLiteral(".onnx"), Qt::CaseInsensitive)) {
        QMessageBox::warning(this, QStringLiteral("逐层体检"), QStringLiteral("请先选择有效的 .onnx 模型"));
        return;
    }
    const QString image = m_imagePath->text().trimmed();
    if (image.isEmpty() || !QFileInfo::exists(image)) {
        QMessageBox::warning(this, QStringLiteral("逐层体检"),
                             QStringLiteral("请先选择一张输入图片（体检 = 对这张图跑一次全层推理）"));
        return;
    }
    const QString outDir = QDir::temp().filePath(QStringLiteral("onnx_probe_out"));
    m_btnProbe->setEnabled(false);
    appendLog(QStringLiteral("逐层体检: %1 ← %2（图手术暴露全部中间输出，纯 PC）")
                  .arg(QFileInfo(onnx).fileName(), QFileInfo(image).fileName()), "#8b97a8");
    m_probe->start(QStringLiteral("Ubuntu-22.04"),
                   OnnxRunner::toWslPath(QString::fromUtf8(RVS_TOOLS_DIR) +
                                         QStringLiteral("/onnx_probe.py")),
                   OnnxRunner::toWslPath(onnx),
                   OnnxRunner::toWslPath(image),
                   QStringLiteral("/tmp/onnx_probe_out"));
}

void MainWindow::onRknnSim() {
    if (m_rknn->running()) return;
    const QString model = m_rkModelPath->text().trimmed();
    const QString image = m_rkImagePath->text().trimmed();
    if (model.isEmpty() || !QFileInfo::exists(model)) {
        appendLog(QStringLiteral("模型路径无效: %1").arg(model), "#ff6b6b");
        return;
    }
    if (image.isEmpty() || !QFileInfo::exists(image)) {
        appendLog(QStringLiteral("图片路径无效: %1").arg(image), "#ff6b6b");
        return;
    }
    m_btnRkSim->setEnabled(false);
    appendLog(QStringLiteral("转换导出: %1 ← %2 (conf=%3 iou=%4 norm=%5)%6")
                  .arg(QFileInfo(model).fileName(), QFileInfo(image).fileName())
                  .arg(m_rkConfSpin->value(), 0, 'f', 2).arg(m_rkIouSpin->value(), 0, 'f', 2)
                  .arg(m_rkMeanStd->text())
                  .arg(m_rkSavePath->text().trimmed().isEmpty()
                           ? QStringLiteral("  ⚠ 未填存盘路径，仅模拟自检")
                           : QString()),
                  "#8b97a8");
    m_rknn->run(image, model, m_rkConfSpin->value(), m_rkIouSpin->value(),
                m_rkMeanStd->text().trimmed(), m_rkSavePath->text().trimmed());
}

void MainWindow::autoRunRknnSim(const QString &image, const QString &model,
                                const QString &meanstd) {
    m_views->setCurrentIndex(1);               // 回归：切到模型实验室页
    m_labStack->setCurrentIndex(1);            // 回归：实验室子页=模拟量化
    m_segSim->setChecked(true);
    m_rkImagePath->setText(image);
    m_rkModelPath->setText(model);
    if (!meanstd.isEmpty()) m_rkMeanStd->setText(meanstd);
    onRknnSim();
}

void MainWindow::autoShowWorkflow() {
    m_views->setCurrentIndex(3);               // 截图回归切到 WORKFLOW 页
    m_navWorkflow->setChecked(true);
}

void MainWindow::autoLiveInfer(const QString &model) {
    m_views->setCurrentIndex(0);               // 回 LIVE 页
    m_navLive->setChecked(true);
    m_liveModelPath->setText(model);
    m_ckLiveInfer->setChecked(true);           // 触发 worker 启动
}

void MainWindow::autoDeploy(const QString &project, const QString &serial) {
    fprintf(stderr, "[autoDeploy] project=%s serial=%s\n",
            project.toUtf8().constData(), serial.toUtf8().constData());
    fflush(stderr);
    if (!serial.isEmpty()) {
        const int di = m_serialBox->findText(serial);
        if (di >= 0) m_serialBox->setCurrentIndex(di);
    }
    const int pi = m_projectBox->findText(project);
    if (pi < 0) {
        appendLog(QStringLiteral("部署回归：工程不存在 %1").arg(project), "#ff6b6b");
        return;
    }
    m_projectBox->setCurrentIndex(pi);
    onDeploy();                                  // 与 ①部署到板 按钮同一代码路径
}

void MainWindow::onBoardPerf() {
    if (m_boardPerf->running()) return;
    const QString serial = m_serialBox->currentText();
    if (serial.isEmpty() || serial == QStringLiteral("(无设备)")) {
        appendLog(QStringLiteral("板端性能：无在线设备"), "#ff6b6b");
        return;
    }
    const QString rknn = m_rkSavePath->text().trimmed();
    if (rknn.isEmpty() || !QFileInfo::exists(rknn)) {
        appendLog(QStringLiteral("板端性能：请在「存/导入 .rknn」填入模型路径（自己 build 存盘的，或外部现成的 .rknn）"), "#ff6b6b");
        return;
    }
    const QString image = m_rkImagePath->text().trimmed();
    if (image.isEmpty() || !QFileInfo::exists(image)) {
        appendLog(QStringLiteral("图片路径无效: %1").arg(image), "#ff6b6b");
        return;
    }
    m_btnBoardPerf->setEnabled(false);
    appendLog(QStringLiteral("板端性能测试中（推模型+跑引擎 5 轮）…"), "#8b97a8");
    appendLog(QStringLiteral("板端性能: %1 ← %2 @ %3")
                  .arg(QFileInfo(rknn).fileName(), QFileInfo(image).fileName(), serial), "#8b97a8");
    const QString engineDir = resolveProjectsRoot() + QStringLiteral("/../build-board/engine");
    m_boardPerf->run(serial, rknn, image,
                     QStringLiteral("/data/local/tmp/rvs"), engineDir, 5);
}

void MainWindow::onAccCompare() {
    if (m_compare->running()) return;
    const QString onnx = m_rkModelPath->text().trimmed();
    if (onnx.isEmpty() || !QFileInfo::exists(onnx) || !onnx.endsWith(QStringLiteral(".onnx"), Qt::CaseInsensitive)) {
        QMessageBox::warning(this, QStringLiteral("精度对比"), QStringLiteral("请先选择有效的 .onnx 模型"));
        return;
    }
    const QString image = m_rkImagePath->text().trimmed();
    if (image.isEmpty() || !QFileInfo::exists(image)) {
        QMessageBox::warning(this, QStringLiteral("精度对比"), QStringLiteral("请先选择一张对比图片"));
        return;
    }
    // 校准集：当前=所选一张图（数据闭环后可扩多图）
    const QString dsLocal = QDir::temp().filePath(QStringLiteral("rvs_acc_dataset.txt"));
    {
        QFile ds(dsLocal);
        if (!ds.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::warning(this, QStringLiteral("精度对比"), QStringLiteral("无法写 dataset.txt"));
            return;
        }
        ds.write((OnnxRunner::toWslPath(image) + QStringLiteral("\n")).toUtf8());
    }
    QString meanstd = m_rkMeanStd->text().trimmed();
    if (meanstd.isEmpty() || meanstd.contains(QStringLiteral("none")))
        meanstd = QStringLiteral("0,0,0|255,255,255");
    meanstd.replace(QLatin1Char('|'), QLatin1Char('/'));
    const QString device = m_accDevice->text().trimmed();   // 空则跳过④

    m_btnCompare->setEnabled(false);
    appendLog(QStringLiteral("精度对比: %1（七通道漏斗流式输出，先出 ONNX 基线，约 5~10 分钟）")
                  .arg(QFileInfo(onnx).fileName()), "#8b97a8");
    // 结果就地显示在本页右大区（对比面板），流式填充；默认停在信息面板看操作流
    m_views->setCurrentIndex(1);
    m_navLab->setChecked(true);
    m_labStack->setCurrentIndex(1);
    m_cmpViewStack->setCurrentIndex(0);
    m_cmpLog->clear();
    for (int i = 0; i < 7; ++i) {
        m_chanLabel[i]->setText(i == 0 ? QStringLiteral("推理中…")
                                       : QStringLiteral("排队中…"));
        m_chanTable[i]->setRowCount(0);
    }
    m_imgOnnx->setPlaceholder(QStringLiteral("① ONNX 推理中…"));
    m_imgSimFp16->setPlaceholder(QStringLiteral("② 构建中…"));
    m_imgSimInt8->setPlaceholder(QStringLiteral("③ 排队中…"));
    m_imgSimInt16->setPlaceholder(QStringLiteral("④ 排队中…"));
    m_imgBdFp16->setPlaceholder(QStringLiteral("板离线/未跑"));
    m_imgBdInt8->setPlaceholder(QStringLiteral("板离线/未跑"));
    m_imgBdInt16->setPlaceholder(QStringLiteral("板离线/未跑"));
    // dump 目录给 Windows 路径：WSL python 经 /mnt/c 写入（norm_path 自动转），
    // Qt 直接用原路径读——同一物理目录，两侧一致
    const QString dumpWin = QDir::temp().filePath(QStringLiteral("acc_dump"));
    m_compare->start(QStringLiteral("Ubuntu-22.04"),
                     OnnxRunner::toWslPath(QString::fromUtf8(RVS_TOOLS_DIR) +
                                           QStringLiteral("/acc_compare.py")),
                     OnnxRunner::toWslPath(onnx),
                     OnnxRunner::toWslPath(image),
                     device, meanstd,
                     OnnxRunner::toWslPath(dsLocal),
                     dumpWin);
}

// ---- 七通道辅助：键名/页签名/索引 ----
static const struct { const char *key, *tag; } kChans[7] = {
    {"1_onnx", "① ONNX 基线"},     {"2_sim_fp16", "② FP16 模拟"},
    {"3_sim_int8", "③ int8 模拟"}, {"4_sim_int16", "④ int16 模拟"},
    {"5_board_fp16", "⑤ 板上 FP16"}, {"6_board_int8", "⑥ 板上 int8"},
    {"7_board_int16", "⑦ 板上 int16"},
};
int MainWindow::chanIndex(const QString &stageKey) {
    for (int i = 0; i < 7; ++i)
        if (stageKey == QLatin1String(kChans[i].key)) return i;
    return -1;
}
QString MainWindow::chanName(int i) {
    return (i >= 0 && i < 7) ? QString::fromUtf8(kChans[i].tag) : QString();
}

// 通道总体指标（写进数据面板对应页签的标签）：stages 缺键=排队；skipped/error 如实标注
void MainWindow::refreshChanLabel(int idx, const QJsonObject &o) {
    if (idx < 0 || idx >= 7 || !m_chanLabel[idx]) return;
    const QString k = QString::fromLatin1(kChans[idx].key);
    const QJsonObject st = o.value(QStringLiteral("stages")).toObject().value(k).toObject();
    QString txt;
    if (k == QLatin1String("1_onnx")) {
        txt = st.contains(QStringLiteral("n"))
                  ? QStringLiteral("float32 金标准 · 检出 %1 框\n逐层体检无意义：它就是参考答案")
                        .arg(st.value(QStringLiteral("n")).toInt())
                  : QStringLiteral("推理中…");
    } else if (st.isEmpty()) {
        txt = QStringLiteral("排队中…");
    } else if (st.contains(QStringLiteral("skipped"))) {
        txt = QStringLiteral("未连板，跳过（填板地址后重跑可测）");
    } else if (st.contains(QStringLiteral("error"))) {
        txt = QStringLiteral("失败：%1").arg(st.value(QStringLiteral("error")).toString().left(80));
        if (k == QLatin1String("7_board_int16"))
            txt += QStringLiteral("\n（RK3576 平台断点：Sigmoid/Concat 的 int16 NPU 输出回传不通，"
                                  "已最小复现实证，等固件修复）");
    } else {
        const QString mk = k + QStringLiteral("_vs_1");   // match/cosine 键带 _vs_1 后缀
        const QJsonArray drops = o.value(QStringLiteral("match")).toObject()
                                     .value(mk).toObject()
                                     .value(QStringLiteral("conf_drops")).toArray();
        double drop = 0;
        for (const QJsonValue &v : drops) drop += v.toDouble();
        if (!drops.isEmpty()) drop /= drops.size();
        const QJsonArray a = o.value(QStringLiteral("tensor_cosine")).toObject()
                                 .value(mk).toArray();
        double cos = 0;
        for (const QJsonValue &v : a) cos += v.toDouble();
        if (!a.isEmpty()) cos /= a.size();
        txt = QStringLiteral("检出 %1 框 · mean IoU %2 · conf 掉点 %3 · 张量余弦 %4")
                  .arg(st.value(QStringLiteral("n")).toInt())
                  .arg(o.value(QStringLiteral("match")).toObject().value(mk).toObject()
                           .value(QStringLiteral("mean_iou")).toDouble(), 0, 'f', 4)
                  .arg(drop, 0, 'f', 4)
                  .arg(cos, 0, 'f', 5);
    }
    // 保留 layersUpdate 追加的"逐层:"摘要行
    QString keep = m_chanLabel[idx]->text();
    const qsizetype p = keep.indexOf(QStringLiteral("\n逐层: "));
    if (p >= 0) txt += keep.mid(p);
    m_chanLabel[idx]->setText(txt);
}

// 终局汇总：刷新全部通道标签与逐层表 + 漏斗判读进信息面板；仅 finished 调用
void MainWindow::updateCompareSummary(const QJsonObject &o) {
    for (int i = 0; i < 7; ++i) refreshChanLabel(i, o);
    const QJsonObject layersAll = o.value(QStringLiteral("layers")).toObject();
    for (int i = 0; i < 7; ++i) {
        const QJsonArray arr = layersAll.value(QString::fromLatin1(kChans[i].key)).toArray();
        if (!arr.isEmpty()) fillLayerTable(m_chanTable[i], arr);
    }

    const QJsonObject stages = o.value("stages").toObject();
    auto nOf = [&stages](const QString &k) {
        return stages.value(k).toObject().value("n").toInt(-1);
    };
    QString body;
    for (int i = 1; i < 7; ++i) {
        const QString k = QString::fromLatin1(kChans[i].key);
        const QString tag = chanName(i);
        const QJsonObject st = stages.value(k).toObject();
        if (st.contains(QStringLiteral("skipped"))) {
            body += QStringLiteral("<br>%1 —— 未连板，跳过").arg(tag);
        } else if (st.contains(QStringLiteral("error"))) {
            body += QStringLiteral("<br><span style='color:#ff6b6b'>%1 —— 失败</span>").arg(tag);
        } else {
            const QString mk = k + QStringLiteral("_vs_1");
            const QJsonObject m = o.value("match").toObject().value(mk).toObject();
            const QJsonArray drops = m.value(QStringLiteral("conf_drops")).toArray();
            double drop = 0;
            for (const QJsonValue &v : drops) drop += v.toDouble();
            if (!drops.isEmpty()) drop /= drops.size();
            const QJsonArray a = o.value("tensor_cosine").toObject().value(mk).toArray();
            double cos = 0;
            for (const QJsonValue &v : a) cos += v.toDouble();
            if (!a.isEmpty()) cos /= a.size();
            body += QStringLiteral("<br>%1  检出 %2 框 · mean IoU %3 · conf 掉点 %4 · 张量余弦 %5")
                        .arg(tag, -18).arg(nOf(k))
                        .arg(m.value(QStringLiteral("mean_iou")).toDouble(), 0, 'f', 4)
                        .arg(drop, 0, 'f', 4).arg(cos, 0, 'f', 5);
        }
    }
    body += QStringLiteral("<br><i>判读: 模拟②③④=转换+量化算法损失 · 板⑤⑥⑦再叠加运行时实现差异 · 同 dtype 下 板-模拟≈0 说明运行时无损</i>");
    m_cmpLog->append(QStringLiteral("<b>%1</b> @ %2 · 基准 ① ONNX float32（检出 %3 框）· 全程 %4 s<br>%5")
                              .arg(o.value("model").toString().toHtmlEscaped(),
                                   o.value("image").toString().toHtmlEscaped())
                              .arg(nOf("1_onnx"))
                              .arg(o.value("total_ms").toDouble() / 1000, 0, 'f', 1)
                              .arg(body));
}

// 逐层表填充：三列（层名 + entire/single cos）——首层带 sim_* 键则读 sim，否则读 rt；低余弦标红
void MainWindow::fillLayerTable(QTableWidget *t, const QJsonArray &layers) {
    const bool sim = !layers.isEmpty() &&
                     layers.first().toObject().contains(QStringLiteral("sim_entire_cos"));
    const QString ek = sim ? QStringLiteral("sim_entire_cos")
                           : QStringLiteral("rt_entire_cos");
    const QString sk = sim ? QStringLiteral("sim_single_cos")
                           : QStringLiteral("rt_single_cos");
    t->setRowCount(0);
    int row = 0;
    for (const QJsonValue &v : layers) {
        const QJsonObject o = v.toObject();
        const double ent = o.value(ek).toDouble(-1);
        const double sgl = o.value(sk).toDouble(-1);
        if (ent < 0 && sgl < 0) continue;
        t->insertRow(row);
        auto mk = [](double val) {
            auto *it = new QTableWidgetItem(val > 0 ? QString::number(val, 'f', 5)
                                                    : QStringLiteral("—"));
            if (val > 0 && val < 0.99)
                it->setForeground(QColor(0xff, 0x6b, 0x6b));   // 低于验收线标红
            return it;
        };
        t->setItem(row, 0, new QTableWidgetItem(o.value("name").toString()));
        t->setItem(row, 1, mk(ent));
        t->setItem(row, 2, mk(sgl));
        ++row;
    }
    t->resizeColumnsToContents();
}

// 导出精度对比数据：JSON 原始 + 双表 CSV + 标注图引用清单，存到用户选的目录
void MainWindow::onExportCompare() {
    if (m_lastCompareJson.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("导出报告"),
                             QStringLiteral("还没有精度分析数据——先点「⚖️精度对比」或「🎯量化精度」"));
        return;
    }
    const QString dir = QFileDialog::getExistingDirectory(
        this, QStringLiteral("选择导出目录"),
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    if (dir.isEmpty()) return;

    const QString stamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_hhmmss"));
    QDir().mkpath(dir);

    // 1) JSON 原始数据
    const QString jsonPath = dir + QStringLiteral("/acc_report_%1.json").arg(stamp);
    QFile jf(jsonPath);
    jf.open(QIODevice::WriteOnly);
    jf.write(m_lastCompareJson.toUtf8());
    jf.close();

    // 2) 双表 CSV（逗号分隔，Excel 直开）
    const QJsonDocument doc = QJsonDocument::fromJson(m_lastCompareJson.toUtf8());
    const QJsonArray layers = doc.object().value("layers").toArray();
    auto dumpCsv = [&dir, &stamp](QTableWidget *t, const QString &name) {
        QFile f(dir + QStringLiteral("/%1_%2.csv").arg(name, stamp));
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) return;
        QTextStream ts(&f);
        ts.setGenerateByteOrderMark(true);           // Excel 中文兼容
        for (int c = 0; c < t->columnCount(); ++c)
            ts << t->horizontalHeaderItem(c)->text() << (c < t->columnCount() - 1 ? "," : "\n");
        for (int r = 0; r < t->rowCount(); ++r) {
            for (int c = 0; c < t->columnCount(); ++c) {
                QTableWidgetItem *it = t->item(r, c);
                QString v = it ? it->text() : QString();
                v.replace(QLatin1Char(','), QLatin1Char(' '));
                ts << v << (c < t->columnCount() - 1 ? "," : "\n");
            }
        }
    };
    for (int i = 0; i < 7; ++i)
        dumpCsv(m_chanTable[i], QStringLiteral("layers_%1").arg(
                    QString::fromLatin1(kChans[i].key)));

    // 3) 标注图引用清单（存在则写路径）
    QStringList refs;
    if (!m_cmpDumpDir.isEmpty()) {
        for (const QString &img : {QStringLiteral("onnx_result.jpg"),
                                   QStringLiteral("int8sim_result.jpg"),
                                   QStringLiteral("board_result.jpg")}) {
            const QString p = m_cmpDumpDir + QStringLiteral("/") + img;
            if (QFileInfo::exists(p)) refs << p;
        }
    }
    QFile rf(dir + QStringLiteral("/images_%1.txt").arg(stamp));
    if (rf.open(QIODevice::WriteOnly | QIODevice::Text)) {
        for (const QString &p : refs) rf.write((p + QStringLiteral("\n")).toUtf8());
    }

    appendLog(QStringLiteral("精度报告已导出: %1（JSON+CSV+%2 张标注图引用）").arg(dir).arg(refs.size()),
              "#3ecf72");
    QMessageBox::information(this, QStringLiteral("导出报告"),
                             QStringLiteral("已导出到:\n%1\n\nacc_report_*.json（原始数据）\n"
                                            "sim_layers_*.csv / runtime_layers_*.csv（逐层表）\n"
                                            "images_*.txt（标注图路径，共 %2 张）")
                                 .arg(dir).arg(refs.size()));
}

void MainWindow::onNetronView() {
    if (m_netronProc) return;
    const QString model = m_modelPath->text().trimmed();
    if (model.isEmpty() || !QFileInfo::exists(model)) {
        appendLog(QStringLiteral("模型路径无效: %1").arg(model), "#ff6b6b");
        return;
    }
    m_btnNetron->setEnabled(false);
    appendLog(QStringLiteral("Netron 启动中: %1").arg(QFileInfo(model).fileName()), "#8b97a8");

    // 杀旧实例（[n] 技巧避免匹配到本 shell）→ 起服务 → 轮询 HTTP 就绪
    const QString script = QStringLiteral(
        "pkill -f '[n]etron --host' >/dev/null 2>&1; sleep 0.4; "
        "nohup netron --host 127.0.0.1 --port 8086 '%1' >/tmp/netron.log 2>&1 & "
        "code=000; for i in $(seq 1 16); do sleep 0.5; "
        "code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8086 2>/dev/null); "
        "[ \"$code\" = \"200\" ] && break; done; echo NETRON:$code")
        .arg(OnnxRunner::toWslPath(model));

    m_netronProc = new QProcess(this);
    connect(m_netronProc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this](int, QProcess::ExitStatus) {
        const QString out = QString::fromUtf8(m_netronProc->readAllStandardOutput());
        m_netronProc->deleteLater();
        m_netronProc = nullptr;
        m_btnNetron->setEnabled(true);
        if (out.contains(QLatin1String("NETRON:200"))) {
            appendLog(QStringLiteral("Netron 已就绪: http://127.0.0.1:8086"), "#3ecf72");
            QDesktopServices::openUrl(QUrl(QStringLiteral("http://127.0.0.1:8086")));
        } else {
            appendLog(QStringLiteral("Netron 启动失败（WSL 内执行: pip install netron 后重试）"),
                      "#ff6b6b");
        }
    });
    m_netronProc->start(QStringLiteral("wsl.exe"),
                        {QStringLiteral("-d"), QStringLiteral(RVS_WSL_DISTRO),
                         QStringLiteral("--"), QStringLiteral("bash"), QStringLiteral("-lc"),
                         script});
    QTimer::singleShot(30000, this, [this] {
        if (m_netronProc) m_netronProc->kill();
    });
}

void MainWindow::autoStartRtsp(const QString &url) {
    m_views->setCurrentIndex(0);               // 预览回归回到 LIVE 页
    m_navLive->setChecked(true);
    m_sourceBox->setCurrentIndex(1);
    m_rtspUrl->setText(url);
    onPreviewStart();
}
