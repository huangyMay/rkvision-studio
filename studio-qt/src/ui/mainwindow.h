#pragma once
// 主窗口：组装各面板、连接 core 服务层信号槽
#include <QMainWindow>

class AdbClient;
class BoardMonitor;
class BoardPerfRunner;
class InferStream;
class MjpegClient;
class OnnxAnalyzer;
class OnnxRunner;
class OnnxProbeRunner;
class PreviewWidget;
class RknnSimRunner;
class AccCompareRunner;
class WorkflowPage;
class PerfBars;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFrame;
class QLabel;
class QLineEdit;
class QMediaPlayer;
class QPushButton;
class QSpinBox;
class QStackedWidget;
class QTableWidget;
class QTextEdit;
class QVideoSink;

struct OnnxResult;
struct OnnxAnalysis;
struct BoardPerf;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void refreshStatus();
    void onDeploy();
    void onRun();
    void onPreviewStart();
    void onPreviewStop();
    void onInfer();
    void onAnalyze();
    void onProbe();
    void onNetronView();
    void onRknnSim();
    void onBoardPerf();
    void onAccCompare();
    void onHybridPropose();                     // 🧬 向导 step1：toolkit 敏感层提议
    void onHybridBuild();                       // 🧬 向导 step2：勾选层构建+板复测
    void onDissect();                           // 🔪 rknn 文件解剖（布局/任务数/差分）
    void onPerfGo();                            // 🚀 性能剖析（mask 矩阵+定频+分段计时）
    void onExportCompare();
    void fillLayerTable(QTableWidget *t, const QJsonArray &layers);  // 填充逐层表（5 列 sim|rt）
    void updateCompareSummary(const QJsonObject &o);  // 对比视图漏斗摘要（流式渐进刷新）
    void appendLog(const QString &msg, const QString &color = {});

public slots:
    void autoStartRtsp(const QString &url);   // 截图回归/一键演示用
    void autoRunInfer(const QString &image, const QString &model); // 截图回归用
    void autoRunRknnSim(const QString &image, const QString &model,
                        const QString &meanstd);                   // 截图回归用
    void autoLiveInfer(const QString &model);                      // LIVE 实时推理回归
    void autoDeploy(const QString &project, const QString &serial = {}); // 部署回归
    void autoShowWorkflow();                                       // 截图回归：切 WORKFLOW 页
    void autoShowPerfLab();                                        // 截图回归：切 🚀 性能剖析段

private:
    void buildUi();
    QImage annotateImage(const OnnxResult &r);   // 载入图片并画检测框（两个 LAB 共用）
    void drawInferResult(const OnnxResult &r);
    void showAnalysis(const OnnxAnalysis &a);
    AdbClient makeAdb() const;

    // 顶栏
    QLabel *m_devicePill = nullptr;
    QComboBox *m_serialBox = nullptr;
    // 左列
    QComboBox *m_projectBox = nullptr;
    QComboBox *m_sourceBox = nullptr;
    QLineEdit *m_rtspUrl = nullptr;
    QPushButton *m_btnDeploy = nullptr;
    QPushButton *m_btnRun = nullptr;
    QPushButton *m_btnPreview = nullptr;
    QPushButton *m_btnStop = nullptr;
    // 本地 ONNX 验证卡
    QLineEdit *m_modelPath = nullptr;
    QLineEdit *m_imagePath = nullptr;
    QDoubleSpinBox *m_confSpin = nullptr;      // 置信度阈值
    QDoubleSpinBox *m_iouSpin = nullptr;       // NMS IoU 阈值
    QPushButton *m_btnInfer = nullptr;
    QPushButton *m_btnAnalyze = nullptr;       // 📊 模型分析
    QPushButton *m_btnProbe = nullptr;         // 🧪 逐层体检（量化友好度）
    QPushButton *m_btnNetron = nullptr;        // 🔍 Netron 查看
    OnnxRunner *m_onnx = nullptr;
    OnnxAnalyzer *m_analyzer = nullptr;
    OnnxProbeRunner *m_probe = nullptr;
    class QProcess *m_netronProc = nullptr;
    // 右主区（LIVE 页）
    PreviewWidget *m_preview = nullptr;
    QVideoSink *m_sink = nullptr;              // RTSP 模式解码帧出口
    QMediaPlayer *m_player = nullptr;
    QLabel *m_boardStats = nullptr;            // 板端 CPU/内存/NPU/温度 状态条
    BoardMonitor *m_bmon = nullptr;
    // LIVE 左列推理信息卡
    QTextEdit *m_inferInfo = nullptr;
    // ONNX LAB 页
    QStackedWidget *m_views = nullptr;         // 页面栈：0=LIVE 1=模型实验室 2=WORKFLOW
    QStackedWidget *m_labStack = nullptr;      // 实验室内部子页栈：0=ONNX 基线 1=模拟量化
    QPushButton *m_navLive = nullptr;          // 导航页签（与页面栈双向同步）
    QPushButton *m_navLab = nullptr;           // 模型实验室页签（合并 ONNX LAB + RKNN LAB）
    QPushButton *m_segInfer = nullptr;         // 实验室内部段：ONNX 基线
    QPushButton *m_segSim = nullptr;           // 实验室内部段：模拟量化
    PreviewWidget *m_imgOnnx = nullptr;           // 图片面板：① ONNX 基线
    PreviewWidget *m_imgSimFp16 = nullptr;        // 图片面板：② FP16 模拟
    PreviewWidget *m_imgSimInt8 = nullptr;        // 图片面板：③ int8 模拟
    PreviewWidget *m_imgSimInt16 = nullptr;       // 图片面板：④ int16 模拟
    PreviewWidget *m_imgBdFp16 = nullptr;         // 图片面板：⑤ 板上 FP16
    PreviewWidget *m_imgBdInt8 = nullptr;         // 图片面板：⑥ 板上 int8
    PreviewWidget *m_imgBdInt16 = nullptr;        // 图片面板：⑦ 板上 int16
    QStackedWidget *m_cmpViewStack = nullptr;     // 对比面板：0=信息面板 1=图片面板 2=数据面板
    QTextEdit *m_cmpLog = nullptr;                // 信息面板：操作流日志
    QLabel *m_chanLabel[7] = {};                  // 数据面板：各通道总体指标标签
    QTableWidget *m_chanTable[7] = {};            // 数据面板：各通道逐层表
    // 🧬 混合量化向导（对比面板第 4 页签）
    QTableWidget *m_hqTable = nullptr;            // 敏感层勾选表（勾选|层名|相似度分）
    QPushButton *m_hqBtnGo = nullptr;             // 开工（step2 构建+板复测）
    QLabel *m_hqResult = nullptr;                 // 结果（IoU 对比 + .rknn 路径）
    QString m_hqCfg;                              // step1 产出的 cfg（WSL 路径）
    int m_hqPhase = 0;                            // 0=propose 1=build（finished 分流）
    QPushButton *m_btnExport = nullptr;        // 💾 导出报告
    QString m_lastCompareJson;                 // 最近一次精度分析的原始 JSON（导出用）
    QString m_cmpDumpDir;                      // 最近一次对比的标注图目录
    int chanIndex(const QString &stageKey);    // 通道键 → 0..6；未知 -1
    QString chanName(int i);                   // 0..6 → 页签名
    void refreshChanLabel(int idx, const QJsonObject &o);  // 通道总体指标 → 页签标签
    QPushButton *m_navWorkflow = nullptr;
    WorkflowPage *m_workflow = nullptr;        // 工作流配置器页
    PreviewWidget *m_inferPreview = nullptr;   // 推理结果大屏
    QLabel *m_inferStats = nullptr;            // 推理细节条
    // RKNN LAB 页（toolkit2 模拟器）
    QLineEdit *m_rkModelPath = nullptr;        // .onnx（模拟输入是 ONNX+转换配置）
    QLineEdit *m_rkImagePath = nullptr;
    QLineEdit *m_rkMeanStd = nullptr;          // "0,0,0|255,255,255" / "none|none"
    QLineEdit *m_rkSavePath = nullptr;         // 可选：build 产出的 .rknn 保存路径
    QDoubleSpinBox *m_rkConfSpin = nullptr;
    QDoubleSpinBox *m_rkIouSpin = nullptr;
    QPushButton *m_btnRkSim = nullptr;
    QPushButton *m_btnCompare = nullptr;       // ⚖️ 七通道精度对比
    QPushButton *m_btnHq = nullptr;            // 🧬 混合量化向导（propose 入口）
    QPushButton *m_btnBoardPerf = nullptr;     // 📡 板端 NPU 性能
    QLineEdit *m_accDevice = nullptr;          // 网络 adb 地址（IP:端口，⚖️/📡 共用）
    QPushButton *m_btnDissect = nullptr;       // 🔪 rknn 文件解剖（用「存/导入」的 .rknn）
    AccCompareRunner *m_dissectRunner = nullptr;
    // 🚀 性能剖析段（实验室第 3 段）
    QPushButton *m_segPerf = nullptr;          // 段切换按钮
    QLineEdit *m_perfModelPath = nullptr;      // 被剖析的 .rknn
    QComboBox *m_perfTarget = nullptr;         // adb 序列/IP:端口 或 root@IP(ssh)
    QComboBox *m_perfMask = nullptr;           // mask 组合预设
    QSpinBox *m_perfLoops = nullptr;           // 计时帧数
    QCheckBox *m_perfFixFreq = nullptr;        // 锁 NPU/DDR 频率
    QPushButton *m_btnPerfGo = nullptr;
    QTextEdit *m_perfLog = nullptr;            // 进度日志
    QTableWidget *m_perfTable = nullptr;       // 结果表（mask×set/run/get/e2e/fps）
    PerfBars *m_perfBars = nullptr;            // 瀑布图
    AccCompareRunner *m_perfRunner = nullptr;
    AccCompareRunner *m_compare = nullptr;
    AccCompareRunner *m_hqRunner = nullptr;    // 🧬 向导专用（两段复用）
    RknnSimRunner *m_rknn = nullptr;
    BoardPerfRunner *m_boardPerf = nullptr;
    // LIVE 页 PC 实时推理
    QCheckBox *m_ckLiveInfer = nullptr;        // ⚡ PC 实时推理开关
    QLineEdit *m_liveModelPath = nullptr;      // ONNX 模型（预览流叠加用）
    InferStream *m_istream = nullptr;
    // 日志
    QTextEdit *m_log = nullptr;

    MjpegClient *m_mjpeg = nullptr;
    int m_frames = 0;
    qint64 m_lastFpsMs = 0;
    double m_liveInferMs = 0;                  // 最近一次 PC 实时推理耗时（状态行显示用）
};
