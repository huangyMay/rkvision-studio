// 工作流表单 → pipeline.json 生成器（model 层纯逻辑，无 QWidget；UI 只做表单采集）。
// 字段名与引擎插件逐字对齐：image_dir{path,loop}/v4l2{device,width,height,prefer_mjpeg}/
// letterbox{size,pad_value}/{yolov5_det,yolov8_det}{conf_thres,iou_thres,classes}/
// judge{rules,verdict,actions}/archive{ng_dir,keep_days}/image_out{path,draw,save_json}/mjpeg{port,...}
#pragma once
#include <QString>
#include <QList>
#include <QJsonObject>

struct WorkflowRule {
    QString type;       // count|score|size|region|persist|regex
    QString cls = "*";
    QString param1;     // count:min  score:thres  size:min_px  region:"x,y;x,y;..."  persist:frames  regex:pattern
    QString param2;     // count:max  size:max_px  region:inside("inside"/"outside")
};

struct WorkflowForm {
    QString projectName;

    bool sourceV4l2 = false;
    QString sourcePath = QStringLiteral("assets/test_images");
    QString device = QStringLiteral("/dev/video0");
    int width = 1280, height = 720;

    int size = 640, padValue = 255;

    QString modelFile;                      // 任意路径；JSON 只写 models/<basename>
    QString family = QStringLiteral("yolov5-det");   // yolov5-det | yolov8-det
    QString classes = QStringLiteral("head,helmet");
    double conf = 0.25, iou = 0.45;

    QList<WorkflowRule> rules;
    bool ngArchive = true;
    QString ngDir = QStringLiteral("ng_images");

    bool imageOut = true;
    bool mjpeg = false;
    int mjpegPort = 8000;
};

namespace WorkflowGen {

// 纯函数；规则行非法（未知类型/region 点数<3/数值字段非数）抛 QString（what 直接可展示）
QJsonObject buildJson(const WorkflowForm &f);

// 建目录 + 复制模型进 <工程>/models/ + 写 pipeline.json；失败返回空串并写 *err
QString save(const WorkflowForm &f, const QString &projectsRoot, QString *err);

} // namespace WorkflowGen
