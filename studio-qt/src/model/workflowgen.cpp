#include "model/workflowgen.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>

namespace {

QJsonObject sourceNode(const WorkflowForm &f) {
    if (f.sourceV4l2) {
        return QJsonObject{{"id", QStringLiteral("src")},
                           {"plugin", QStringLiteral("v4l2")},
                           {"params", QJsonObject{{"device", f.device},
                                                  {"width", f.width},
                                                  {"height", f.height},
                                                  {"prefer_mjpeg", true}}}};
    }
    return QJsonObject{{"id", QStringLiteral("src")},
                       {"plugin", QStringLiteral("image_dir")},
                       {"params", QJsonObject{{"path", f.sourcePath},
                                              {"loop", false}}}};
}

QString doubleText(const QString &s, const char *what) {   // 数值字段校验
    bool ok = false;
    const double v = s.toDouble(&ok);
    if (!ok) throw QObject::tr("规则参数不是数字: %1 (%2)").arg(s, QString::fromLatin1(what));
    return QString::number(v);
}

QJsonObject ruleToJson(const WorkflowRule &r) {
    QJsonObject o{{"type", r.type}, {"class", r.cls.isEmpty() ? QStringLiteral("*") : r.cls}};
    if (r.type == QLatin1String("count")) {
        o["min"] = doubleText(r.param1, "min").toInt();
        o["max"] = doubleText(r.param2, "max").toInt();
    } else if (r.type == QLatin1String("score")) {
        o["thres"] = doubleText(r.param1, "thres").toDouble();
    } else if (r.type == QLatin1String("size")) {
        o["min_px"] = doubleText(r.param1, "min_px").toDouble();
        o["max_px"] = doubleText(r.param2, "max_px").toDouble();
    } else if (r.type == QLatin1String("region")) {
        const QStringList pts = r.param1.split(QLatin1Char(';'), Qt::SkipEmptyParts);
        if (pts.size() < 3)
            throw QObject::tr("region 规则需要至少 3 个点（格式 x,y;x,y;x,y）: %1").arg(r.param1);
        QJsonArray poly;
        for (const QString &p : pts) {
            const QStringList xy = p.split(QLatin1Char(','));
            if (xy.size() != 2)
                throw QObject::tr("region 点格式应为 x,y: %1").arg(p);
            poly.append(QJsonArray{xy[0].trimmed().toDouble(), xy[1].trimmed().toDouble()});
        }
        o["polygon"] = poly;
        o["inside"] = (r.param2.trimmed().isEmpty() || r.param2 == QLatin1String("inside"));
    } else if (r.type == QLatin1String("persist")) {
        o["frames"] = doubleText(r.param1, "frames").toInt();
    } else if (r.type == QLatin1String("regex")) {
        o["pattern"] = r.param1;
    } else {
        throw QObject::tr("未知规则类型: %1").arg(r.type);
    }
    return o;
}

} // namespace

QJsonObject WorkflowGen::buildJson(const WorkflowForm &f) {
    const QString plugin = (f.family == QLatin1String("yolov8-det"))
                               ? QStringLiteral("yolov8_det") : QStringLiteral("yolov5_det");
    QJsonArray rules;
    for (const WorkflowRule &r : f.rules)
        rules.append(ruleToJson(r));

    QJsonArray pipeline;
    pipeline.append(sourceNode(f));
    pipeline.append(QJsonObject{{"id", QStringLiteral("pre")},
                                {"plugin", QStringLiteral("letterbox")},
                                {"params", QJsonObject{{"size", f.size},
                                                       {"pad_value", f.padValue}}}});
    pipeline.append(QJsonObject{{"id", QStringLiteral("inf")},
                                {"plugin", QStringLiteral("infer")},
                                {"model", QStringLiteral("det")}});
    QJsonObject postParams{{"conf_thres", f.conf}, {"iou_thres", f.iou}};
    if (!f.classes.trimmed().isEmpty()) {
        QJsonArray cls;
        for (const QString &c : f.classes.split(QLatin1Char(','), Qt::SkipEmptyParts))
            cls.append(c.trimmed());
        postParams["classes"] = cls;
    }
    pipeline.append(QJsonObject{{"id", QStringLiteral("post")},
                                {"plugin", plugin},
                                {"params", postParams}});
    if (!rules.isEmpty()) {
        pipeline.append(QJsonObject{{"id", QStringLiteral("jdg")},
                                    {"plugin", QStringLiteral("judge")},
                                    {"params", QJsonObject{{"rules", rules},
                                                           {"verdict", QStringLiteral("ANY_NG")},
                                                           {"actions", QJsonArray{QStringLiteral("log:") + f.projectName,
                                                                                  QStringLiteral("snapshot:") + f.ngDir}}}}});
    }
    if (f.ngArchive) {
        pipeline.append(QJsonObject{{"id", QStringLiteral("arc")},
                                    {"plugin", QStringLiteral("archive")},
                                    {"params", QJsonObject{{"ng_dir", f.ngDir},
                                                           {"keep_days", 7}}}});
    }
    if (f.imageOut) {
        pipeline.append(QJsonObject{{"id", QStringLiteral("out")},
                                    {"plugin", QStringLiteral("image_out")},
                                    {"params", QJsonObject{{"path", QStringLiteral("output")},
                                                           {"draw", true},
                                                           {"save_json", true}}}});
    }
    if (f.mjpeg) {
        pipeline.append(QJsonObject{{"id", QStringLiteral("mjp")},
                                    {"plugin", QStringLiteral("mjpeg")},
                                    {"params", QJsonObject{{"port", f.mjpegPort},
                                                           {"path", QStringLiteral("/stream")},
                                                           {"draw", true},
                                                           {"fps", 10}}}});
    }

    const QString base = QFileInfo(f.modelFile).fileName();
    if (base.isEmpty())
        throw QObject::tr("模型文件为空，无法生成 models/<basename>");
    return QJsonObject{
        {"version", 1},
        {"name", f.projectName},
        {"target", QJsonObject{{"platform", QStringLiteral("rk3576")},
                               {"runtime", QStringLiteral("android-native")},
                               {"core_mask", QStringLiteral("auto")}}},
        {"models", QJsonObject{{"det", QJsonObject{
            {"family", f.family},
            {"file", QStringLiteral("models/") + base},
            {"input", QJsonObject{{"name", QString()},
                                  {"shape", QJsonArray{1, 3, f.size, f.size}},
                                  {"layout", QStringLiteral("NHWC")},
                                  {"dtype", QStringLiteral("uint8")}}}}}}},
        {"pipeline", pipeline}};
}

QString WorkflowGen::save(const WorkflowForm &f, const QString &projectsRoot, QString *err) {
    try {
        if (f.projectName.trimmed().isEmpty() || f.projectName.contains(QLatin1Char('/'))
                || f.projectName.contains(QLatin1Char('\\')) || f.projectName.contains(QLatin1Char(' '))
                || f.projectName.contains(QLatin1Char('..'))) {
            if (err) *err = QObject::tr("工程名非法（不能为空/含空格 / \\ ..）");
            return {};
        }
        if (f.modelFile.isEmpty() || !QFileInfo::exists(f.modelFile)) {
            if (err) *err = QObject::tr("模型文件不存在，拒绝生成半截工程");
            return {};
        }
        const QString dir = projectsRoot + QLatin1Char('/') + f.projectName;
        if (!QDir().mkpath(dir + QStringLiteral("/models"))) {
            if (err) *err = QObject::tr("无法创建目录: %1").arg(dir);
            return {};
        }
        const QString dst = dir + QStringLiteral("/models/") + QFileInfo(f.modelFile).fileName();
        if (!QFileInfo::exists(dst) && !QFile::copy(f.modelFile, dst)) {
            if (err) *err = QObject::tr("复制模型失败: %1 -> %2").arg(f.modelFile, dst);
            return {};
        }
        QFile jf(dir + QStringLiteral("/pipeline.json"));
        if (!jf.open(QIODevice::WriteOnly)) {
            if (err) *err = QObject::tr("无法写 pipeline.json");
            return {};
        }
        jf.write(QJsonDocument(buildJson(f)).toJson());
        return dir;
    } catch (const QString &msg) {
        if (err) *err = msg;
        return {};
    }
}
