#include "pipelinespec.h"
#include <QJsonArray>
#include <QJsonDocument>

PipelineSpec PipelineSpec::parse(const QString &jsonText, QString *err) {
    auto fail = [err](const QString &m) {
        if (err) *err = m;
        return PipelineSpec{};
    };
    QJsonParseError pe;
    const QJsonDocument doc = QJsonDocument::fromJson(jsonText.toUtf8(), &pe);
    if (doc.isNull()) return fail(QStringLiteral("JSON 解析失败: %1").arg(pe.errorString()));
    const QJsonObject root = doc.object();
    if (!root.contains("version")) return fail(QStringLiteral("缺少字段: version"));
    if (root.value("version").toInt() != 1) return fail(QStringLiteral("不支持的 version"));

    PipelineSpec spec;
    spec.version = 1;
    spec.name = root.value("name").toString();
    if (root.value("target").toObject().contains("core_mask"))
        spec.coreMask = root.value("target").toObject().value("core_mask").toString("auto");

    const QJsonObject models = root.value("models").toObject();
    for (auto it = models.begin(); it != models.end(); ++it) {
        ModelSpec m;
        m.family = it.value().toObject().value("family").toString();
        m.file = it.value().toObject().value("file").toString();
        spec.models.insert(it.key(), m);
    }

    const QJsonArray pipeline = root.value("pipeline").toArray();
    for (const QJsonValue &v : pipeline) {
        const QJsonObject s = v.toObject();
        StageSpec ss;
        ss.id = s.value("id").toString();
        ss.plugin = s.value("plugin").toString();
        if (s.contains("model")) ss.model = s.value("model").toString();
        ss.params = s.value("params").toObject();
        spec.pipeline.push_back(ss);
    }
    return spec;
}
