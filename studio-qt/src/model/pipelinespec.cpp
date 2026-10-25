#include "pipelinespec.h"
#include <QJsonArray>
#include <QJsonDocument>
#include <QSet>

// 校验规则与 engine/core/spec.cpp 对齐：同一份 pipeline.json 在 PC 侧能通过，
// 就不该在板上才发现字段缺失 / 引用悬空。
PipelineSpec PipelineSpec::parse(const QString &jsonText, QString *err) {
    auto fail = [err](const QString &m) {
        if (err) *err = m;
        return PipelineSpec{};
    };
    QJsonParseError pe;
    const QJsonDocument doc = QJsonDocument::fromJson(jsonText.toUtf8(), &pe);
    if (doc.isNull()) return fail(QStringLiteral("JSON 解析失败: %1").arg(pe.errorString()));
    if (!doc.isObject()) return fail(QStringLiteral("顶层必须是 JSON 对象"));
    const QJsonObject root = doc.object();

    if (!root.contains("version")) return fail(QStringLiteral("缺少字段: version"));
    if (!root.value("version").isDouble()) return fail(QStringLiteral("version 必须是整数"));
    if (root.value("version").toInt() != 1)
        return fail(QStringLiteral("不支持的 version: %1（当前只支持 1）")
                        .arg(root.value("version").toInt()));
    if (root.contains("name") && !root.value("name").isString())
        return fail(QStringLiteral("name 必须是字符串"));

    PipelineSpec spec;
    spec.version = 1;
    spec.name = root.value("name").toString();
    if (root.value("target").toObject().contains("core_mask"))
        spec.coreMask = root.value("target").toObject().value("core_mask").toString("auto");

    if (root.contains("models")) {
        if (!root.value("models").isObject())
            return fail(QStringLiteral("models 必须是对象"));
        const QJsonObject models = root.value("models").toObject();
        for (auto it = models.begin(); it != models.end(); ++it) {
            if (!it.value().isObject())
                return fail(QStringLiteral("models.%1 必须是对象").arg(it.key()));
            const QJsonObject mo = it.value().toObject();
            if (!mo.contains("family"))
                return fail(QStringLiteral("models.%1 缺少字段: family").arg(it.key()));
            if (!mo.contains("file"))
                return fail(QStringLiteral("models.%1 缺少字段: file").arg(it.key()));
            if (!mo.value("family").isString() || !mo.value("file").isString())
                return fail(QStringLiteral("models.%1 的 family/file 必须是字符串").arg(it.key()));
            ModelSpec m;
            m.family = mo.value("family").toString();
            m.file = mo.value("file").toString();
            spec.models.insert(it.key(), m);
        }
    }

    if (!root.contains("pipeline")) return fail(QStringLiteral("缺少字段: pipeline"));
    if (!root.value("pipeline").isArray()) return fail(QStringLiteral("pipeline 必须是数组"));

    QSet<QString> seenIds;
    const QJsonArray pipeline = root.value("pipeline").toArray();
    for (const QJsonValue &v : pipeline) {
        if (!v.isObject()) return fail(QStringLiteral("pipeline 的每个 stage 必须是对象"));
        const QJsonObject s = v.toObject();
        if (!s.contains("id")) return fail(QStringLiteral("stage 缺少字段: id"));
        if (!s.contains("plugin"))
            return fail(QStringLiteral("stage '%1' 缺少字段: plugin").arg(s.value("id").toString()));
        if (!s.value("id").isString() || !s.value("plugin").isString())
            return fail(QStringLiteral("stage 的 id/plugin 必须是字符串"));
        if (s.contains("params") && !s.value("params").isObject())
            return fail(QStringLiteral("stage '%1' 的 params 必须是对象").arg(s.value("id").toString()));

        StageSpec ss;
        ss.id = s.value("id").toString();
        if (ss.id.isEmpty()) return fail(QStringLiteral("stage 的 id 不能为空"));
        if (seenIds.contains(ss.id))
            return fail(QStringLiteral("重复的 stage id: %1").arg(ss.id));
        seenIds.insert(ss.id);
        ss.plugin = s.value("plugin").toString();
        if (ss.plugin.isEmpty())
            return fail(QStringLiteral("stage '%1' 的 plugin 不能为空").arg(ss.id));
        if (s.contains("model")) {
            if (!s.value("model").isString())
                return fail(QStringLiteral("stage '%1' 的 model 必须是字符串").arg(ss.id));
            ss.model = s.value("model").toString();
            if (!spec.models.contains(ss.model))
                return fail(QStringLiteral("stage '%1' 引用了不存在的 model: %2")
                                .arg(ss.id, ss.model));
        }
        ss.params = s.value("params").toObject();
        spec.pipeline.push_back(ss);
    }
    return spec;
}
