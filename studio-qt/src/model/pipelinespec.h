#pragma once
// pipeline.json 的 Qt 侧域模型 —— 与板端 rvs_engine 的 nlohmann 结构同构
#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QVector>

struct StageSpec {
    QString id;
    QString plugin;
    QString model;          // 仅 infer 类：models 键名
    QJsonObject params;
};

struct ModelSpec {
    QString family;
    QString file;
};

struct PipelineSpec {
    int version = 0;
    QString name;
    QString coreMask = QStringLiteral("auto");
    QMap<QString, ModelSpec> models;
    QVector<StageSpec> pipeline;

    // 解析失败返回 {} 并填充 err
    static PipelineSpec parse(const QString &jsonText, QString *err);
    bool isValid() const { return version == 1; }
};
