#pragma once
// 工程扫描器：projects/ 目录 → 可用工程列表（含 pipeline.json 校验）
#include <QString>
#include <QStringList>

class ProjectRegistry {
public:
    explicit ProjectRegistry(const QString &projectsRoot);

    QStringList names() const;                        // 有效工程名列表（按 pipeline.json）
    QString projectPath(const QString &name) const;   // 工程目录绝对路径
    QString pipelineJsonPath(const QString &name) const;

private:
    QString m_root;
};
