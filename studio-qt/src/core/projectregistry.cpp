#include "projectregistry.h"
#include <QDir>
#include <QFile>

ProjectRegistry::ProjectRegistry(const QString &projectsRoot) : m_root(projectsRoot) {}

QStringList ProjectRegistry::names() const {
    QStringList out;
    const QDir root(m_root);
    for (const QString &d : root.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)) {
        if (QFile::exists(m_root + "/" + d + "/pipeline.json")) out << d;
    }
    return out;
}

QString ProjectRegistry::projectPath(const QString &name) const {
    return m_root + "/" + name;
}

QString ProjectRegistry::pipelineJsonPath(const QString &name) const {
    return projectPath(name) + "/pipeline.json";
}
