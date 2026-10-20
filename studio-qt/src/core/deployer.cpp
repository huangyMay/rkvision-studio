#include "deployer.h"
#include "adbclient.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>

namespace {

QString readFile(const QString &path) {
    QFile f(path);
    return f.open(QIODevice::ReadOnly) ? QString::fromUtf8(f.readAll()) : QString();
}

// 递归收集目录下相对路径
void collectFiles(const QDir &dir, const QString &prefix, QStringList *out) {
    for (const QFileInfo &fi : dir.entryInfoList(QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot)) {
        const QString rel = prefix.isEmpty() ? fi.fileName() : prefix + "/" + fi.fileName();
        if (fi.isDir()) collectFiles(QDir(fi.absoluteFilePath()), rel, out);
        else out->append(rel);
    }
}

} // namespace

namespace Deployer {

QString deploy(const AdbClient &adb, const QString &projectName,
               const QString &projectsRoot, const QString &remoteRoot) {
    const QDir projDir(projectsRoot + "/" + projectName);
    if (!projDir.exists()) return QStringLiteral("工程目录不存在: %1").arg(projectName);

    const QString json = readFile(projDir.filePath("pipeline.json"));
    const QJsonObject spec = QJsonDocument::fromJson(json.toUtf8()).object();
    if (spec.value("version").toInt() != 1) return QStringLiteral("pipeline.json version 不是 1");
    const QString name = spec.value("name").toString(projectName);

    // 收集待推文件（pipeline.json + models + assets）
    QStringList rels;
    rels << QStringLiteral("pipeline.json");
    collectFiles(projDir.filePath("models"), "models", &rels);
    collectFiles(projDir.filePath("assets"), "assets", &rels);

    const QString devRoot = remoteRoot + "/" + name;
    fprintf(stderr, "[deploy] mkdir %s\n", devRoot.toUtf8().constData());
    fflush(stderr);
    adb.shell(QStringLiteral("mkdir -p %1").arg(devRoot));
    fprintf(stderr, "[deploy] mkdir done\n");
    fflush(stderr);
    fprintf(stderr, "[deploy] %d files to push\n", rels.size());
    fflush(stderr);
    for (const QString &rel : rels) {
        fprintf(stderr, "[deploy] push %s\n", rel.toUtf8().constData());
        fflush(stderr);
        const QString local = projDir.filePath(rel);
        if (!QFileInfo(local).isFile()) continue;
        if (!adb.push(local, devRoot + "/" + rel)) {
            return QStringLiteral("push 失败: %1").arg(rel);
        }
    }

    // 引擎与运行时若在 PC 侧就绪则一并推（首次）
    const QString engineLocal = projectsRoot + "/../../build-board/engine/rvs_engine";
    const QString rtLocal = projectsRoot + "/../../build-board/engine/librknnrt.so";
    if (QFileInfo::exists(engineLocal)) adb.push(engineLocal, remoteRoot + "/rvs_engine");
    if (QFileInfo::exists(rtLocal)) adb.push(rtLocal, remoteRoot + "/librknnrt.so");
    adb.shell(QStringLiteral("chmod +x %1/rvs_engine").arg(remoteRoot));
    return {};
}

QString launch(const AdbClient &adb, const QString &projectName,
               const QString &remoteRoot) {
    Deployer::stopEngine(adb);
    const QString cmd = QStringLiteral(
        "cd %1/%2 && nohup sh -c 'LD_LIBRARY_PATH=%1 %1/rvs_engine .' "
        "> %1/%2/engine.log 2>&1 &").arg(remoteRoot, projectName);
    adb.shell(cmd);
    return {};
}

void stopEngine(const AdbClient &adb) {
    adb.shell(QStringLiteral("pkill rvs_engine 2>/dev/null; true"));
}

} // namespace Deployer
