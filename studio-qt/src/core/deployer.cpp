#include "deployer.h"
#include "adbclient.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QtGlobal>

#include <cstdio>

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

// POSIX shell 单引号转义：路径里有空格/引号也不会把命令拆坏
QString sh(const QString &s) {
    QString q = s;
    q.replace(QLatin1Char('\''), QStringLiteral("'\\''"));
    return QLatin1Char('\'') + q + QLatin1Char('\'');
}

bool remoteFileExists(const AdbClient &adb, const QString &path) {
    return adb.shell(QStringLiteral("test -e %1 && echo yes").arg(sh(path))).contains("yes");
}

// 板端运行时按优先级找：build-board 产物 → RVS_LIBRKNNRT → RVS_RKNN_SDK_DIR 官方布局
QString findRuntime(const QString &projectsRoot) {
    QStringList cands;
    cands << QDir(projectsRoot).filePath(QStringLiteral("../build-board/engine/librknnrt.so"));
    const QString envRt = qEnvironmentVariable("RVS_LIBRKNNRT");
    if (!envRt.isEmpty()) cands << envRt;
    const QString sdk = qEnvironmentVariable("RVS_RKNN_SDK_DIR");
    if (!sdk.isEmpty())
        cands << sdk + QStringLiteral("/runtime/Android/librknn_api/arm64-v8a/librknnrt.so");
    for (const QString &c : cands) {
        if (QFileInfo::exists(c)) return QDir::cleanPath(c);
    }
    return {};
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

    // 板端目录名用工程目录名：launch() 只拿到 projectName，用 spec.name 会两边不一致
    const QString name = projectName;

    // 引擎/运行时先确认有来源（本地产物或板上已存在），避免推完工程才发现起不来
    const QString engineLocal =
        QDir::cleanPath(QDir(projectsRoot).filePath(QStringLiteral("../build-board/engine/rvs_engine")));
    const QString rtLocal = findRuntime(projectsRoot);
    if (!QFileInfo::exists(engineLocal) && !remoteFileExists(adb, remoteRoot + "/rvs_engine")) {
        return QStringLiteral("未找到板端引擎: %1（先运行 scripts/build_board.sh）").arg(engineLocal);
    }
    if (rtLocal.isEmpty() && !remoteFileExists(adb, remoteRoot + "/librknnrt.so")) {
        return QStringLiteral("未找到 librknnrt.so（设置 RVS_LIBRKNNRT 或 RVS_RKNN_SDK_DIR）");
    }

    // 收集待推文件：pipeline.json + 工程内其余资源（不再只认 models/assets；
    // smoke/preview 这类工程把图片放在 imgs/）。output/ng_images 是运行产物，跳过。
    QStringList rels;
    rels << QStringLiteral("pipeline.json");
    const QStringList skipDirs = {QStringLiteral("output"), QStringLiteral("ng_images"),
                                  QStringLiteral("build"), QStringLiteral("build-staging"),
                                  QStringLiteral("__pycache__")};
    for (const QFileInfo &fi : projDir.entryInfoList(QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot)) {
        if (fi.fileName().startsWith(QLatin1Char('.'))) continue;
        if (fi.isDir()) {
            if (skipDirs.contains(fi.fileName())) continue;
            collectFiles(QDir(fi.absoluteFilePath()), fi.fileName(), &rels);
        } else if (fi.fileName() != QLatin1String("pipeline.json")) {
            rels << fi.fileName();
        }
    }

    const QString devRoot = remoteRoot + "/" + name;
    fprintf(stderr, "[deploy] mkdir %s\n", devRoot.toUtf8().constData());
    fflush(stderr);
    adb.shell(QStringLiteral("mkdir -p %1").arg(sh(devRoot)));
    fprintf(stderr, "[deploy] %d files to push\n", rels.size());
    fflush(stderr);
    for (const QString &rel : rels) {
        const QString local = projDir.filePath(rel);
        if (!QFileInfo(local).isFile()) continue;
        fprintf(stderr, "[deploy] push %s\n", rel.toUtf8().constData());
        fflush(stderr);
        if (!adb.push(local, devRoot + "/" + rel)) {
            return QStringLiteral("push 失败: %1").arg(rel);
        }
    }

    if (QFileInfo::exists(engineLocal) &&
        !adb.push(engineLocal, remoteRoot + "/rvs_engine")) {
        return QStringLiteral("push 引擎失败: %1").arg(engineLocal);
    }
    if (!rtLocal.isEmpty() && !adb.push(rtLocal, remoteRoot + "/librknnrt.so")) {
        return QStringLiteral("push librknnrt.so 失败: %1").arg(rtLocal);
    }
    adb.shell(QStringLiteral("chmod +x %1").arg(sh(remoteRoot + "/rvs_engine")));
    return {};
}

QString launch(const AdbClient &adb, const QString &projectName,
               const QString &remoteRoot) {
    Deployer::stopEngine(adb);
    // 不用嵌套 sh -c：单引号路径进了内层 shell 会二次解析
    const QString cmd = QStringLiteral("cd %1 && LD_LIBRARY_PATH=%2 nohup %2/rvs_engine . "
                                       "> %1/engine.log 2>&1 &")
                            .arg(sh(remoteRoot + "/" + projectName), sh(remoteRoot));
    adb.shell(cmd);
    return {};
}

void stopEngine(const AdbClient &adb) {
    adb.shell(QStringLiteral("pkill rvs_engine 2>/dev/null; true"));
}

} // namespace Deployer
