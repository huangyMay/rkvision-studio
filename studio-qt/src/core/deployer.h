// 工程打包部署：projects/<name> → adb push → 板上拉起
#pragma once
#include <QString>

class AdbClient;

namespace Deployer {
// 打包并部署工程（pipeline.json/models/assets → /data/local/tmp/rvs/<name>/）
// 返回空串=成功，否则错误信息
QString deploy(const AdbClient &adb, const QString &projectName,
               const QString &projectsRoot, const QString &remoteRoot);

// 启动板上引擎（后台常驻）
QString launch(const AdbClient &adb, const QString &projectName,
               const QString &remoteRoot);

// 停止板上引擎
void stopEngine(const AdbClient &adb);
}
