#include "model/workflowgen.h"
#include "ui/mainwindow.h"

#include <QApplication>
#include <QFile>
#include <QJsonDocument>
#include <QStyleFactory>
#include <QTimer>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QApplication::setStyle(QStyleFactory::create(QStringLiteral("Fusion")));

    // 工作流 JSON 生成回归（无 GUI）：--workflowgen <out.json>
    const QStringList args = app.arguments();
    const int wfIdx = args.indexOf(QStringLiteral("--workflowgen"));
    if (wfIdx >= 0 && args.size() > wfIdx + 1) {
        WorkflowForm f;                    // 默认表单（与 tests/workflowgen_golden.json 同源）
        f.projectName = QStringLiteral("workflow_demo");
        f.modelFile = QStringLiteral("models/helmet_detect_my.rknn");
        f.rules = {{QStringLiteral("score"), QStringLiteral("*"), QStringLiteral("0.45"), QString()},
                   {QStringLiteral("count"), QStringLiteral("*"), QStringLiteral("1"), QStringLiteral("99")}};
        const QJsonObject obj = WorkflowGen::buildJson(f);
        QFile out(args.at(wfIdx + 1));
        if (out.open(QIODevice::WriteOnly)) {
            out.write(QJsonDocument(obj).toJson());
            out.close();
            fprintf(stdout, "workflowgen ok: %s\n", args.at(wfIdx + 1).toUtf8().constData());
            return 0;
        }
        fprintf(stderr, "workflowgen: cannot write %s\n", args.at(wfIdx + 1).toUtf8().constData());
        return 1;
    }

    // 科技感主题
    QFile qss(QStringLiteral(":/res/theme.qss"));
    if (qss.open(QIODevice::ReadOnly | QIODevice::Text))
        app.setStyleSheet(QString::fromUtf8(qss.readAll()));

    MainWindow w;
    w.showMaximized();

    // 回归触发器（独立于截图模式；与 --screenshot 组合时可自动存盘退出）
    const int rtIdx = args.indexOf(QStringLiteral("--autortsp"));
    if (rtIdx >= 0 && args.size() > rtIdx + 1) {
        const QString url = args.at(rtIdx + 1);
        QTimer::singleShot(800, &w, [&w, url]() { w.autoStartRtsp(url); });
    }
    const int imgIdx = args.indexOf(QStringLiteral("--inferimage"));
    const int mdlIdx = args.indexOf(QStringLiteral("--infermodel"));
    if (imgIdx >= 0 && args.size() > imgIdx + 1
            && mdlIdx >= 0 && args.size() > mdlIdx + 1) {
        const QString img = args.at(imgIdx + 1);
        const QString mdl = args.at(mdlIdx + 1);
        QTimer::singleShot(500, &w, [&w, img, mdl]() { w.autoRunInfer(img, mdl); });
    }
    const int simImgIdx = args.indexOf(QStringLiteral("--simimage"));
    const int simMdlIdx = args.indexOf(QStringLiteral("--simmodel"));
    if (simImgIdx >= 0 && args.size() > simImgIdx + 1
            && simMdlIdx >= 0 && args.size() > simMdlIdx + 1) {
        const QString img = args.at(simImgIdx + 1);
        const QString mdl = args.at(simMdlIdx + 1);
        const int msIdx = args.indexOf(QStringLiteral("--simmeanstd"));
        const QString ms = (msIdx >= 0 && args.size() > msIdx + 1)
                               ? args.at(msIdx + 1) : QStringLiteral("0,0,0/255,255,255");
        QTimer::singleShot(500, &w, [&w, img, mdl, ms]() { w.autoRunRknnSim(img, mdl, ms); });
    }
    const int liIdx = args.indexOf(QStringLiteral("--liveinfer"));
    if (liIdx >= 0 && args.size() > liIdx + 1) {
        const QString mdl = args.at(liIdx + 1);
        QTimer::singleShot(500, &w, [&w, mdl]() { w.autoLiveInfer(mdl); });
    }
    const int depIdx = args.indexOf(QStringLiteral("--deploy"));
    if (depIdx >= 0 && args.size() > depIdx + 1) {
        const QString proj = args.at(depIdx + 1);
        const int serIdx = args.indexOf(QStringLiteral("--serial"));
        const QString ser = (serIdx >= 0 && args.size() > serIdx + 1)
                                ? args.at(serIdx + 1) : QString();
        QTimer::singleShot(500, &w, [&w, proj, ser]() { w.autoDeploy(proj, ser); });
        QTimer::singleShot(60000, &w, []() { QApplication::quit(); });  // 部署完自动退出
    }

    const int wfViewIdx = args.indexOf(QStringLiteral("--workflow"));
    if (wfViewIdx >= 0)
        QTimer::singleShot(300, &w, [&w]() { w.autoShowWorkflow(); });

    const int perfLabIdx = args.indexOf(QStringLiteral("--perflab"));
    if (perfLabIdx >= 0)
        QTimer::singleShot(300, &w, [&w]() { w.autoShowPerfLab(); });

    // 截图模式：--screenshot <out.png> —— 渲染后存盘退出（配合上面任一触发器）
    const int shotIdx = args.indexOf(QStringLiteral("--screenshot"));
    if (shotIdx >= 0 && args.size() > shotIdx + 1) {
        const QString out = args.at(shotIdx + 1);
        QTimer::singleShot(9000, &w, [&w, out]() {
            w.grab().save(out);
            QApplication::quit();
        });
    }

    return app.exec();
}
