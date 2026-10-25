// workflowgen 黄金文件回归：表单默认值 → pipeline.json 必须逐字节等于 tests/workflowgen_golden.json。
// 同时用 PipelineSpec::parse 校验黄金文件本身能过 PC 侧 schema（与 engine/core/spec.cpp 同规则）。
#include "model/pipelinespec.h"
#include "model/workflowgen.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>

#include <cstdio>

#ifndef RVS_GOLDEN_PATH
#error "RVS_GOLDEN_PATH must be defined by CMake"
#endif

static int fail(const char *what) {
    fprintf(stderr, "test_workflowgen: %s\n", what);
    return 1;
}

int main() {
    WorkflowForm f;
    f.projectName = QStringLiteral("workflow_demo");
    f.modelFile = QStringLiteral("/tmp/helmet_detect_my.rknn");   // 只取 basename
    f.classes = QStringLiteral("head,helmet");
    f.conf = 0.25;
    f.iou = 0.45;
    f.ngArchive = true;
    f.imageOut = true;
    f.mjpeg = false;

    WorkflowRule score;
    score.type = QStringLiteral("score");
    score.cls = QStringLiteral("*");
    score.param1 = QStringLiteral("0.45");
    WorkflowRule count;
    count.type = QStringLiteral("count");
    count.cls = QStringLiteral("*");
    count.param1 = QStringLiteral("1");
    count.param2 = QStringLiteral("99");
    f.rules = {score, count};

    QFile gf(QStringLiteral(RVS_GOLDEN_PATH));
    if (!gf.open(QIODevice::ReadOnly)) return fail("cannot open golden file");
    QJsonParseError pe;
    const QJsonDocument golden = QJsonDocument::fromJson(gf.readAll(), &pe);
    if (golden.isNull()) return fail(pe.errorString().toUtf8().constData());

    QJsonObject got;
    try {
        got = WorkflowGen::buildJson(f);
    } catch (const QString &e) {
        fprintf(stderr, "test_workflowgen: buildJson threw: %s\n", e.toUtf8().constData());
        return 1;
    }

    const QByteArray a = QJsonDocument(got).toJson(QJsonDocument::Compact);
    const QByteArray b = QJsonDocument(golden.object()).toJson(QJsonDocument::Compact);
    if (a != b) {
        fprintf(stderr, "test_workflowgen: golden mismatch\n  got : %s\n  want: %s\n", a.constData(),
                b.constData());
        return 1;
    }

    QString err;
    const PipelineSpec spec = PipelineSpec::parse(QString::fromUtf8(b), &err);
    if (!err.isEmpty()) {
        fprintf(stderr, "test_workflowgen: golden rejected by PipelineSpec::parse: %s\n",
                err.toUtf8().constData());
        return 1;
    }
    if (spec.pipeline.size() != 7) return fail("expected 7 stages in golden pipeline");

    printf("test_workflowgen: ok (%d stages)\n", (int)spec.pipeline.size());
    return 0;
}
