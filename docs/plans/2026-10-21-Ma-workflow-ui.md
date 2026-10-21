# M-a WORKFLOW 配置器 UI 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans（内联执行）。步骤用 checkbox 跟踪。

**Goal:** Qt Studio 第四页签 WORKFLOW：分段表单卡片 → 生成 `projects/<名>/pipeline.json`（含 judge/archive）→ 自动进工程下拉 → 一键部署运行，闭环"编辑与创建"能力。

**Architecture:** JSON 生成做成 model 层纯函数（`WorkflowGen::buildJson`，无 QWidget 可回归验证）；UI 页只做表单采集并调用生成器；保存后复用现有 ProjectRegistry 扫描（新目录自动入列）与 `onDeploy()` 代码路径。设计依据 `docs/2026-10-18-工业工作流设计.md` §5。

**Tech Stack:** Qt 6 Widgets（QTableWidget 规则表格）/ QJson / 现有 nav(QButtonGroup)+QStackedWidget 结构。

**Spec:** `docs/2026-10-18-工业工作流设计.md` §5（UI 形态）、§2.1（judge DSL）、§6（schema：插件名 judge/archive 已在引擎注册）。

## Global Constraints

- 引擎插件参数名逐字对齐（已核实）：image_dir{path,loop}、v4l2{device,width,height,prefer_mjpeg}、letterbox{size,pad_value}、infer(model 由 spec 注入)、yolov5_det/yolov8_det{conf_thres,iou_thres,classes}、judge{rules,verdict,actions}、archive{ng_dir,keep_days}、image_out{path,draw,save_json}、mjpeg{port,path,draw,fps}
- family→post 插件映射：`yolov5-det→yolov5_det`、`yolov8-det→yolov8_det`
- 生成 JSON 必须是自包含工程：模型文件保存时复制进 `<工程>/models/<basename>`，JSON 只写相对路径
- judge 规则 DSL 与引擎 JudgeCore::ParseRules 完全同构（type/class/min/max/thres/min_px/max_px/polygon/inside/frames/pattern）
- 不做节点画布；不加新第三方依赖；旧三页行为零改动

## Review Focus

1. 规则表某行类型为 region 但参数不是合法 "[x,y;x,y;...]" 文本 → 生成时 Init 期即可被引擎 ParseRules 拒绝，但 UI 保存前应给出行级错误提示（Task 2 验证）
2. 模型文件不存在/未选 → 保存必须拒绝并提示（不产半截工程）
3. 工程名含空格/中文/非法路径字符 → 保存拒绝或安全转换
4. v4l2 源时不得生成 image_dir 节点（反之亦然）；image_out 与 mjpeg 可同时开
5. `--workflowgen` 回归产物与 golden 逐键相等（键序无关）

---

### Task 1: model 层生成器 + `--workflowgen` 回归

**Files:**
- Create: `studio-qt/src/model/workflowgen.h` / `.cpp`
- Modify: `studio-qt/src/main.cpp`（顶层触发器 `--workflowgen <out.json>`，与 --deploy 同模式）
- Create: `studio-qt/tests/workflowgen_golden.json`（golden，手写自默认表单）
- Modify: `studio-qt/CMakeLists.txt`（源文件）

**Interfaces:**
- Produces:
```cpp
struct WorkflowRule { QString type, cls, param1, param2; };   // 参数含义随 type，见 UI tooltip
struct WorkflowForm {
    QString projectName;
    // 源
    bool sourceV4l2 = false;  QString sourcePath = "assets/test_images";
    QString device = "/dev/video0"; int width = 1280, height = 720;
    // 预处理
    int size = 640, padValue = 255;
    // 模型 + 后处理
    QString modelFile;  QString family = "yolov5-det";
    QString classes = "head,helmet";  double conf = 0.25, iou = 0.45;
    // 判决
    QList<WorkflowRule> rules;  bool ngArchive = true;  QString ngDir = "ng_images";
    // 输出
    bool imageOut = true;  bool mjpeg = false;  int mjpegPort = 8000;
};
namespace WorkflowGen {
    QJsonObject buildJson(const WorkflowForm &f);            // 纯函数；规则行非法抛 QString
    QString save(const WorkflowForm &f, const QString &projectsRoot, QString *err);
    // save：建目录 + 复制模型 + 写 pipeline.json；失败写 *err 返回空串
}
```
- 默认表单（--workflowgen 与 golden 用同一份）：helmet_gold 同构——image_dir+judge(score *,0.45)+judge(count *,1,99)+archive+image_out

- [ ] **Step 1: RED**：main.cpp 加 `--workflowgen <path>`（先调 buildJson——未实现编译失败即 RED）
- [ ] **Step 2: 确认编译失败**
- [ ] **Step 3: 实现 workflowgen.h/.cpp（buildJson + save）**
- [ ] **Step 4: GREEN**：`rvs_studio.exe --workflowgen /tmp/wf.json` 退出 0；WSL python3 对比 golden 逐键相等
- [ ] **Step 5: Commit** `feat: workflow pipeline.json generator (model layer)`

### Task 2: WORKFLOW 页表单 UI

**Files:**
- Create: `studio-qt/src/ui/workflowpage.h` / `.cpp`
- Modify: `studio-qt/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1 `WorkflowForm/WorkflowGen`
- Produces: `class WorkflowPage : public QWidget`；`WorkflowForm collect() const`（读表单）；`void signal projectSaved(const QString &name)`；[生成并保存]=collect→WorkflowGen::save→emit projectSaved；[保存并部署到板]=同上+emit deployRequested(name)

- [ ] **Step 1: 实现表单卡**（源/预处理/模型/后处理/判决/输出六卡；规则表 QTableWidget 4 列[类型|类别|参数1|参数2]+增/删行按钮+按类型的列头 tooltip；模型浏览 QFileDialog *.rknn）
- [ ] **Step 2: 校验与错误提示**（Review Focus 2/3：模型缺失/工程名非法→QMessageBox 拒绝；region 参数文本格式提示）
- [ ] **Step 3: 构建 via cmake --build rvs_qt_build** 通过
- [ ] **Step 4: Commit** `feat: workflow configurator page UI`

### Task 3: mainwindow 集成第四页签

**Files:**
- Modify: `studio-qt/src/ui/mainwindow.cpp` / `.h`（navWorkflow 按钮 + m_views index 3 + WorkflowPage 实例 + projectSaved→refreshProjects+选中+deployRequested→onDeploy()）
- Modify: `studio-qt/src/main.cpp`（`--workflow` 触发器：启动切第 4 页，配合 --screenshot 回归）

**Interfaces:**
- Consumes: WorkflowPage::projectSaved/deployRequested；现有 refreshProjects()/onDeploy()

- [ ] **Step 1: 集成 + `--workflow --screenshot` 回归通过（第 4 页可见、六卡布局完整）**
- [ ] **Step 2: 端到端**：`--workflowgen` 产物落 projects/ 后 refresh 能列出（用临时 projectsRoot 或直接 UI 手工路径验证）
- [ ] **Step 3: Commit** `feat: WORKFLOW tab integration + regression trigger`

### Task 4: 文档 + 推送

- [ ] **Step 1: 手册补 WORKFLOW 页用法节；部署目录 exe 更新**
- [ ] **Step 2: Commit + git push（含此前积压的 11 个 commit）**

## Self-Review

规格 §5 的表单卡/规则表/生成保存/复用部署链均有任务；范围外（PLC、异常检测向导、模板库）不进本计划。Task 1 的 Interfaces 即 Task 2/3 的消费契约，命名一致。
