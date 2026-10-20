"""金标准比对：engine 结果 vs demo 基线。

匹配规则（设计文档 §10 Q1）：贪心按 IoU 降序一一配对，类别必须一致；
IoU >= 0.98 且 |conf 差| <= 0.01 视为匹配。数量不同直接 FAIL。
"""
import re


def parse_demo_log(text):
    """解析 demo 基线日志行：`label @ (x0 y0 x1 y1) conf`"""
    boxes = []
    pat = re.compile(r"(\S+)\s+@\s+\(([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\)\s+([-\d.eE+]+)")
    for m in pat.finditer(text):
        boxes.append({
            "label": m.group(1),
            "box": [float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))],
            "score": float(m.group(6)),
        })
    return boxes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compare(engine_boxes, gold_boxes, iou_tol=0.98, conf_tol=0.01):
    """返回 (PASS, 报告文本)。engine/gold 元素形如 {"label","score","box":[x1,y1,x2,y2]}"""
    lines = []
    pairs = []
    for gi, g in enumerate(gold_boxes):
        for ei, e in enumerate(engine_boxes):
            if g["label"] != e["label"]:
                continue
            v = iou(g["box"], e["box"])
            if v > 0:
                pairs.append((v, gi, ei))
    pairs.sort(reverse=True)

    gold_matched = {}
    engine_matched = {}
    for v, gi, ei in pairs:
        if gi in gold_matched or ei in engine_matched:
            continue
        g, e = gold_boxes[gi], engine_boxes[ei]
        if v < iou_tol:
            continue
        if abs(g["score"] - e["score"]) > conf_tol:
            continue
        gold_matched[gi] = ei
        engine_matched[ei] = gi

    for gi, g in enumerate(gold_boxes):
        if gi not in gold_matched:
            best = max((iou(g["box"], e["box"]) for e in engine_boxes
                        if e["label"] == g["label"]), default=0.0)
            lines.append(f"unmatched gold[{gi}] {g['label']} {g['box']} conf={g['score']:.4f} "
                         f"best_same_label_iou={best:.4f}")
    for ei, e in enumerate(engine_boxes):
        if ei not in engine_matched:
            lines.append(f"unmatched engine[{ei}] {e['label']} {e['box']} conf={e['score']:.4f}")

    ok = len(gold_matched) == len(gold_boxes) == len(engine_boxes)
    lines.insert(0, f"matched {len(gold_matched)}/{len(gold_boxes)} gold, "
                    f"{len(engine_matched)}/{len(engine_boxes)} engine, "
                    f"tol(iou>={iou_tol}, |dconf|<={conf_tol})")
    return ok, "\n".join(lines)
