# Task 12: compare.py 的行为规格（金标准比对：类别一致 + IoU>=0.98 + conf |Δ|<=0.01）
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rvs.compare import compare, iou, parse_demo_log  # noqa: E402


def box(x1, y1, x2, y2, label="helmet", score=0.9):
    return {"label": label, "score": score, "box": [x1, y1, x2, y2]}


def test_parse_demo_log():
    text = "rknn_init\ntime_use is 111.734001\nhelmet @ (879 389 948 479) 0.875565\n" \
           "head @ (10 20 30 40) 0.5\n"
    boxes = parse_demo_log(text)
    assert len(boxes) == 2
    assert boxes[0] == {"label": "helmet", "score": 0.875565, "box": [879.0, 389.0, 948.0, 479.0]}
    assert boxes[1]["label"] == "head"


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_compare_pass():
    gold = [box(10, 10, 50, 50, score=0.9), box(100, 100, 200, 200, score=0.8)]
    engine = [box(10, 10, 50, 50, score=0.895), box(100, 100, 200, 200, score=0.805)]
    ok, report = compare(engine, gold)
    assert ok, report


def test_compare_fail_low_iou():
    gold = [box(10, 10, 50, 50)]
    engine = [box(12, 12, 50, 50)]   # IoU ≈ 0.83 < 0.98
    ok, report = compare(engine, gold)
    assert not ok
    assert "IoU" in report or "unmatched" in report


def test_compare_fail_conf_delta():
    gold = [box(10, 10, 50, 50, score=0.9)]
    engine = [box(10, 10, 50, 50, score=0.88)]   # |Δ|=0.02 > 0.01
    ok, _ = compare(engine, gold)
    assert not ok


def test_compare_fail_label():
    gold = [box(10, 10, 50, 50, label="helmet")]
    engine = [box(10, 10, 50, 50, label="head")]
    ok, _ = compare(engine, gold)
    assert not ok


def test_compare_count_mismatch():
    gold = [box(0, 0, 10, 10), box(20, 20, 30, 30)]
    engine = [box(0, 0, 10, 10)]
    ok, _ = compare(engine, gold)
    assert not ok


if __name__ == "__main__":
    raise SystemExit("run with pytest")
