# Task 13: convert 命令骨架（dry-run 生成转换脚本）
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rvs.convert import render_convert_script  # noqa: E402


def test_render_contains_key_config():
    text = render_convert_script(
        onnx="/data/model.onnx", rknn_out="/data/out.rknn",
        dataset="/data/calib", target="rk3576")
    assert "target_platform='rk3576'" in text
    assert "do_quantization=True" in text
    assert "/data/model.onnx" in text
    assert "/data/out.rknn" in text
    assert "/data/calib" in text
    assert "export_rknn" in text


def test_render_without_dataset_places_placeholder():
    text = render_convert_script(
        onnx="/data/model.onnx", rknn_out="/data/out.rknn",
        dataset=None, target="rk3576")
    assert "DATASET_PLACEHOLDER" in text


def test_effective_dry_run_semantics():
    from rvs.convert import effective_dry_run

    class A:
        dry_run = False
        dataset = None
        toolkit2_python = None

    a = A()
    assert effective_dry_run(a) is True          # 什么都不给 → dry-run（默认安全）

    a.dataset = "/data/calib"
    assert effective_dry_run(a) is True          # 只有 dataset → 仍 dry-run

    a.toolkit2_python = "/opt/tk2/bin/python"
    assert effective_dry_run(a) is False         # 齐了才真跑

    a.dry_run = True                             # 显式 dry-run 永远 dry-run
    assert effective_dry_run(a) is True
