"""yolov5 家族转换脚本模板（toolkit2 标准五段：config/load_onnx/build/export/报告）。"""

TEMPLATE = '''#!/usr/bin/env python3
# 由 rvs convert 生成 —— yolov5 家族 ONNX -> RKNN 转换脚本
# 真跑需在 toolkit2 环境（x86 Linux Python）执行；量化不可逐位复现，
# 验收标准见设计文档 §10 Q4（accuracy_analysis 余弦 + 板上金标准匹配率）。
import sys
from rknn.api import RKNN

ONNX_MODEL = "{onnx}"
RKNN_OUT = "{rknn_out}"
DATASET = "{dataset}"

def main():
    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform='{target}',
        quantized_algorithm='normal',
        optimization_level=3,
    )
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('load_onnx failed'); sys.exit(1)
    ret = rknn.build(do_quantization=True, dataset=DATASET)
    if ret != 0:
        print('build failed'); sys.exit(1)
    ret = rknn.export_rknn(RKNN_OUT)
    if ret != 0:
        print('export_rknn failed'); sys.exit(1)
    print('OK:', RKNN_OUT)

if __name__ == '__main__':
    main()
'''


def render_convert_script(onnx, rknn_out, dataset, target):
    return TEMPLATE.format(
        onnx=onnx,
        rknn_out=rknn_out,
        dataset=dataset if dataset else "DATASET_PLACEHOLDER",
        target=target,
    )


def effective_dry_run(args):
    """dry-run 是默认安全态；dataset + toolkit2_python 齐备且未显式 --dry-run 才真跑。"""
    if getattr(args, "dry_run", False):
        return True
    return not (getattr(args, "dataset", None) and getattr(args, "toolkit2_python", None))


def cmd_convert(args) -> int:
    from pathlib import Path
    if effective_dry_run(args):
        text = render_convert_script(
            onnx=str(Path(args.onnx).resolve()),
            rknn_out=str(Path(args.out).resolve()),
            dataset=str(Path(args.dataset).resolve()) if args.dataset else None,
            target=args.target,
        )
        out = Path(args.out + ".convert.py") if args.out.endswith(".rknn") else Path(args.out)
        out.write_text(text)
        print(f"convert script written: {out}")
        print("(dry-run: 未执行真转换；真跑需 --dataset + --toolkit2-python 且不加 --dry-run)")
        return 0
    text = render_convert_script(
        onnx=str(Path(args.onnx).resolve()),
        rknn_out=str(Path(args.out).resolve()),
        dataset=str(Path(args.dataset).resolve()),
        target=args.target,
    )
    out = Path(args.out + ".convert.py") if args.out.endswith(".rknn") else Path(args.out)
    out.write_text(text)
    print(f"convert script written: {out}")
    import subprocess
    return subprocess.call([args.toolkit2_python, str(out)])
