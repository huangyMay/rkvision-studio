#!/usr/bin/env python3
"""RK3576 板上 w16a16i(i16) NPU 算子输出回传 bug 最小复现。

背景（2026-10-25 实验链，详见 rknn问题处理分析.md §6.3）：
  w16a16i 模型中 Conv 自动回退 CPU（build 日志 Conv Not Support u8->i16），
  但 Sigmoid / Concat 仍以 i16 排上 NPU —— 其输出回传恒 ACK_FAIL，
  任何含这两算子的 i16 模型（= 所有检测模型）板上必炸；
  op_target 强制 cpu 在 i16 模式不生效，无软件修法。
  全 CPU 的小模型（Conv+Relu/Mul/Add/MaxPool）输出全部正常。

用法（WSL，板以网络 adb 在线）:
  python3 int16_board_repro.py <板IP:端口> [dataset.txt]
不带参数则用内置随机图自建校准集。每个算子约 40~60s。
"""
import os
import sys
import tempfile

import numpy as np


def build_cases():
    import onnx
    from onnx import TensorProto, helper
    W = (np.random.rand(4, 3, 3, 3).astype(np.float32) - 0.5) * 0.1
    Wt = helper.make_tensor('w', TensorProto.FLOAT, W.shape, W.flatten().tolist())

    def conv(i, o):
        return helper.make_node('Conv', [i, 'w'], [o], name='conv0',
                                strides=[1, 1], pads=[1, 1, 1, 1])

    sigmoid = lambda: ([conv('x', 'c'),
                        helper.make_node('Sigmoid', ['c'], ['y'], name='sg')],
                       [Wt], [1, 4, 640, 640])
    concat = lambda: ([conv('x', 'c'),
                       helper.make_node('Concat', ['c', 'c'], ['y'], name='cc', axis=1)],
                      [Wt], [1, 8, 640, 640])
    relu = lambda: ([conv('x', 'c'),
                     helper.make_node('Relu', ['c'], ['y'], name='rl')],
                    [Wt], [1, 4, 640, 640])
    return {'Sigmoid(预期炸)': sigmoid, 'Concat(预期炸)': concat, 'Relu(预期过)': relu}


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else None
    if not device:
        print(__doc__)
        sys.exit(1)
    import onnx
    from onnx import TensorProto, helper
    import cv2
    from rknn.api import RKNN

    tmp = tempfile.mkdtemp(prefix='i16repro_')
    img = os.path.join(tmp, 'calib.jpg')
    cv2.imwrite(img, np.random.rand(640, 640, 3).astype(np.float32) * 255)
    ds = os.path.join(tmp, 'ds.txt')
    open(ds, 'w').write(img + '\n')

    for tag, build in build_cases().items():
        nodes, inits, oshape = build()
        g = helper.make_graph(nodes, 'm',
                              [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 640, 640])],
                              [helper.make_tensor_value_info('y', TensorProto.FLOAT, oshape)], inits)
        path = os.path.join(tmp, tag.replace('(', '_').replace(')', '') + '.onnx')
        onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid('', 18)]), path)
        rk = RKNN(verbose=False)
        rk.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                  target_platform='rk3576', quantized_dtype='w16a16i')
        rk.load_onnx(path)
        ok = rk.build(do_quantization=True, dataset=ds) == 0 \
            and rk.init_runtime(target='rk3576', device_id=device) == 0
        if ok:
            try:
                o = rk.inference(inputs=[np.random.rand(1, 640, 640, 3).astype(np.float32) * 255])
                print(f"[{tag}] {'OK 输出可回传' if o is not None else 'None'}", flush=True)
            except Exception:
                print(f"[{tag}] FAIL Get outputs failed (ACK_FAIL)", flush=True)
        else:
            print(f"[{tag}] build/init 失败", flush=True)
        rk.release()


if __name__ == '__main__':
    main()
