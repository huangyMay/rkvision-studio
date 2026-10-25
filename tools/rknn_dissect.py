#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rknn 文件解剖（收编自 05-librknn编写/tools/rknn_dissect.py，2026-10-25）。

离线解析 .rknn 容器布局——布局参照板上 iotrace 实证（librknnrt原理与架构分析.md §4）：
  0x0 "RKNN" 头 ... 16B 节头{ver,magic=0x00040004,count,total} @ 某偏移
  节头后 total 字节 = 任务记录 N×40B（+28 u32=regcmd_offset, +32 u64=加载期重定位地址）
  任务段之后到 EOF = regcmd 流 + 内嵌权重

Studio 集成（🔪 按钮）：
  单文件: python3 rknn_dissect.py <model.rknn>            → JSON 布局报告
  双文件: python3 rknn_dissect.py <a.rknn> <b.rknn>       → 附差分摘要（混合量化前后对比用）

est_submit_groups 为文件级估计（任务 regcmd 偏移按间隙聚类）；
精确 SUBMIT 段数需板上 iotrace（05-librknn编写/tools/libiotrace.c）。
"""
import json
import struct
import sys

MAGIC = struct.pack('<I', 0x00040004)
TARGETS = [b'rk3566', b'rk3568', b'rk3576', b'rk3588', b'rk3562', b'rv1103', b'rv1106']


def parse(path):
    """单文件解析：布局四段 + 任务表 + target 扫描 + 提交分组估计。"""
    b = open(path, 'rb').read()
    info = {'size': len(b), 'head': b[:16].hex()}
    # target_platform：容器内明文 ASCII（转换期写入）
    for t in TARGETS:
        if t in b:
            info['target'] = t.decode()
            break
    # 搜节头 magic（跳过文件头 0x100，避开巧合；4B 对齐扫描）
    for off in range(0x100, min(len(b), 0x200000), 4):
        if b[off:off + 4] != MAGIC:
            continue
        ver, mgc, cnt, total = struct.unpack('<IIII', b[off - 4:off + 12])
        if ver == 0 and 0 < total < len(b) and total % 40 == 0:
            info['sec_off'] = off - 4
            info['sec'] = {'ver': ver, 'magic': hex(mgc), 'count': cnt, 'total': total}
            n = total // 40
            info['n_tasks'] = n
            info['task_off'] = off + 12
            info['reg_off'] = off + 12 + total
            rc = [struct.unpack('<I', b[info['task_off'] + i * 40 + 28:
                                         info['task_off'] + i * 40 + 32])[0] for i in range(n)]
            info['rc_offs'] = rc
            # 提交分组估计：regcmd 偏移单调流里的大间隙 ≈ 提交段边界。
            # 锚点：y8p 实测 1332 任务 / 板上 67 段；hybrid 1033 任务 / 67 段——
            # 间隙聚类是文件级最接近的代理，精确段数以板上 iotrace 为准。
            if n > 2:
                diffs = [rc[i + 1] - rc[i] for i in range(n - 1) if rc[i + 1] > rc[i]]
                if diffs:
                    diffs.sort()
                    thr = diffs[len(diffs) // 2] * 8      # 中位间隙 ×8 以上算段边界
                    groups = 1 + sum(1 for d in diffs if d > thr)
                    info['est_submit_groups'] = groups
                    info['est_note'] = 'regcmd 间隙聚类估计；精确值需板上 iotrace'
            break
    # 布局四段表（供 UI 条形展示）
    if 'n_tasks' in info:
        info['layout'] = [
            {'name': 'RKNN 文件头（版本/元信息）', 'start': 0, 'end': info['sec_off']},
            {'name': '节头 16B（magic 0x00040004）', 'start': info['sec_off'], 'end': info['task_off']},
            {'name': '任务表 %d × 40B' % info['n_tasks'], 'start': info['task_off'], 'end': info['reg_off']},
            {'name': 'regcmd 指令流 + 权重（到 EOF）', 'start': info['reg_off'], 'end': len(b)},
        ]
        info['weight_ratio'] = round((len(b) - info['reg_off']) / len(b), 4)
    return info


def diff_summary(a, b):
    """双文件字节差分 → 差异区段归属（混合量化前后对比）。"""
    ia, ib = parse(a), parse(b)
    ba, bb = open(a, 'rb').read(), open(b, 'rb').read()
    out = {'a': ibrief(ia), 'b': ibrief(ib)}
    if ia.get('n_tasks') and ib.get('n_tasks'):
        out['task_delta'] = ib['n_tasks'] - ia['n_tasks']
    n = min(len(ba), len(bb))
    regions = []
    i = 0
    while i < n:
        if ba[i] != bb[i]:
            j, gap, last = i, 0, i
            while j < n and gap < 64:
                if ba[j] != bb[j]:
                    last, gap = j, 0
                else:
                    gap += 1
                j += 1
            if last - i >= 16:
                regions.append([i, last + 1])
            i = last + 1 + gap
        else:
            i += 1
    out['diff_regions'] = len(regions)
    out['diff_bytes'] = sum(e - s for s, e in regions)
    if len(bb) != len(ba):
        out['size_delta'] = len(bb) - len(ba)
    return out


def ibrief(i):
    return {k: i.get(k) for k in ('size', 'target', 'n_tasks', 'est_submit_groups', 'reg_off')}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({'ok': False, 'error': 'usage: rknn_dissect.py <a.rknn> [b.rknn]'}))
        return
    a = sys.argv[1]
    ia = parse(a)
    out = {'ok': 'n_tasks' in ia, 'model': a.split('/')[-1]}
    out.update({k: v for k, v in ia.items() if k != 'rc_offs'})
    if 'rc_offs' in ia:
        out['rc_min'], out['rc_max'] = min(ia['rc_offs']), max(ia['rc_offs'])
    if len(sys.argv) > 2:
        out['diff'] = diff_summary(a, sys.argv[2])
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
