#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RKNN 板端性能剖析（Studio 性能剖析页后端，2026-10-25）。

编排：连接板（adb | ssh 双通道）→ 部署 perf_run → [锁 NPU/DDR 频率] →
core_mask 矩阵逐档跑分段计时 → 汇总 JSON。

分段含义（perf_run.c 实测口径，预热 5 帧后 N 帧均值）：
  set = rknn_inputs_set   归一化 RNE / 布局拷贝税
  run = rknn_run          纯 NPU 执行（架构对比该看的数字）
  get = rknn_outputs_get  want_float=1 的 fp32 展开回传税

用法:
  python3 perf_probe.py <model.rknn> <target> <masks csv> <loops> <fixfreq 0|1>
  target: adb 序列 / IP:端口（走 adb，Android 板）
          root@IP（走 ssh，Debian 板——需免密 key）

流式协议（AccCompareRunner 同款）：LOG 行 = 进度；末行 {"ok":...}
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_infer import norm_path                                    # noqa: E402
from rknn_acc import adb_path_prepend                               # noqa: E402

TOOLS = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print("LOG %s" % msg, flush=True)


def run(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def parse_json_line(out):
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith('{"ok"'):
            try:
                return json.loads(ln)
            except ValueError:
                pass
    return None


class Chan:
    """板通道：传输 = adb | ssh；perf_run 按 OS 探测选 NDK 预编(Android) / 板上 gcc 自编(Debian)。

    Debian 板也可能开 root adb（RK3588 Debian 镜像默认带）——传输与 OS 正交。
    """

    def __init__(self, target):
        self.via_ssh = '@' in target
        self.target = target
        self.serial = None if self.via_ssh else target

    # ---- 传输原语 ----
    def sh(self, cmd, timeout=60):
        if self.via_ssh:
            return run("ssh -o StrictHostKeyChecking=no -o ConnectTimeout=6 %s %s" %
                       (self.target, _q(cmd)), timeout)
        adb_path_prepend()
        if run("adb -s %s get-state" % self.serial).stdout.strip() != 'device':
            run("adb connect %s" % self.serial)
            if run("adb -s %s get-state" % self.serial).stdout.strip() != 'device':
                raise RuntimeError("adb 设备不在线: %s" % self.serial)
        return run("adb -s %s shell %s" % (self.serial, _q(cmd)), timeout)

    def push(self, local, remote):
        if self.via_ssh:
            r = run("scp -o StrictHostKeyChecking=no %s %s:%s" % (_q(local), self.target, _q(remote)), 120)
            return r.returncode == 0
        adb_path_prepend()
        r = run("adb -s %s push %s %s" % (self.serial, _q(local), _q(remote)), 120)
        return r.returncode == 0

    # ---- 部署 ----
    def prepare(self):
        is_android = 'release' in self.sh("getprop ro.build.version.release 2>/dev/null").stdout
        # Debian(root) 用 /root/rvs；Android 只能写 /data/local/tmp
        self.libroot = '/root/rvs' if not is_android else '/data/local/tmp/rvs'
        self.dir = self.libroot + '/perfprobe'
        self.sh("mkdir -p %s" % self.dir)
        if 'ok' in self.sh("test -x %s/perf_run && echo ok" % self.dir).stdout:
            pass
        elif is_android:
            bin_local = os.path.join(TOOLS, 'bin', 'perf_run_android_arm64')
            if not os.path.exists(bin_local):
                raise RuntimeError("perf_run 未编译：先跑 tools/build_perf_run.sh")
            if not self.push(bin_local, "%s/perf_run" % self.dir):
                raise RuntimeError("推 perf_run 失败")
            self.sh("chmod 755 %s/perf_run" % self.dir)
        else:
            gcc = self.sh("command -v gcc || echo none").stdout.strip()
            if gcc == 'none':
                raise RuntimeError("板上无 gcc：apt install -y gcc 后重试")
            for f in ('perf_run.c', '../engine/third_party/rknn_api.h'):
                if not self.push(os.path.join(TOOLS, f), "%s/%s" % (self.dir, os.path.basename(f))):
                    raise RuntimeError("推 %s 失败" % f)
            self.sh("cd %s && gcc -O2 -o perf_run perf_run.c -I. "
                    "-L%s -Wl,-rpath,%s -lrknnrt 2>&1" % (self.dir, self.libroot, self.libroot), 180)
            if 'ok' not in self.sh("test -x %s/perf_run && echo ok" % self.dir).stdout:
                raise RuntimeError("板上编译 perf_run 失败（板上 gcc 缺依赖？）")
        # 库：libroot 下找 librknnrt*.so，规范名软链（librknnrt25.so 等非标名）
        self.sh("cd %s && ls librknnrt*.so 2>/dev/null | head -1 | xargs -r -I{} ln -sf {} librknnrt.so" % self.libroot)
        if 'ok' not in self.sh("test -f %s/librknnrt.so && echo ok" % self.libroot).stdout:
            raise RuntimeError("板上 %s 无 librknnrt*.so（先推运行库）" % self.libroot)

    def run_mask(self, model, loops, mask, timeout):
        out = self.sh("cd %s && LD_LIBRARY_PATH=%s ./perf_run %s %d %d" %
                      (self.dir, self.libroot, model, loops, mask), timeout)
        return parse_json_line(out.stdout + out.stderr)

    def read_freq(self):
        return freq_snapshot(lambda c: self.sh(c).stdout)

    def write_gov(self, dev, val):
        self.sh("echo %s > /sys/class/devfreq/%s/governor" % (val, dev))


def _q(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


def freq_snapshot(do):
    """读 devfreq 现状：{dev: {gov, cur, min, max}}——npu/dmc 都要。"""
    out = {}
    listing = do("ls -1 /sys/class/devfreq/ 2>/dev/null")
    for dev in listing.split():
        if 'npu' not in dev and dev != 'dmc':
            continue
        def rd(f):
            return do("cat /sys/class/devfreq/%s/%s 2>/dev/null" % (dev, f)).strip()
        out[dev] = {'gov': rd('governor'), 'cur': rd('cur_freq'),
                    'min': rd('min_freq'), 'max': rd('max_freq')}
    return out


def main():
    if len(sys.argv) < 6:
        print(json.dumps({'ok': False, 'error': 'usage: perf_probe.py <model.rknn> <target> <masks> <loops> <fixfreq>'}))
        return
    model_local = norm_path(sys.argv[1])
    target, masks_arg, loops, fixfreq = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] == '1'
    masks = [int(m) for m in masks_arg.split(',') if m.strip()]
    model = os.path.basename(model_local)

    chan = Chan(target)
    rows, freq_before, freq_after = [], None, None
    try:
        log("连接 %s（%s 传输）…" % (target, 'ssh' if chan.via_ssh else 'adb'))
        chan.prepare()
        log("部署完成，推模型 %s（%.1f MB）…" % (model, os.path.getsize(model_local) / 1e6))
        if not chan.push(model_local, "%s/model.rknn" % chan.dir):
            raise RuntimeError("推模型失败")
        freq_before = chan.read_freq()
        log("频率现状: %s" % json.dumps(freq_before))
        if fixfreq:
            for dev in freq_before:
                chan.write_gov(dev, 'performance')
            snap = chan.read_freq()
            log("已锁 performance: %s" % json.dumps({k: v['gov'] for k, v in snap.items()}))
        for mask in masks:
            log("core_mask=0x%x 跑 %d 帧…" % (mask, loops))
            timeout = max(120, loops * 15 + 60)
            row = chan.run_mask('model.rknn', loops, mask, timeout)
            if not row or not row.get('ok'):
                rows.append({'mask': mask, 'error': (row or {}).get('stage', 'no-json') if row else 'perf_run 无输出'})
                log("core_mask=0x%x 失败：%s" % (mask, rows[-1]['error']))
                if row and row.get('stage') in ('set_core_mask',):
                    break   # 该板不支持此 mask，后续同族 mask 直接放弃
                continue
            rows.append(row)
            log("core_mask=0x%x: set=%.1f run=%.1f get=%.1f e2e=%.1f ms (%.1f fps)" %
                (mask, row['set_ms'], row['run_ms'], row['get_ms'], row['e2e_ms'], row['fps']))
        print(json.dumps({'ok': any('set_ms' in r for r in rows), 'target': target,
                          'model': model, 'loops': loops, 'fixfreq': fixfreq,
                          'freq_locked': freq_before, 'rows': rows}, ensure_ascii=False))
    except Exception as e:                                          # noqa: BLE001
        print(json.dumps({'ok': False, 'error': str(e), 'rows': rows}, ensure_ascii=False))
    finally:
        if fixfreq and freq_before:
            for dev, info in freq_before.items():
                if info.get('gov') and info['gov'] != 'performance':
                    try:
                        chan.write_gov(dev, info['gov'])
                    except Exception:                              # noqa: BLE001
                        pass


if __name__ == '__main__':
    main()
