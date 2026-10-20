"""adb 通道抽象。默认宿主可配：RVS_ADB 环境变量 → Windows adb.exe（WSL interop）或 PATH 里的 adb。"""
import os
import shlex
import subprocess


class AdbError(RuntimeError):
    pass


class AdbHost:
    def __init__(self, bin_path=None, serial=None, timeout_s=30):
        self.bin_path = bin_path or os.environ.get("RVS_ADB", "adb")
        self.serial = serial
        self.timeout_s = timeout_s

    def _base(self):
        cmd = [self.bin_path]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def _run(self, args, timeout_s=None, check=True):
        cmd = self._base() + args
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout_s or self.timeout_s)
        except subprocess.TimeoutExpired as e:
            raise AdbError(f"adb timeout after {e.timeout}s (serial={self.serial}): {args}") from e
        if check and proc.returncode != 0:
            raise AdbError(f"adb failed rc={proc.returncode} (serial={self.serial}): "
                           f"{args}\nstderr: {proc.stderr.strip()}")
        return proc.stdout

    def push(self, local, remote):
        self._run(["push", str(local), remote])

    def shell(self, cmd):
        return self._run(["shell", cmd]).rstrip("\n")

    def pull(self, remote, local):
        self._run(["pull", remote, str(local)])

    def devices(self):
        out = self._run(["devices"])
        lines = [l for l in out.splitlines()[1:] if l.strip()]
        return [l.split("\t")[0] for l in lines]

    def quote(self, s):
        return shlex.quote(s)
