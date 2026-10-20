import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rvs.adb import AdbError, AdbHost  # noqa: E402
from rvs.deploy import deploy, run_on_board  # noqa: E402
from rvs.pack import pack  # noqa: E402


def make_fake_project(tmp: Path) -> Path:
    proj = tmp / "helmet_gold"
    (proj / "models").mkdir(parents=True)
    (proj / "assets" / "test_images").mkdir(parents=True)
    (proj / "models" / "helmet_detect_my.rknn").write_bytes(b"FAKE_RKNN")
    (proj / "assets" / "test_images" / "a.jpg").write_bytes(b"FAKE_JPG")
    (proj / "pipeline.json").write_text(json.dumps({
        "version": 1, "name": "helmet_gold",
        "models": {"det": {"family": "yolov5-det", "file": "models/helmet_detect_my.rknn"}},
        "pipeline": [],
    }))
    return proj


def test_pack_stages_expected_files(tmp_path):
    proj = make_fake_project(tmp_path)
    staging = pack(proj)
    assert (staging / "pipeline.json").is_file()
    assert (staging / "models" / "helmet_detect_my.rknn").read_bytes() == b"FAKE_RKNN"
    assert (staging / "assets" / "test_images" / "a.jpg").is_file()


class RecordingAdb(AdbHost):
    def __init__(self):
        super().__init__(bin_path="adb", serial="SER1")
        self.calls = []

    def push(self, local, remote):
        self.calls.append(("push", str(local), remote))

    def shell(self, cmd):
        self.calls.append(("shell", cmd))
        return ""

    def pull(self, remote, local):
        self.calls.append(("pull", remote, str(local)))


def test_deploy_push_sequence(tmp_path):
    proj = make_fake_project(tmp_path)
    engine = tmp_path / "rvs_engine"
    engine.write_bytes(b"ELF")
    lib = tmp_path / "librknnrt.so"
    lib.write_bytes(b"SO")

    adb = RecordingAdb()
    deploy(proj, adb, remote_root="/data/local/tmp/rvs",
           engine_bin=engine, librknnrt=lib)

    remotes = [c[2] for c in adb.calls if c[0] == "push"]
    assert "/data/local/tmp/rvs/helmet_gold" in remotes
    assert "/data/local/tmp/rvs/rvs_engine" in remotes
    assert "/data/local/tmp/rvs/librknnrt.so" in remotes
    assert any(c[0] == "shell" and "chmod" in c[1] for c in adb.calls)


def test_run_pulls_output(tmp_path):
    proj = make_fake_project(tmp_path)
    engine = tmp_path / "rvs_engine"
    engine.write_bytes(b"ELF")
    lib = tmp_path / "librknnrt.so"
    lib.write_bytes(b"SO")

    adb = RecordingAdb()
    summary = run_on_board(proj, adb, remote_root="/data/local/tmp/rvs",
                           engine_bin=engine, librknnrt=lib)
    # 拉回 output 目录到本地工程包旁
    pulls = [c for c in adb.calls if c[0] == "pull"]
    assert pulls, "run_on_board must pull output dir"
    assert pulls[0][1] == "/data/local/tmp/rvs/helmet_gold/output"
    # shell 里必须以工程目录为 CWD 启动引擎，且 LD_LIBRARY_PATH 指向 remote_root
    shells = [c[1] for c in adb.calls if c[0] == "shell"]
    assert any("rvs_engine" in s and "helmet_gold" in s for s in shells)


def test_adb_timeout_raises_with_serial(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="adb", timeout=1)
    monkeypatch.setattr(subprocess, "run", fake_run)
    adb = AdbHost(bin_path="adb", serial="SER9", timeout_s=1)
    try:
        adb.shell("echo hi")
        raise AssertionError("expected AdbError")
    except AdbError as e:
        assert "SER9" in str(e)
