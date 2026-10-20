"""scrcpy stdout -> ffmpeg stdin 泵（Windows 二进制管道 + 流量计数）

可执行路径默认走 PATH，也可用环境变量指定 Windows 侧二进制：
    RVS_SCRCPY=C:\\path\\to\\scrcpy.exe  RVS_FFMPEG=C:\\path\\to\\ffmpeg.exe
"""
import os
import subprocess
import sys
import threading
import time

SCRCPY = os.environ.get("RVS_SCRCPY", "scrcpy")
FFMPEG = os.environ.get("RVS_FFMPEG", "ffmpeg")
SERIAL = sys.argv[1] if len(sys.argv) > 1 else ""
RTSP = sys.argv[2] if len(sys.argv) > 2 else "rtsp://127.0.0.1:8554/hdmiin"
if not SERIAL:
    sys.exit("usage: relay.py <adb_serial> [rtsp_url]")

sc = subprocess.Popen(
    [SCRCPY, "-s", SERIAL, "--no-playback", "--no-audio",
     "--video-codec=h264", "--record=-", "--record-format=mkv",
     "--video-bit-rate=4M"],
    stdout=subprocess.PIPE, stderr=open("scrcpy_relay.log", "w"))

ff = subprocess.Popen(
    [FFMPEG, "-loglevel", "warning", "-f", "matroska", "-i", "pipe:0",
     "-c:v", "copy", "-f", "rtsp", RTSP],
    stdin=subprocess.PIPE, stderr=open("ff_relay.log", "w"))

total = 0
last = 0
while True:
    chunk = sc.stdout.read(65536)
    if not chunk:
        print("scrcpy stdout closed", flush=True)
        break
    total += len(chunk)
    if total - last >= 1_000_000:
        print(f"pumped {total} bytes", flush=True)
        last = total
    try:
        ff.stdin.write(chunk)
        ff.stdin.flush()
    except BrokenPipeError:
        print("ffmpeg stdin closed", flush=True)
        break
ff.terminate()
