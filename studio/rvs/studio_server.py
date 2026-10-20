"""RKVision Studio M1-minimal：零依赖 Web 服务（stdlib http.server）。

页面: GET /                → studio/www/index.html
API:
  GET  /api/status                 → {devices, projects}
  POST /api/deploy                 → {project, serial?}   打包并推板（含引擎/runtime）
  POST /api/run                    → {project, serial?}   部署+执行+拉结果（同步，图片目录工程用）
  POST /api/preview/start          → {project, serial?, device_port, local_port}
                                   → 推板+后台启动引擎+adb forward
  POST /api/preview/stop           → {serial?}            停引擎+删 forward
  GET  /api/results?project=xxx    → output 目录文件列表
  GET  /api/file?project=xxx&name=yyy → 结果文件（图/JSON）
预览流不代理：浏览器直连 adb forward 出来的 http://localhost:<local_port>/stream。
"""
import json
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WWW = Path(__file__).resolve().parents[1] / "www"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 脚本直跑时补包路径

from rvs.adb import AdbError, AdbHost  # noqa: E402
from rvs.deploy import deploy, run_on_board  # noqa: E402

_lock = threading.Lock()


def _adb(serial=None):
    return AdbHost(bin_path=None, serial=serial)


def _projects():
    root = ROOT / "projects"
    return sorted(p.name for p in root.iterdir() if (p / "pipeline.json").is_file())


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def log_message(self, fmt, *args):  # 静音访问日志
        pass

    # ---- GET ----
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            data = (WWW / "index.html").read_bytes()
            self._send(200, data, "text/html; charset=utf-8")
            return
        if self.path == "/api/status":
            devices = []
            adb_error = None
            try:
                devices = _adb().devices()
            except Exception as e:  # noqa: BLE001  adb 缺失/不在线都不致命
                adb_error = f"{type(e).__name__}: {e}"
            self._send(200, {"devices": devices, "projects": _projects(),
                             "adb_error": adb_error})
            return
        if self.path.startswith("/api/results"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            project = (q.get("project") or [""])[0]
            out = ROOT / "projects" / project / "output"
            if not out.is_dir():
                self._send(200, {"files": []})
                return
            files = sorted((f.name, f.stat().st_mtime) for f in out.iterdir())
            self._send(200, {"files": [f[0] for f in files]})
            return
        if self.path.startswith("/api/file"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            project = (q.get("project") or [""])[0]
            name = (q.get("name") or [""])[0]
            f = ROOT / "projects" / project / "output" / name
            if not f.is_file() or ".." in name:
                self._send(404, {"error": "not found"})
                return
            ctype = "image/jpeg" if name.endswith(".jpg") else "application/json"
            self._send(200, f.read_bytes(), ctype)
            return
        self._send(404, {"error": "not found"})

    # ---- POST ----
    def do_POST(self):
        try:
            body = self._body()
            serial = body.get("serial") or None
            adb = _adb(serial)
            if self.path == "/api/deploy":
                staging = deploy(ROOT / "projects" / body["project"], adb)
                self._send(200, {"ok": True, "staged": staging.name})
            elif self.path == "/api/run":
                summary = run_on_board(ROOT / "projects" / body["project"], adb,
                                       body.get("remote_root", "/data/local/tmp/rvs"))
                self._send(200, {"ok": True, "summary": summary})
            elif self.path == "/api/preview/start":
                project = body["project"]
                device_port = int(body.get("device_port", 8000))
                local_port = int(body.get("local_port", 8000))
                remote_root = body.get("remote_root", "/data/local/tmp/rvs")
                deploy(ROOT / "projects" / project, adb, remote_root)
                adb.shell(f"pkill rvs_engine 2>/dev/null; true")
                run_cmd = (f"cd {remote_root}/{project} && nohup sh -c "
                           f"'LD_LIBRARY_PATH={remote_root} {remote_root}/rvs_engine .' "
                           f"> {remote_root}/engine.log 2>&1 &")
                adb.shell(run_cmd)
                subprocess.run([adb.bin_path] + (["-s", serial] if serial else []) +
                               ["forward", f"tcp:{local_port}", f"tcp:{device_port}"],
                               capture_output=True, timeout=15, check=True)
                self._send(200, {"ok": True,
                                 "stream_url": f"http://localhost:{local_port}/stream"})
            elif self.path == "/api/preview/stop":
                adb.shell("pkill rvs_engine 2>/dev/null; true")
                subprocess.run([adb.bin_path] + (["-s", serial] if serial else []) +
                               ["forward", "--remove", f"tcp:{body.get('local_port', 8000)}"],
                               capture_output=True, timeout=15)
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
        except AdbError as e:
            self._send(500, {"ok": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})


def serve(port=8800):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"RKVision Studio: http://127.0.0.1:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    import sys
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8800)
