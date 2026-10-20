# Studio server 的行为规格（零依赖 stdlib http.server；不起 adb 的端点直接打）
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rvs import studio_server  # noqa: E402


def _get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_endpoints(tmp_path):
    # 隔离：项目目录指向仓库真实 projects/（只读端点，无副作用）
    port = 18999
    th = threading.Thread(
        target=lambda: studio_server.serve(port), daemon=True)
    th.start()
    threading.Event().wait(0.5)

    # ① 首页可访问且含标题
    st, body = _get(port, "/")
    assert st == 200 and "RKVision Studio".encode() in body

    # ② /api/status：有 projects 列表（devices 可能为空，不依赖 adb 成功）
    st, body = _get(port, "/api/status")
    j = json.loads(body)
    assert st == 200 and isinstance(j["projects"], list) and "helmet_gold" in j["projects"]

    # ③ /api/results：不存在的工程 → 空 files
    st, body = _get(port, "/api/results?project=no_such")
    assert st == 200 and json.loads(body)["files"] == []

    # ④ /api/file：目录穿越被拒
    st, body = _get(port, "/api/file?project=x&name=" + "../pipeline.json")
    assert st == 404

    # ⑤ 未知路径 404
    st, body = _get(port, "/api/nope")
    assert st == 404
