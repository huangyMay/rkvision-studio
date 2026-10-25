"""工程包打包：把工程目录的 pipeline.json + 其余资源暂存到 build-staging/<name>/。

只排除"运行产物"（output/ng_images 等）和隐藏目录；不再硬编码 models/assets ——
工程完全可以引用 imgs/ 之类的自定义目录（仓库里的 smoke/preview 就是 imgs/）。
"""
import json
import shutil
from pathlib import Path

# 板上重新生成、不需要回传的资源；避免把上一轮结果当输入推上去
SKIP_DIRS = {"output", "ng_images", "build", "build-staging", "__pycache__"}


def pack(project_dir: Path, staging_root: Path = None) -> Path:
    project_dir = Path(project_dir)
    spec = json.loads((project_dir / "pipeline.json").read_text())
    name = spec.get("name") or project_dir.name
    if spec.get("version") != 1:
        raise SystemExit(f"pack: unsupported pipeline.json version: {spec.get('version')}")

    staging_root = Path(staging_root) if staging_root else project_dir.parents[1] / "build-staging"
    staging = staging_root / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copy2(project_dir / "pipeline.json", staging / "pipeline.json")
    for entry in sorted(project_dir.iterdir()):
        if entry.name == "pipeline.json" or entry.name.startswith("."):
            continue
        if entry.is_dir():
            if entry.name in SKIP_DIRS:
                continue
            shutil.copytree(entry, staging / entry.name)
        elif entry.is_file():
            shutil.copy2(entry, staging / entry.name)
    return staging
