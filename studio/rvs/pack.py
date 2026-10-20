"""工程包打包：把工程目录的 pipeline.json + models/ + assets/ 暂存到 build-staging/<name>/。"""
import json
import shutil
from pathlib import Path


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
    for sub in ("models", "assets"):
        src = project_dir / sub
        if src.is_dir():
            shutil.copytree(src, staging / sub)
    return staging
