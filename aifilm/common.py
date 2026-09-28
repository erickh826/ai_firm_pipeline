"""Shared helpers for AI film pipeline CLIs."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def load_simple_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_shot_in_board(root: Path, board: str, shot_id: str) -> dict[str, Any] | None:
    shots_path = resolve(board, root) / "shots.json"
    if not shots_path.exists():
        return None

    cfg = load_json(shots_path)
    for shot in cfg.get("shots", []):
        if shot.get("id") == shot_id:
            return shot
    return None


def version_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    version = 0
    if "_v" in stem:
        suffix = stem.rsplit("_v", 1)[-1]
        if suffix.isdigit():
            version = int(suffix)
    return version, path.name


def find_keyframe(job: dict[str, Any], shot: dict[str, Any] | None, image_dir: Path, root: Path) -> Path:
    if job.get("image"):
        image = resolve(job["image"], root) if "/" in job["image"] else image_dir / job["image"]
        if image.exists():
            return image
        raise FileNotFoundError(f"Configured image not found: {image}")

    if shot and shot.get("selected_image"):
        image = image_dir / shot["selected_image"]
        if image.exists():
            return image

    shot_id = job["id"]
    patterns = [
        f"{shot_id}_v*.jpg",
        f"{shot_id}_v*.jpeg",
        f"{shot_id}_v*.png",
        f"{shot_id}_v*.webp",
        f"*{shot_id}*.jpg",
        f"*{shot_id}*.jpeg",
        f"*{shot_id}*.png",
        f"*{shot_id}*.webp",
    ]

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(image_dir.glob(pattern))

    unique = sorted(set(matches), key=version_sort_key, reverse=True)
    if unique:
        return unique[0]

    raise FileNotFoundError(f"No keyframe image found for shot {shot_id} in {image_dir}")


def download_file(url: str, output_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=300) as response:
        with output_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def append_manifest(manifest_path: Path, record: dict[str, Any]) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_media_file(raw: str, image_dir: Path, root: Path) -> Path:
    if raw.startswith("http://") or raw.startswith("https://") or raw.startswith("gs://"):
        raise ValueError(f"Expected a local image path, got remote URI: {raw}")
    path = resolve(raw, root) if "/" in raw else image_dir / raw
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")
    return path
