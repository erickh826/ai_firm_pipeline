"""
fal.ai image-to-video generator.

Reads a video_jobs.json file, uploads selected keyframes, downloads MP4 files,
and appends a manifest.jsonl record for shot tracking.

Usage:
    pip install fal-client
    # Put FAL_KEY=your_api_key_here in a .env file in your project root (kept out of Git),
    # or export it directly.

    # Validate paths and the selected jobs without an API call:
    python -m aifilm.videos --jobs board1/video_jobs.json --dry-run

    # Generate every queued job after keyframes are approved:
    python -m aifilm.videos --jobs board1/video_jobs.json

    # Generate only selected shots during review:
    python -m aifilm.videos --jobs board1/video_jobs.json --only 1.1 1.6

Relative paths in the jobs file are resolved against --root (default: current directory).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
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


# Load .env from the current working directory (the project root), so API keys
# stay out of the repo. Users can also export FAL_KEY directly.
load_simple_env(Path.cwd() / ".env")


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "kling-o3": {
        "endpoint": "fal-ai/kling-video/o3/standard/reference-to-video",
        "image_arg": "start_image_url",
        "end_image_arg": "end_image_url",
        "duration_min": 3,
        "duration_max": 15,
        "allowed_arguments": {"duration", "aspect_ratio", "generate_audio"},
        "defaults": {
            "duration": 5,
            "aspect_ratio": "16:9",
            "generate_audio": False,
        },
    },
    "seedance": {
        "endpoint": "bytedance/seedance-2.5/image-to-video",
        "image_arg": "image_url",
        "end_image_arg": "end_image_url",
        "duration_min": 4,
        "duration_max": 30,
        "allowed_arguments": {"duration", "resolution", "generate_audio", "camera_fixed"},
        "defaults": {
            "duration": 5,
            "resolution": "720p",
            "generate_audio": False,
        },
    },
    "seedance-text": {
        "endpoint": "bytedance/seedance-2.5/text-to-video",
        "requires_image": False,
        "duration_min": 4,
        "duration_max": 30,
        "allowed_arguments": {"duration", "resolution", "aspect_ratio", "generate_audio", "bitrate_mode"},
        "defaults": {
            "duration": 5,
            "resolution": "720p",
            "aspect_ratio": "16:9",
            "generate_audio": False,
        },
    },
    "minimax": {
        "endpoint": "fal-ai/minimax/hailuo-02/pro/image-to-video",
        "image_arg": "image_url",
        "duration_min": 5,
        "duration_max": 10,
        "defaults": {
            "duration": 5,
            "resolution": "1080p",
        },
    },
}


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


def normalize_duration(model_key: str, duration: int | str) -> int | str:
    preset = MODEL_PRESETS[model_key]
    if duration == "auto":
        return duration

    value = int(duration)
    if "duration_values" in preset and value not in preset["duration_values"]:
        allowed = sorted(preset["duration_values"])
        raise ValueError(f"{model_key} duration must be one of {allowed}, got {value}")
    if "duration_min" in preset and value < preset["duration_min"]:
        raise ValueError(f"{model_key} duration must be >= {preset['duration_min']}, got {value}")
    if "duration_max" in preset and value > preset["duration_max"]:
        raise ValueError(f"{model_key} duration must be <= {preset['duration_max']}, got {value}")
    return value


def build_arguments(
    model_key: str,
    job: dict[str, Any],
    defaults: dict[str, Any],
    image_url: str | None,
    end_image_url: str | None,
    shot: dict[str, Any] | None,
    elements: list[dict[str, Any]] | None = None,
    image_urls: list[str] | None = None,
) -> dict[str, Any]:
    preset = MODEL_PRESETS[model_key]
    model_defaults = preset.get("defaults", {})
    prompt = job.get("prompt") or job.get("video_prompt")
    if not prompt and shot:
        prompt = shot.get("veo_prompt") or shot.get("video_prompt") or shot.get("prompt")
    if not prompt:
        raise ValueError(f"Missing prompt for shot {job['id']}")

    args: dict[str, Any] = {}
    args.update(model_defaults)
    args.update(defaults)
    args.update(job.get("arguments", {}))

    # Common fal options are usually kept at the top level in video_jobs.json
    # so the job file stays readable. Copy them into the API arguments unless
    # an explicit job["arguments"] value already overrides them.
    for key in [
        "duration",
        "resolution",
        "aspect_ratio",
        "negative_prompt",
        "generate_audio",
        "seed",
    ]:
        if key in job and key not in job.get("arguments", {}):
            args[key] = job[key]

    if "allowed_arguments" in preset:
        allowed = preset["allowed_arguments"]
        args = {key: value for key, value in args.items() if key in allowed}

    args["prompt"] = prompt
    if preset.get("requires_image", True):
        if not image_url:
            raise ValueError(f"{model_key} requires a keyframe image")
        args[preset["image_arg"]] = image_url

    if end_image_url and preset.get("end_image_arg"):
        args[preset["end_image_arg"]] = end_image_url

    if elements:
        args["elements"] = elements
    if image_urls:
        args["image_urls"] = image_urls

    if "duration" in args:
        args["duration"] = normalize_duration(model_key, args["duration"])

    # Avoid sending unsupported generic fields from the job/default layer.
    for key in ["model", "image", "end_image", "id", "output_name"]:
        args.pop(key, None)

    return args


def extract_video_url(result: dict[str, Any]) -> str:
    candidates = [
        result.get("video"),
        result.get("output"),
        result.get("file"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("url"):
            return candidate["url"]
        if isinstance(candidate, str) and candidate.startswith("http"):
            return candidate

    videos = result.get("videos")
    if isinstance(videos, list) and videos:
        first = videos[0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"]
        if isinstance(first, str) and first.startswith("http"):
            return first

    raise RuntimeError(f"Could not find video URL in fal result: {json.dumps(result)[:800]}")


def download_file(url: str, output_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=300) as response:
        with output_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def upload_file_ascii_safe(fal_client: Any, path: Path) -> str:
    """Upload files through an ASCII-only temp path to avoid fal client path bugs."""
    try:
        str(path).encode("ascii")
        return fal_client.upload_file(str(path))
    except UnicodeEncodeError:
        suffix = path.suffix if path.suffix else ".bin"
        with tempfile.TemporaryDirectory(prefix="fal_upload_") as tmp_dir:
            temp_path = Path(tmp_dir) / f"keyframe{suffix}"
            shutil.copy2(path, temp_path)
            return fal_client.upload_file(str(temp_path))


def resolve_media_path(fal_client: Any, raw: str, image_dir: Path, root: Path) -> str:
    """Resolve a local image path (or pass through an http URL) to a fal media URL."""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    path = resolve(raw, root) if "/" in raw else image_dir / raw
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")
    return upload_file_ascii_safe(fal_client, path)


def build_elements(
    fal_client: Any,
    elements_config: list[dict[str, Any]],
    image_dir: Path,
    root: Path,
) -> list[dict[str, Any]]:
    """Turn job 'elements' entries into Kling reference-to-video element payloads."""
    result: list[dict[str, Any]] = []
    for elem in elements_config:
        built: dict[str, Any] = {}
        frontal = elem.get("frontal_image")
        if frontal:
            built["frontal_image_url"] = resolve_media_path(fal_client, frontal, image_dir, root)
        elif elem.get("frontal_image_url"):
            built["frontal_image_url"] = elem["frontal_image_url"]

        refs = elem.get("reference_images") or []
        if refs:
            built["reference_image_urls"] = [
                resolve_media_path(fal_client, r, image_dir, root) for r in refs
            ]
        elif elem.get("reference_image_urls"):
            built["reference_image_urls"] = elem["reference_image_urls"]

        if elem.get("video_url"):
            built["video_url"] = elem["video_url"]
        if elem.get("voice_id"):
            built["voice_id"] = elem["voice_id"]

        if built:
            result.append(built)
    return result


def append_manifest(manifest_path: Path, record: dict[str, Any]) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(jobs_path: Path, only: list[str] | None, dry_run: bool, overwrite: bool, root: Path) -> None:
    fal_client = None
    if not dry_run:
        try:
            import fal_client as fal_client_module
            fal_client = fal_client_module
        except ImportError:
            sys.exit("Missing dependency: pip install fal-client")

    cfg = load_json(jobs_path)
    board = cfg["board"]
    image_dir = resolve(cfg["image_dir"], root)
    output_dir = resolve(cfg["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = cfg.get("defaults", {})
    jobs = cfg.get("jobs", [])
    if only:
        jobs = [job for job in jobs if job.get("id") in only]

    if not jobs:
        print("No jobs selected.")
        return

    if not dry_run and not os.environ.get("FAL_KEY"):
        sys.exit("Error: FAL_KEY environment variable is not set.")

    print(f"\n{'=' * 60}")
    print(f"  Jobs   : {jobs_path}")
    print(f"  Board  : {board}")
    print(f"  Output : {output_dir}")
    print(f"  Shots  : {', '.join(job['id'] for job in jobs)}")
    print(f"{'=' * 60}\n")

    manifest_path = output_dir / "manifest.jsonl"

    for job in jobs:
        shot_id = job["id"]
        model_key = job.get("model", defaults.get("model", "kling-o3"))
        if model_key not in MODEL_PRESETS:
            print(f"  [SKIP] {shot_id} unknown model preset: {model_key}")
            continue

        shot = find_shot_in_board(root, board, shot_id)
        preset = MODEL_PRESETS[model_key]
        keyframe: Path | None = None
        if preset.get("requires_image", True):
            try:
                keyframe = find_keyframe(job, shot, image_dir, root)
            except Exception as exc:
                print(f"  [SKIP] {shot_id} {exc}")
                continue

        out_name = job.get("output_name") or f"{shot_id}_{model_key}.mp4"
        out_path = output_dir / out_name
        if out_path.exists() and not overwrite:
            print(f"  [SKIP] {out_path.name} already exists")
            continue

        end_keyframe = None
        if job.get("end_image"):
            end_keyframe = resolve(job["end_image"], root) if "/" in job["end_image"] else image_dir / job["end_image"]
            if not end_keyframe.exists():
                print(f"  [SKIP] {shot_id} end_image not found: {end_keyframe}")
                continue

        endpoint = job.get("endpoint") or preset["endpoint"]

        if dry_run:
            print(f"  [DRY] {shot_id} -> {model_key}")
            if keyframe:
                print(f"        image: {keyframe.relative_to(root)}")
            else:
                print("        image: none (text-to-video)")
            if job.get("elements"):
                print(f"        elements: {[e.get('frontal_image', e.get('frontal_image_url')) for e in job['elements']]}")
            if job.get("image_urls"):
                print(f"        image_urls: {job['image_urls']}")
            print(f"        out:   {out_path.relative_to(root)}")
            continue

        print(f"  [GEN] {shot_id} -> {model_key} ...", end=" ", flush=True)
        started = time.time()
        try:
            if fal_client is None:
                raise RuntimeError("fal_client is not loaded")
            image_url = upload_file_ascii_safe(fal_client, keyframe) if keyframe else None
            end_image_url = upload_file_ascii_safe(fal_client, end_keyframe) if end_keyframe else None
            elements = build_elements(fal_client, job.get("elements") or [], image_dir, root)
            ref_image_urls = [
                resolve_media_path(fal_client, u, image_dir, root)
                for u in (job.get("image_urls") or [])
            ]
            arguments = build_arguments(
                model_key, job, defaults, image_url, end_image_url, shot,
                elements=elements, image_urls=ref_image_urls,
            )

            result = fal_client.subscribe(endpoint, arguments=arguments, with_logs=True)
            video_url = extract_video_url(result)
            download_file(video_url, out_path)

            elapsed = time.time() - started
            record = {
                "shot_id": shot_id,
                "model": model_key,
                "endpoint": endpoint,
                "keyframe": str(keyframe.relative_to(root)) if keyframe else None,
                "output": str(out_path.relative_to(root)),
                "video_url": video_url,
                "duration": arguments.get("duration"),
                "resolution": arguments.get("resolution"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(elapsed, 1),
            }
            append_manifest(manifest_path, record)
            print(f"done ({elapsed:.1f}s) -> {out_path.name}")
        except Exception as exc:
            print(f"FAILED - {exc}")

    print("\nAll done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate storyboard videos via fal.ai")
    parser.add_argument("--jobs", required=True, help="Path to video_jobs.json")
    parser.add_argument(
        "--root",
        default=".",
        help="Base directory for relative paths in the jobs file (default: current directory).",
    )
    parser.add_argument("--only", nargs="+", metavar="ID", help="Only generate selected shot IDs.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected jobs without calling fal.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if output MP4 exists.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    jobs_path = resolve(args.jobs, root)
    if not jobs_path.exists():
        sys.exit(f"jobs file not found: {jobs_path}")

    run(jobs_path=jobs_path, only=args.only, dry_run=args.dry_run,
        overwrite=args.overwrite, root=root)


if __name__ == "__main__":
    main()
