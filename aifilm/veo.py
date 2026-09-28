"""
Google Veo video generator.

Drop-in replacement for ``aifilm.videos`` (fal.ai Seedance / Kling) when you
want Google's stack: image-to-video with native audio and lip sync.

Usage:
    python -m aifilm.veo --jobs board1/video_jobs.json --dry-run
    python -m aifilm.veo --jobs board1/video_jobs.json
    python -m aifilm.veo --jobs board1/video_jobs.json --only 1.1 1.6

Auth (pick one):

    Gemini API
        GEMINI_API_KEY=...          (or GOOGLE_API_KEY)

    Vertex AI / Google Cloud
        GOOGLE_GENAI_USE_VERTEXAI=true
        GOOGLE_CLOUD_PROJECT=your-project
        GOOGLE_CLOUD_LOCATION=us-central1
        # plus Application Default Credentials (gcloud auth application-default login)

Relative paths are resolved against --root (default: current directory).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aifilm.common import (
    append_manifest,
    find_keyframe,
    find_shot_in_board,
    load_json,
    load_simple_env,
    resolve,
    resolve_media_file,
)


load_simple_env(Path.cwd() / ".env")


# Gemini Developer API uses preview ids; Vertex uses GA / numbered ids.
# ``vertex_model`` is used when GOOGLE_GENAI_USE_VERTEXAI is on.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "veo": {
        "gemini_model": "veo-3.1-generate-preview",
        "vertex_model": "veo-3.1-generate-001",
        "duration_values": {4, 6, 8},
        "requires_image": True,
        "supports_last_frame": True,
        "supports_reference_images": True,
        "supports_extension": True,
        "defaults": {
            "duration": 8,
            "aspect_ratio": "16:9",
            "resolution": "720p",
            "generate_audio": True,
            "person_generation": "allow_adult",
        },
    },
    "veo-fast": {
        "gemini_model": "veo-3.1-fast-generate-preview",
        "vertex_model": "veo-3.1-fast-generate-001",
        "duration_values": {4, 6, 8},
        "requires_image": True,
        "supports_last_frame": True,
        "supports_reference_images": True,
        "supports_extension": True,
        "defaults": {
            "duration": 8,
            "aspect_ratio": "16:9",
            "resolution": "720p",
            "generate_audio": True,
            "person_generation": "allow_adult",
        },
    },
    "veo-lite": {
        "gemini_model": "veo-3.1-lite-generate-preview",
        "vertex_model": "veo-3.1-lite-generate-001",
        "duration_values": {4, 6, 8},
        "requires_image": True,
        "supports_last_frame": True,
        "supports_reference_images": False,
        "supports_extension": False,
        "defaults": {
            "duration": 8,
            "aspect_ratio": "16:9",
            "resolution": "720p",
            "generate_audio": True,
            "person_generation": "allow_adult",
        },
    },
    "veo-text": {
        "gemini_model": "veo-3.1-generate-preview",
        "vertex_model": "veo-3.1-generate-001",
        "duration_values": {4, 6, 8},
        "requires_image": False,
        "supports_last_frame": False,
        "supports_reference_images": True,
        "supports_extension": True,
        "defaults": {
            "duration": 8,
            "aspect_ratio": "16:9",
            "resolution": "720p",
            "generate_audio": True,
            "person_generation": "allow_all",
        },
    },
}

# Aliases so existing jobs can switch model:"kling-o3" → model:"veo" without
# extra mapping files, and so "veo-3.1" reads naturally in a jobs JSON.
MODEL_ALIASES = {
    "veo-3": "veo",
    "veo-3.1": "veo",
    "veo-3.1-fast": "veo-fast",
    "veo-3.1-lite": "veo-lite",
    "veo3": "veo",
    "veo3.1": "veo",
}

FAL_ONLY_MODELS = {"kling-o3", "seedance", "seedance-text", "minimax"}

HIGH_RES = {"1080p", "4k", "2160p"}


def using_vertex() -> bool:
    for key in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"):
        flag = os.environ.get(key, "")
        if flag.strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def make_client():
    from google import genai

    if using_vertex():
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise RuntimeError(
                "Vertex AI mode needs GOOGLE_CLOUD_PROJECT "
                "(and GOOGLE_CLOUD_LOCATION, default us-central1)."
            )
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY (Gemini API) or enable Vertex with "
            "GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT."
        )
    return genai.Client(api_key=api_key)


def resolve_model_key(raw: str) -> str:
    key = (raw or "veo").strip()
    if key in FAL_ONLY_MODELS:
        raise ValueError(
            f"{key!r} is a fal.ai preset. Use aifilm-videos for Seedance/Kling, "
            f"or change the job model to 'veo' / 'veo-fast'."
        )
    return MODEL_ALIASES.get(key, key)


def snap_duration(model_key: str, duration: int | str, resolution: str | None) -> int:
    preset = MODEL_PRESETS[model_key]
    allowed = sorted(preset["duration_values"])
    if duration == "auto":
        value = 8
    else:
        value = int(duration)
        if value not in preset["duration_values"]:
            value = min(allowed, key=lambda item: abs(item - value))
    if resolution and resolution.lower() in HIGH_RES and value != 8:
        value = 8
    return value


def quote_dialogue(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[0] in {'"', "'", "“", "‘"} and text[-1] in {'"', "'", "”", "’"}:
        return text
    return f'"{text}"'


def compose_prompt(job: dict[str, Any], shot: dict[str, Any] | None) -> str:
    prompt = job.get("prompt") or job.get("video_prompt") or job.get("veo_prompt")
    if not prompt and shot:
        prompt = shot.get("veo_prompt") or shot.get("video_prompt") or shot.get("prompt")
    if not prompt:
        raise ValueError(f"Missing prompt for shot {job['id']}")

    dialogue = job.get("dialogue") or job.get("speech")
    if not dialogue and shot:
        dialogue = shot.get("dialogue") or shot.get("speech")
    if dialogue:
        line = quote_dialogue(str(dialogue))
        direction = job.get("speaker_direction") or (shot or {}).get("speaker_direction")
        spoken = f"The subject says {line}"
        if direction:
            spoken += f", {direction}"
        spoken += ". Lip-sync the mouth movement precisely to the spoken words."
        if spoken.lower() not in prompt.lower() and line.lower() not in prompt.lower():
            prompt = f"{prompt.rstrip()} {spoken}"
    return prompt


def image_from_path(path: Path):
    from google.genai import types

    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    return types.Image.from_file(location=str(path), mime_type=mime)


def video_from_path(path: Path):
    from google.genai import types

    return types.Video(video_bytes=path.read_bytes(), mime_type="video/mp4")


def collect_reference_images(
    job: dict[str, Any],
    image_dir: Path,
    root: Path,
    preset: dict[str, Any],
) -> list[Any]:
    raw_list = job.get("reference_images") or job.get("image_urls") or []
    if not raw_list:
        return []
    if not preset.get("supports_reference_images"):
        raise ValueError(f"Model does not support reference_images")

    from google.genai import types

    refs: list[Any] = []
    for item in raw_list:
        if isinstance(item, str):
            path = resolve_media_file(item, image_dir, root)
            refs.append(
                types.VideoGenerationReferenceImage(
                    image=image_from_path(path),
                    reference_type="asset",
                )
            )
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Bad reference image entry: {item!r}")
        raw = item.get("image") or item.get("path") or item.get("url")
        if not raw:
            raise ValueError(f"Reference image missing path: {item}")
        path = resolve_media_file(raw, image_dir, root)
        refs.append(
            types.VideoGenerationReferenceImage(
                image=image_from_path(path),
                reference_type=item.get("reference_type") or item.get("type") or "asset",
            )
        )
    if len(refs) > 3:
        raise ValueError("Veo accepts at most 3 reference images")
    return refs


def build_config(
    model_key: str,
    job: dict[str, Any],
    defaults: dict[str, Any],
    last_frame_image: Any | None,
    reference_images: list[Any] | None,
) -> Any:
    from google.genai import types

    preset = MODEL_PRESETS[model_key]
    model_defaults = dict(preset.get("defaults", {}))
    args: dict[str, Any] = {}
    args.update(model_defaults)
    args.update(defaults)
    args.update(job.get("arguments", {}))

    for key in [
        "duration",
        "resolution",
        "aspect_ratio",
        "negative_prompt",
        "generate_audio",
        "seed",
        "person_generation",
        "enhance_prompt",
        "number_of_videos",
        "output_gcs_uri",
        "resize_mode",
    ]:
        if key in job and key not in job.get("arguments", {}):
            args[key] = job[key]

    resolution = args.get("resolution")
    duration = snap_duration(model_key, args.get("duration", 8), resolution)

    config_kwargs: dict[str, Any] = {
        "duration_seconds": duration,
        "aspect_ratio": args.get("aspect_ratio", "16:9"),
        "person_generation": args.get("person_generation") or model_defaults.get("person_generation"),
    }
    if resolution:
        config_kwargs["resolution"] = resolution
    # Veo 3.1 audio is on by default. Only send the flag when the job
    # explicitly wants it off (or when a backend still honors True).
    if args.get("generate_audio") is False:
        config_kwargs["generate_audio"] = False
    if args.get("negative_prompt"):
        config_kwargs["negative_prompt"] = args["negative_prompt"]
    if args.get("seed") is not None:
        config_kwargs["seed"] = int(args["seed"])
    if args.get("enhance_prompt") is not None:
        config_kwargs["enhance_prompt"] = bool(args["enhance_prompt"])
    if args.get("number_of_videos"):
        config_kwargs["number_of_videos"] = int(args["number_of_videos"])
    if args.get("output_gcs_uri"):
        config_kwargs["output_gcs_uri"] = args["output_gcs_uri"]
    if args.get("resize_mode"):
        config_kwargs["resize_mode"] = args["resize_mode"]
    if last_frame_image is not None:
        if not preset.get("supports_last_frame"):
            raise ValueError(f"{model_key} does not support last_frame / end_image")
        config_kwargs["last_frame"] = last_frame_image
    if reference_images:
        config_kwargs["reference_images"] = reference_images

    return types.GenerateVideosConfig(**config_kwargs), duration


def save_generated_video(client: Any, generated: Any, out_path: Path) -> None:
    video = getattr(generated, "video", None)
    if video is None:
        raise RuntimeError(f"No video on generated result: {generated!r}")

    video_bytes = getattr(video, "video_bytes", None)
    if video_bytes:
        out_path.write_bytes(video_bytes)
        return

    destination = str(out_path)
    try:
        client.files.download(file=video, destination=destination)
    except TypeError:
        client.files.download(video, destination)

    if out_path.exists() and out_path.stat().st_size > 0:
        return

    uri = getattr(video, "uri", None)
    if not uri:
        raise RuntimeError("Veo returned a video with no bytes and no download URI")

    import urllib.request

    request = urllib.request.Request(uri)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key and "generativelanguage.googleapis.com" in uri:
        request.add_header("x-goog-api-key", api_key)
    with urllib.request.urlopen(request, timeout=300) as response:
        out_path.write_bytes(response.read())


def poll_operation(client: Any, operation: Any, interval: float) -> Any:
    while not getattr(operation, "done", False):
        time.sleep(interval)
        print("        waiting...", flush=True)
        operation = client.operations.get(operation)
    error = getattr(operation, "error", None)
    if error:
        raise RuntimeError(f"Veo operation failed: {error}")
    return operation


def default_output_name(shot_id: str, model_key: str) -> str:
    return f"{shot_id}_{model_key}.mp4"


def run(
    jobs_path: Path,
    only: list[str] | None,
    dry_run: bool,
    overwrite: bool,
    root: Path,
    poll_interval: float,
) -> None:
    cfg = load_json(jobs_path)
    board = cfg.get("board", "")
    image_dir = resolve(cfg.get("image_dir", "keyframes"), root)
    output_dir = resolve(cfg.get("output_dir", "video"), root)
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = cfg.get("defaults", {})
    jobs = cfg.get("jobs", [])
    if only:
        jobs = [job for job in jobs if job.get("id") in only]

    if not jobs:
        print("No jobs selected.")
        return

    client = None
    if not dry_run:
        try:
            client = make_client()
        except Exception as exc:
            sys.exit(f"Error: {exc}")

    backend_label = "Vertex AI" if using_vertex() else "Gemini API"
    print(f"\n{'=' * 60}")
    print(f"  Jobs     : {jobs_path}")
    print(f"  Board    : {board or '(none)'}")
    print(f"  Output   : {output_dir}")
    print(f"  Backend  : {backend_label}")
    print(f"  Shots    : {', '.join(job['id'] for job in jobs)}")
    print(f"{'=' * 60}\n")

    manifest_path = output_dir / "manifest.jsonl"

    for job in jobs:
        shot_id = job["id"]
        try:
            model_key = resolve_model_key(job.get("model", defaults.get("model", "veo")))
        except ValueError as exc:
            print(f"  [SKIP] {shot_id} {exc}")
            continue
        if model_key not in MODEL_PRESETS:
            print(f"  [SKIP] {shot_id} unknown Veo preset: {model_key}")
            continue

        shot = find_shot_in_board(root, board, shot_id) if board else None
        preset = MODEL_PRESETS[model_key]
        keyframe: Path | None = None
        if preset.get("requires_image", True) or job.get("image"):
            try:
                keyframe = find_keyframe(job, shot, image_dir, root)
            except Exception as exc:
                if preset.get("requires_image", True):
                    print(f"  [SKIP] {shot_id} {exc}")
                    continue
                keyframe = None

        out_name = job.get("output_name") or default_output_name(shot_id, model_key)
        out_path = output_dir / out_name
        if out_path.exists() and not overwrite:
            print(f"  [SKIP] {out_path.name} already exists")
            continue

        end_keyframe = None
        if job.get("end_image"):
            end_keyframe = (
                resolve(job["end_image"], root)
                if "/" in job["end_image"]
                else image_dir / job["end_image"]
            )
            if not end_keyframe.exists():
                print(f"  [SKIP] {shot_id} end_image not found: {end_keyframe}")
                continue

        extend_from = None
        if job.get("extend_from"):
            extend_from = resolve(job["extend_from"], root)
            if not extend_from.exists():
                print(f"  [SKIP] {shot_id} extend_from not found: {extend_from}")
                continue
            if not preset.get("supports_extension"):
                print(f"  [SKIP] {shot_id} {model_key} cannot extend videos")
                continue

        try:
            prompt = compose_prompt(job, shot)
        except Exception as exc:
            print(f"  [SKIP] {shot_id} {exc}")
            continue

        model_id = job.get("endpoint") or job.get("model_id")
        if not model_id:
            model_id = preset["vertex_model"] if using_vertex() else preset["gemini_model"]

        if dry_run:
            requested = job.get("duration", defaults.get("duration", 8))
            resolution = job.get("resolution", defaults.get("resolution"))
            duration = snap_duration(model_key, requested, resolution)
            print(f"  [DRY] {shot_id} -> {model_key} ({model_id})")
            if keyframe:
                print(f"        image: {relpath(keyframe, root)}")
            else:
                print("        image: none (text-to-video)")
            if end_keyframe:
                print(f"        last_frame: {relpath(end_keyframe, root)}")
            if job.get("reference_images") or job.get("image_urls"):
                print(f"        references: {job.get('reference_images') or job.get('image_urls')}")
            if job.get("dialogue"):
                print(f"        dialogue: {job['dialogue']}")
            snap_note = ""
            if str(requested) != str(duration):
                snap_note = f" (snapped from {requested})"
            print(
                f"        duration: {duration}s{snap_note}  "
                f"audio: {job.get('generate_audio', defaults.get('generate_audio', True))}"
            )
            preview = prompt if len(prompt) <= 160 else prompt[:157] + "..."
            print(f"        prompt: {preview}")
            print(f"        out: {relpath(out_path, root)}")
            continue

        print(f"  [GEN] {shot_id} -> {model_key} ...", end=" ", flush=True)
        started = time.time()
        try:
            from google.genai import types

            image = image_from_path(keyframe) if keyframe else None
            last_frame = image_from_path(end_keyframe) if end_keyframe else None
            references = collect_reference_images(job, image_dir, root, preset)
            config, duration = build_config(model_key, job, defaults, last_frame, references)

            source_kwargs: dict[str, Any] = {"prompt": prompt}
            if image is not None:
                source_kwargs["image"] = image
            if extend_from is not None:
                source_kwargs["video"] = video_from_path(extend_from)

            operation = client.models.generate_videos(
                model=model_id,
                source=types.GenerateVideosSource(**source_kwargs),
                config=config,
            )
            operation = poll_operation(client, operation, poll_interval)
            response = getattr(operation, "response", None) or getattr(operation, "result", None)
            generated_videos = getattr(response, "generated_videos", None) if response else None
            if not generated_videos:
                rai = getattr(response, "rai_media_filtered_reasons", None) if response else None
                extra = f" RAI: {rai}" if rai else ""
                raise RuntimeError(f"No generated_videos in Veo response{extra}: {response!r}")

            save_generated_video(client, generated_videos[0], out_path)
            elapsed = time.time() - started
            record = {
                "shot_id": shot_id,
                "model": model_key,
                "endpoint": model_id,
                "backend": "vertex" if using_vertex() else "gemini",
                "keyframe": relpath(keyframe, root) if keyframe else None,
                "output": relpath(out_path, root),
                "duration": duration,
                "resolution": getattr(config, "resolution", None),
                "generate_audio": getattr(config, "generate_audio", None),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(elapsed, 1),
            }
            append_manifest(manifest_path, record)
            print(f"done ({elapsed:.1f}s) -> {out_path.name}")
        except Exception as exc:
            print(f"FAILED - {exc}")

    print("\nAll done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate storyboard videos with Google Veo (native audio + lip sync)."
    )
    parser.add_argument("--jobs", required=True, help="Path to video_jobs.json")
    parser.add_argument(
        "--root",
        default=".",
        help="Base directory for relative paths in the jobs file (default: current directory).",
    )
    parser.add_argument("--only", nargs="+", metavar="ID", help="Only generate selected shot IDs.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected jobs without calling Veo.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if output MP4 exists.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between Veo operation polls (default: 10).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    jobs_path = resolve(args.jobs, root)
    if not jobs_path.exists():
        sys.exit(f"jobs file not found: {jobs_path}")

    run(
        jobs_path=jobs_path,
        only=args.only,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        root=root,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
