#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_lipsync_fal.py — 讀 jobs JSON，批次做 fal.ai lipsync。

設計要點
--------
1. 直接讀 job 的 output_name 去定位影片，不靠檔名 pattern（不同版本/regen 的檔名會漏）。
2. 分割畫面必須拆 panel 各自 lipsync 再 hstack 拼回，否則兩張臉會被同一段音頻同時驅動。
3. 先 trim 到 trim_to_seconds 再上傳。按秒收費，別為要剪走的幀付錢。
4. 內建 cache：同樣的 video+audio+engine 不會重複呼叫 API。
5. --dry-run 會印出預估費用，不呼叫 API。

前置
----
    pip install fal-client
    把 FAL_KEY 寫進專案根目錄 .env（或直接 export）
    需要 ffmpeg / ffprobe 在 PATH。

jobs JSON 需要新增的欄位
------------------------
單人鏡頭：
    "lipsync": {
      "engine": "sync-v2",              # sync-v2 | sync-v2-pro | sync-v3 | kling | omnihuman | skip
      "audio":  "audio/shot_2.1.wav",
      "offset": 0.0                     # 選填，音頻前面補幾秒靜音
    }

分割畫面 / 多人同框：
    "lipsync": {
      "engine": "sync-v2",
      "panels": [
        {"crop": "left",  "audio": "audio/shot_3.4_a.wav", "offset": 0.0},
        {"crop": "right", "audio": "audio/shot_3.4_b.wav", "offset": 3.5}
      ]
    }
    crop 可以是 "left" / "right" / "top" / "bottom"，或 "x,y,w,h"（像素）。

image-to-video 引擎（omnihuman）會忽略已生成的影片，
改用 job.elements[0].frontal_image 直接出片：
    "lipsync": {
      "engine": "omnihuman",
      "audio": "audio/shot_2.1.wav",
      "prompt": "medium shot, video call framing, subject speaking to camera"
    }

用法
----
    python -m aifilm.lipsync --jobs jobs.json --dry-run
    python -m aifilm.lipsync --jobs jobs.json --only 3.4
    python -m aifilm.lipsync --jobs jobs.json --loudnorm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(Path.cwd() / ".env")

# --------------------------------------------------------------------------
# 引擎登記表。價錢是 2026-09 的公開牌價，會變，只作預估用。
# --------------------------------------------------------------------------

@dataclass
class Engine:
    endpoint: str
    kind: str                    # "v2v"（影片轉影片）或 "i2v"（圖生影片）
    price: float                 # 每秒美金
    round_to: float = 0.0        # 計費進位（秒），0 = 不進位
    extra_args: dict = field(default_factory=dict)


ENGINES: dict[str, Engine] = {
    # sync.so，穩定、支援 sync_mode 處理時長不一致
    "sync-v2":     Engine("fal-ai/sync-lipsync/v2", "v2v", 3.0 / 60,
                          extra_args={"model": "lipsync-2"}),
    "sync-v2-pro": Engine("fal-ai/sync-lipsync/v2", "v2v", 5.0 / 60,
                          extra_args={"model": "lipsync-2-pro"}),
    "sync-v3":     Engine("fal-ai/sync-lipsync/v3", "v2v", 8.0 / 60),
    # Kling 自家 lipsync，最平，但源片限 2–10 秒、處理約 12 分鐘
    "kling":       Engine("fal-ai/kling-video/lipsync/audio-to-video", "v2v",
                          0.014, round_to=5.0),
    # 圖 + 音直接出片，talking head 專用，無貼合縫
    "omnihuman":   Engine("fal-ai/bytedance/omnihuman/v1.5", "i2v", 0.16),
}

CROP_PRESETS = {
    "left":   "iw/2:ih:0:0",
    "right":  "iw/2:ih:iw/2:0",
    "top":    "iw:ih/2:0:0",
    "bottom": "iw:ih/2:0:ih/2",
}


# --------------------------------------------------------------------------
# shell / ffmpeg 小工具
# --------------------------------------------------------------------------

def run(cmd: list[str], quiet: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"命令失敗: {' '.join(cmd[:4])}...\n{proc.stderr[-1500:]}"
        )
    return proc.stdout


def probe_duration(path: Path) -> float:
    out = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(out.strip())


def trim_video(src: Path, dst: Path, seconds: float) -> Path:
    """精確剪到指定長度。重編碼，避免關鍵幀對不齊導致首幀凍結。"""
    run([
        "ffmpeg", "-y", "-i", str(src), "-t", f"{seconds:.3f}",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-an", str(dst),
    ])
    return dst


def crop_video(src: Path, dst: Path, spec: str) -> Path:
    """spec 是 CROP_PRESETS 的 key，或者 'x,y,w,h' 像素值。"""
    if spec in CROP_PRESETS:
        crop_filter = f"crop={CROP_PRESETS[spec]}"
    else:
        try:
            x, y, w, h = (int(v.strip()) for v in spec.split(","))
        except ValueError as exc:
            raise ValueError(f"看不懂的 crop 設定: {spec!r}") from exc
        crop_filter = f"crop={w}:{h}:{x}:{y}"
    run([
        "ffmpeg", "-y", "-i", str(src), "-vf", crop_filter,
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-an", str(dst),
    ])
    return dst


def hstack_videos(parts: list[Path], dst: Path) -> Path:
    """把 lipsync 完的 panel 橫向拼回去。分割畫面專用。"""
    if len(parts) == 1:
        shutil.copy(parts[0], dst)
        return dst
    inputs: list[str] = []
    for p in parts:
        inputs += ["-i", str(p)]
    run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", f"hstack=inputs={len(parts)}",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p", str(dst),
    ])
    return dst


def prep_audio(src: Path, dst: Path, offset: float = 0.0,
               loudnorm: bool = False) -> Path:
    """統一轉 48k 單聲道 wav，可選前置靜音與響度正規化。

    輸入音量不穩會直接影響 mel 特徵，粵英夾雜的音軌尤其常見。
    """
    filters: list[str] = []
    if offset > 0:
        filters.append(f"adelay={int(offset * 1000)}|{int(offset * 1000)}")
    if loudnorm:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    run(cmd)
    return dst


# --------------------------------------------------------------------------
# 快取
# --------------------------------------------------------------------------

def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cache_key(engine: str, video: Path | None, audio: Path,
              image: Path | None) -> str:
    parts = [engine, file_digest(audio)]
    if video:
        parts.append(file_digest(video))
    if image:
        parts.append(file_digest(image))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


# --------------------------------------------------------------------------
# fal 呼叫
# --------------------------------------------------------------------------

def call_fal(engine: Engine, *, video: Path | None, audio: Path,
             image: Path | None, prompt: str | None,
             sync_mode: str) -> str:
    import fal_client

    args: dict[str, Any] = dict(engine.extra_args)
    args["audio_url"] = fal_client.upload_file(str(audio))

    if engine.kind == "i2v":
        if image is None:
            raise ValueError("omnihuman 這類引擎需要 frontal_image")
        args["image_url"] = fal_client.upload_file(str(image))
        if prompt:
            args["prompt"] = prompt
    else:
        if video is None:
            raise ValueError("video-to-video 引擎需要已生成的影片")
        args["video_url"] = fal_client.upload_file(str(video))
        if engine.endpoint.startswith("fal-ai/sync-lipsync"):
            args["sync_mode"] = sync_mode

    def on_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"      {log['message']}")

    result = fal_client.subscribe(
        engine.endpoint, arguments=args,
        with_logs=True, on_queue_update=on_update,
    )
    url = (result.get("video") or {}).get("url")
    if not url:
        raise RuntimeError(f"回應中找不到影片 URL: {json.dumps(result)[:500]}")
    return url


def download(url: str, dst: Path) -> Path:
    import urllib.request
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dst.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    return dst


def estimate_cost(engine: Engine, seconds: float) -> float:
    billable = seconds
    if engine.round_to:
        billable = math.ceil(seconds / engine.round_to) * engine.round_to
    return billable * engine.price


# --------------------------------------------------------------------------
# 單個 job
# --------------------------------------------------------------------------

def process_job(job: dict, cfg: dict, args: argparse.Namespace,
                workdir: Path) -> dict | None:
    job_id = job["id"]
    spec = job.get("lipsync")
    if not spec or spec.get("engine") == "skip":
        return None

    engine_name = args.engine or spec["engine"]
    if engine_name not in ENGINES:
        raise ValueError(f"[{job_id}] 未知引擎 {engine_name!r}；"
                         f"可用：{', '.join(ENGINES)}")
    engine = ENGINES[engine_name]

    root = cfg["root"]
    out_dir = root / cfg["output_dir"]
    src_video = out_dir / job["output_name"]
    stem = Path(job["output_name"]).stem
    final = root / cfg["lipsync_dir"] / f"{stem}_lipsync.mp4"

    panels = spec.get("panels")
    if not panels:
        panels = [{"crop": None,
                   "audio": spec["audio"],
                   "offset": spec.get("offset", 0.0)}]

    # ---- image-to-video：不需要已生成的影片 ----------------------------
    if engine.kind == "i2v":
        elements = job.get("elements") or []
        if not elements:
            raise ValueError(f"[{job_id}] 沒有 elements，omnihuman 無圖可用")
        image = root / elements[0]["frontal_image"]
        audio_src = root / panels[0]["audio"]
        audio = prep_audio(audio_src, workdir / f"{job_id}_a.wav",
                           panels[0].get("offset", 0.0), args.loudnorm)
        seconds = probe_duration(audio)
        cost = estimate_cost(engine, seconds)
        print(f"  [{job_id}] {engine_name}  圖生影片  {seconds:.1f}s  "
              f"≈ US${cost:.2f}")
        if args.dry_run:
            return {"id": job_id, "engine": engine_name,
                    "seconds": seconds, "cost": cost, "output": str(final)}
        key = cache_key(engine_name, None, audio, image)
        cached = root / cfg["cache_dir"] / f"{key}.mp4"
        if cached.exists() and not args.force:
            print(f"      快取命中，跳過 API")
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cached, final)
        else:
            url = call_fal(engine, video=None, audio=audio, image=image,
                           prompt=spec.get("prompt"), sync_mode=args.sync_mode)
            cached.parent.mkdir(parents=True, exist_ok=True)
            download(url, cached)
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cached, final)
        return {"id": job_id, "engine": engine_name, "seconds": seconds,
                "cost": cost, "output": str(final)}

    # ---- video-to-video ------------------------------------------------
    if not src_video.exists():
        raise FileNotFoundError(f"[{job_id}] 找不到來源影片 {src_video}")

    # 先剪再上傳。按秒收費，不要為之後會剪走的幀付錢。
    trim_to = job.get("trim_to_seconds")
    base = src_video
    actual = probe_duration(src_video)
    if trim_to and actual > trim_to + 0.05:
        base = trim_video(src_video, workdir / f"{job_id}_trim.mp4",
                          float(trim_to))
        print(f"  [{job_id}] 已剪 {actual:.1f}s → {trim_to}s")
    seconds = probe_duration(base)

    cost = estimate_cost(engine, seconds) * len(panels)
    label = f"{len(panels)} panel" if len(panels) > 1 else "全畫面"
    print(f"  [{job_id}] {engine_name}  {label}  {seconds:.1f}s  "
          f"≈ US${cost:.2f}")

    if engine_name == "kling" and not (2.0 <= seconds <= 10.0):
        print(f"      ⚠ kling lipsync 源片限 2–10 秒，此片 {seconds:.1f}s")

    if args.dry_run:
        return {"id": job_id, "engine": engine_name, "panels": len(panels),
                "seconds": seconds, "cost": cost, "output": str(final)}

    done_parts: list[Path] = []
    for idx, panel in enumerate(panels):
        piece = base
        if panel.get("crop"):
            piece = crop_video(base, workdir / f"{job_id}_p{idx}.mp4",
                               panel["crop"])
        audio = prep_audio(root / panel["audio"],
                           workdir / f"{job_id}_p{idx}.wav",
                           panel.get("offset", 0.0), args.loudnorm)

        key = cache_key(engine_name, piece, audio, None)
        cached = root / cfg["cache_dir"] / f"{key}.mp4"
        if cached.exists() and not args.force:
            print(f"      panel {idx}: 快取命中")
        else:
            print(f"      panel {idx}: 呼叫 {engine.endpoint}")
            url = call_fal(engine, video=piece, audio=audio, image=None,
                           prompt=None, sync_mode=args.sync_mode)
            cached.parent.mkdir(parents=True, exist_ok=True)
            download(url, cached)
        done_parts.append(cached)

    final.parent.mkdir(parents=True, exist_ok=True)
    hstack_videos(done_parts, final)
    return {"id": job_id, "engine": engine_name, "panels": len(panels),
            "seconds": seconds, "cost": cost, "output": str(final)}


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="批次 fal.ai lipsync")
    ap.add_argument("--jobs", required=True, help="jobs JSON 路徑")
    ap.add_argument("--root", default=".", help="專案根目錄，JSON 內相對路徑的基準")
    ap.add_argument("--lipsync-dir",
                    default="lipsync")
    ap.add_argument("--cache-dir", default=".lipsync_cache")
    ap.add_argument("--only", nargs="*", help="只跑指定 job id，例如 --only 2.1 3.4")
    ap.add_argument("--engine", help="覆寫所有 job 的引擎")
    ap.add_argument("--sync-mode", default="cut_off",
                    choices=["cut_off", "loop", "bounce", "silence", "remap"],
                    help="sync.so 處理音視頻時長不一致的方式")
    ap.add_argument("--loudnorm", action="store_true",
                    help="上傳前對音頻做響度正規化（粵英夾雜音軌建議開）")
    ap.add_argument("--force", action="store_true", help="忽略快取")
    ap.add_argument("--dry-run", action="store_true",
                    help="只印預估費用，不呼叫 API")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"找不到 {tool}，請先安裝 ffmpeg", file=sys.stderr)
            return 1
    if not args.dry_run and not os.environ.get("FAL_KEY"):
        print("未設定 FAL_KEY（請寫進專案根目錄 .env 或直接 export）",
              file=sys.stderr)
        return 1

    jobs_path = Path(args.jobs)
    data = json.loads(jobs_path.read_text(encoding="utf-8"))
    cfg = {
        "root": Path(args.root).resolve(),
        "output_dir": data["output_dir"],
        "lipsync_dir": args.lipsync_dir,
        "cache_dir": args.cache_dir,
    }

    jobs = data["jobs"]
    if args.only:
        wanted = set(args.only)
        jobs = [j for j in jobs if j["id"] in wanted]

    todo = [j for j in jobs
            if j.get("lipsync") and j["lipsync"].get("engine") != "skip"]
    if not todo:
        print("沒有帶 lipsync 設定的 job。請先在 JSON 加 lipsync 區塊"
              "（見本檔開頭的說明）。")
        return 0

    print(f"{'預估' if args.dry_run else '處理'} {len(todo)} 個鏡頭\n")
    results, failures = [], []
    with tempfile.TemporaryDirectory(prefix="lipsync_") as tmp:
        workdir = Path(tmp)
        for job in todo:
            try:
                res = process_job(job, cfg, args, workdir)
                if res:
                    results.append(res)
            except Exception as exc:                # noqa: BLE001
                print(f"  [{job['id']}] 失敗：{exc}", file=sys.stderr)
                failures.append({"id": job["id"], "error": str(exc)})

    total = sum(r["cost"] for r in results)
    print(f"\n{'預估' if args.dry_run else '完成'} {len(results)} 個鏡頭，"
          f"總計 ≈ US${total:.2f}")
    if failures:
        print(f"失敗 {len(failures)} 個：" +
              ", ".join(f["id"] for f in failures), file=sys.stderr)

    if not args.dry_run and results:
        manifest = cfg["root"] / cfg["lipsync_dir"] / "lipsync_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps({"results": results, "failures": failures,
                        "total_usd": round(total, 2)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"清單：{manifest}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
