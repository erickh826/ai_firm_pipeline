#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimax_tts_from_srt.py — 讀 SRT，按角色批次生成粵語配音。

設計要點
--------
1. 粵英夾雜不分段生成。language_boost="Chinese,Yue" 之下句內英文會自然
   用英文讀；硬拆兩段再拼會斷掉語調連貫性。發音問題用 pronunciation_dict
   （支援粵拼帶 1-6 聲調）處理，不是用切割處理。
2. 跨剪接點的對白一次過生成，再用 silencedetect 找最接近目標秒數的
   靜音點切開，不是分兩次 render。
3. 生成後印出「SRT 分配時長 vs 實際時長」對照，並算出建議 offset，
   可直接貼回 lipsync 設定。
4. 內建 cache：文本 + 聲線設定不變就不重複呼叫 API。

前置
----
    export MINIMAX_API_KEY="..."
    export MINIMAX_GROUP_ID="..."      # 部分區域端點需要（可寫進專案根目錄 .env）
    需要 ffmpeg / ffprobe。

用法
----
    python -m aifilm.tts --srt script.srt --cast voice_cast.json --dry-run
    python -m aifilm.tts --srt script.srt --cast voice_cast.json
    python -m aifilm.tts --srt script.srt --cast voice_cast.json --only 1 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ENDPOINT = "https://api.minimax.io/v1/t2a_v2"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(Path.cwd() / ".env")

SRT_BLOCK = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"(.*?)(?=\n\s*\n|\Z)",
    re.S,
)
SPEAKER_SPLIT = re.compile(r"^\s*([^：:]{1,12})\s*[：:]\s*(.*)$", re.S)


@dataclass
class Cue:
    index: int
    start: float
    end: float
    speaker: str
    text: str

    @property
    def allotted(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------
# SRT
# --------------------------------------------------------------------------

def parse_timestamp(ts: str) -> float:
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000


def parse_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig")
    cues: list[Cue] = []
    for m in SRT_BLOCK.finditer(raw):
        idx, start, end, body = m.groups()
        body = " ".join(line.strip() for line in body.strip().splitlines())
        sm = SPEAKER_SPLIT.match(body)
        speaker, text = (sm.group(1).strip(), sm.group(2).strip()) if sm else ("", body)
        cues.append(Cue(int(idx), parse_timestamp(start),
                        parse_timestamp(end), speaker, text))
    if not cues:
        raise ValueError(f"{path} 解析不到任何字幕塊")
    return cues


# --------------------------------------------------------------------------
# ffmpeg
# --------------------------------------------------------------------------

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def must_run(cmd: list[str]) -> str:
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} 失敗:\n{proc.stderr[-1200:]}")
    return proc.stdout


def duration_of(path: Path) -> float:
    out = must_run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out.strip())


def to_wav(src: Path, dst: Path, loudnorm: bool = True) -> Path:
    """統一成 48k 單聲道 PCM。輸入音量不穩會直接影響 lipsync 的 mel 特徵。"""
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    cmd += ["-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    must_run(cmd)
    return dst


def find_silences(path: Path, noise_db: int = -35,
                  min_dur: float = 0.12) -> list[tuple[float, float]]:
    proc = run(["ffmpeg", "-i", str(path), "-af",
                f"silencedetect=noise={noise_db}dB:d={min_dur}",
                "-f", "null", "-"])
    starts = [float(v) for v in re.findall(r"silence_start:\s*([\d.]+)", proc.stderr)]
    ends = [float(v) for v in re.findall(r"silence_end:\s*([\d.]+)", proc.stderr)]
    return list(zip(starts, ends))


def split_at_silence(src: Path, target: float, out_a: Path, out_b: Path,
                     window: float = 1.2) -> float:
    """在最接近 target 秒的靜音中點切開。切唔到就硬切並警告。

    跨剪接點的對白必須一次過生成保住語調，再在這裡切，
    而不是分兩次 render。
    """
    silences = find_silences(src)
    total = duration_of(src)
    candidates = [(abs((s + e) / 2 - target), (s + e) / 2)
                  for s, e in silences
                  if abs((s + e) / 2 - target) <= window
                  and 0.2 < (s + e) / 2 < total - 0.2]
    if candidates:
        cut = min(candidates)[1]
    else:
        cut = min(max(target, 0.2), total - 0.2)
        print(f"      ⚠ {target:.2f}s 附近 ±{window}s 找不到靜音，"
              f"硬切在 {cut:.2f}s，請試聽是否切斷字詞")
    must_run(["ffmpeg", "-y", "-i", str(src), "-t", f"{cut:.3f}",
              "-c", "copy", str(out_a)])
    must_run(["ffmpeg", "-y", "-i", str(src), "-ss", f"{cut:.3f}",
              "-c", "copy", str(out_b)])
    return cut


def concat_wavs(parts: list[Path], dst: Path, gap: float = 0.0) -> Path:
    if len(parts) == 1:
        shutil.copy(parts[0], dst)
        return dst
    inputs: list[str] = []
    for p in parts:
        inputs += ["-i", str(p)]
    if gap > 0:
        # 每段之間插靜音，粵英切換或換氣點用得上
        chain = []
        for i in range(len(parts)):
            chain.append(f"[{i}:a]")
            if i < len(parts) - 1:
                chain.append(f"aevalsrc=0:d={gap}[g{i}];[g{i}]")
        filt = "".join(chain) + f"concat=n={2 * len(parts) - 1}:v=0:a=1[out]"
    else:
        filt = "".join(f"[{i}:a]" for i in range(len(parts))) + \
               f"concat=n={len(parts)}:v=0:a=1[out]"
    must_run(["ffmpeg", "-y", *inputs, "-filter_complex", filt,
              "-map", "[out]", "-ar", "48000", "-ac", "1",
              "-c:a", "pcm_s16le", str(dst)])
    return dst


# --------------------------------------------------------------------------
# MiniMax
# --------------------------------------------------------------------------

def synthesize(text: str, voice: dict, cfg: dict, endpoint: str,
               api_key: str, group_id: str | None,
               retries: int = 4) -> tuple[bytes, dict]:
    body: dict[str, Any] = {
        "model": cfg.get("model", "speech-2.8-hd"),
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": voice["voice_id"],
            "speed": voice.get("speed", 1.0),
            "vol": voice.get("vol", 1.0),
            "pitch": voice.get("pitch", 0),
        },
        "audio_setting": cfg.get("audio_setting", {
            "sample_rate": 44100, "bitrate": 128000,
            "format": "wav", "channel": 1,
        }),
    }
    if voice.get("emotion"):
        body["voice_setting"]["emotion"] = voice["emotion"]
    if cfg.get("language_boost"):
        body["language_boost"] = cfg["language_boost"]
    if cfg.get("pronunciation_dict"):
        body["pronunciation_dict"] = {"tone": cfg["pronunciation_dict"]}
    if cfg.get("text_normalization") is not None:
        body["text_normalization"] = cfg["text_normalization"]

    url = endpoint + (f"?GroupId={group_id}" if group_id else "")
    req_body = json.dumps(body, ensure_ascii=False).encode("utf-8")

    last_err = ""
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=req_body, method="POST",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}: {exc.read()[:300]!r}"
            time.sleep(2 ** attempt)
            continue
        except Exception as exc:                       # noqa: BLE001
            last_err = str(exc)
            time.sleep(2 ** attempt)
            continue

        status = (payload.get("base_resp") or {}).get("status_code", 0)
        if status != 0:
            msg = (payload.get("base_resp") or {}).get("status_msg", "")
            last_err = f"status_code={status} {msg}"
            # 免費帳戶 RPM 只有 3，撞到限流就退避重試
            if "rate" in msg.lower() or status in (1002, 1039):
                time.sleep(4 * (attempt + 1))
                continue
            raise RuntimeError(f"MiniMax 回應錯誤：{last_err}")

        audio_hex = (payload.get("data") or {}).get("audio")
        if not audio_hex:
            raise RuntimeError(f"回應中沒有 audio 欄位：{json.dumps(payload)[:400]}")
        return bytes.fromhex(audio_hex), payload.get("extra_info", {})

    raise RuntimeError(f"重試 {retries} 次後仍失敗：{last_err}")


# --------------------------------------------------------------------------

def cache_path(cache_dir: Path, text: str, voice: dict, cfg: dict) -> Path:
    key = json.dumps({"t": text, "v": voice,
                      "m": cfg.get("model"),
                      "lb": cfg.get("language_boost"),
                      "pd": cfg.get("pronunciation_dict")},
                     ensure_ascii=False, sort_keys=True)
    return cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".wav")


def render_cue_group(cues: list[Cue], plan: dict, cast: dict, cfg: dict,
                     args, workdir: Path, out_dir: Path,
                     api_key: str, group_id: str | None) -> list[dict]:
    """處理一個 plan 條目：可能是單 cue、多 cue 合併、或生成後再切開。"""
    label = plan.get("label") or f"cue{cues[0].index:02d}"
    speaker = plan.get("speaker") or cues[0].speaker
    voice = cast.get(speaker)
    if not voice:
        raise KeyError(f"[{label}] cast 裡沒有角色 {speaker!r}；"
                       f"已定義：{', '.join(cast)}")

    # 合併多個 cue 時用設定的停頓串起來，維持自然斷句
    text = plan.get("text_override")
    if text is None:
        joiner = plan.get("joiner", "")
        text = joiner.join(c.text for c in cues)
    allotted = sum(c.allotted for c in cues)

    if args.dry_run:
        chars = len(text)
        print(f"  [{label}] {speaker}  {chars} 字  SRT 分配 {allotted:.1f}s")
        return [{"label": label, "speaker": speaker, "chars": chars,
                 "allotted": allotted}]

    cache_dir = out_dir.parent / ".tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_path(cache_dir, text, voice, cfg)

    if cached.exists() and not args.force:
        print(f"  [{label}] {speaker}  快取命中")
        raw = cached
    else:
        print(f"  [{label}] {speaker}  呼叫 MiniMax…")
        audio, info = synthesize(text, voice, cfg, args.endpoint,
                                 api_key, group_id)
        tmp = workdir / f"{label}_raw.wav"
        tmp.write_bytes(audio)
        to_wav(tmp, cached, loudnorm=not args.no_loudnorm)
        raw = cached
        if info.get("audio_length"):
            print(f"      MiniMax 報告 {info['audio_length'] / 1000:.2f}s")
        time.sleep(args.sleep)

    actual = duration_of(raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    split_near = plan.get("split_near")
    outputs = plan["outputs"]

    if split_near is not None and len(outputs) == 2:
        a, b = out_dir / outputs[0], out_dir / outputs[1]
        cut = split_at_silence(raw, float(split_near),
                               workdir / "a.wav", workdir / "b.wav",
                               window=float(plan.get("split_window", 1.2)))
        shutil.move(str(workdir / "a.wav"), a)
        shutil.move(str(workdir / "b.wav"), b)
        print(f"      在 {cut:.2f}s 切開 → {outputs[0]} / {outputs[1]}")
        for p in (a, b):
            results.append({"label": label, "speaker": speaker,
                            "file": p.name, "duration": round(duration_of(p), 2)})
    else:
        dst = out_dir / outputs[0]
        shutil.copy(raw, dst)
        results.append({"label": label, "speaker": speaker,
                        "file": dst.name, "duration": round(actual, 2)})

    ratio = actual / allotted if allotted else 0
    flag = ""
    if ratio > 1.15:
        flag = f"  ⚠ 超出 SRT 分配 {actual - allotted:.1f}s，需延長鏡頭或刪字"
    elif 0 < ratio < 0.7:
        flag = f"  ⚠ 只用了分配的 {ratio:.0%}，鏡頭會有空檔"
    print(f"      實際 {actual:.2f}s / 分配 {allotted:.1f}s{flag}")

    for r in results:
        r["allotted"] = round(allotted, 2)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="用 MiniMax 由 SRT 批次生成粵語配音")
    ap.add_argument("--srt", required=True)
    ap.add_argument("--cast", required=True, help="聲線與 plan 設定 JSON")
    ap.add_argument("--out", default="audio", help="輸出目錄")
    ap.add_argument("--endpoint", default=os.environ.get(
        "MINIMAX_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--only", nargs="*", help="只跑指定 plan label")
    ap.add_argument("--sleep", type=float, default=1.5,
                    help="每次呼叫之間的間隔秒數（免費帳戶 RPM 只有 3）")
    ap.add_argument("--no-loudnorm", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"找不到 {tool}，請先安裝 ffmpeg", file=sys.stderr)
            return 1

    api_key = os.environ.get("MINIMAX_API_KEY", "")
    group_id = os.environ.get("MINIMAX_GROUP_ID") or None
    if not args.dry_run and not api_key:
        print("未設定 MINIMAX_API_KEY", file=sys.stderr)
        return 1

    cues = {c.index: c for c in parse_srt(Path(args.srt))}
    cfg = json.loads(Path(args.cast).read_text(encoding="utf-8"))
    cast = cfg["cast"]
    plans = cfg["plan"]
    if args.only:
        wanted = set(args.only)
        plans = [p for p in plans
                 if (p.get("label") or f"cue{p['cues'][0]:02d}") in wanted]

    out_dir = Path(args.out)
    print(f"{'預估' if args.dry_run else '生成'} {len(plans)} 組配音\n")

    import tempfile
    all_results, failures = [], []
    with tempfile.TemporaryDirectory(prefix="mmtts_") as tmp:
        workdir = Path(tmp)
        for plan in plans:
            ids = plan["cues"] if isinstance(plan.get("cues"), list) \
                else [plan["cue"]]
            try:
                group = [cues[i] for i in ids]
            except KeyError as exc:
                print(f"  SRT 裡沒有 cue {exc}", file=sys.stderr)
                failures.append({"plan": plan.get("label"), "error": str(exc)})
                continue
            try:
                all_results += render_cue_group(
                    group, plan, cast, cfg, args, workdir, out_dir,
                    api_key, group_id)
            except Exception as exc:                   # noqa: BLE001
                label = plan.get("label") or f"cue{ids[0]:02d}"
                print(f"  [{label}] 失敗：{exc}", file=sys.stderr)
                failures.append({"plan": label, "error": str(exc)})

    if not args.dry_run and all_results:
        manifest = out_dir / "tts_manifest.json"
        manifest.write_text(json.dumps(
            {"results": all_results, "failures": failures},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n清單：{manifest}")
        print("把實際 duration 填回 lipsync 設定的 offset 再跑 lipsync --dry-run")

    if failures:
        print(f"失敗 {len(failures)} 組", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
