# AI Film Pipeline

Reusable CLI tools for AI short-film production.

Two video backends share the same `video_jobs.json` shape:

| Path | Video | Speech / lip sync |
| --- | --- | --- |
| **fal.ai** | Kling O3 / Seedance / MiniMax | MiniMax TTS → fal.ai lip sync |
| **Google Veo** | Veo 3.1 image-to-video | Native audio + lip sync in one pass |

Use the Veo path for Google Cloud AI Builder Cup (the submission must run on Google Cloud). Use the fal.ai path when you need Cantonese MiniMax voices and a dedicated lip-sync engine.

## Tools

| Tool | Command | Purpose | Auth |
| --- | --- | --- | --- |
| Generate video (fal) | `aifilm-videos` | fal.ai image-to-video (Kling O3 / Seedance / MiniMax) | `FAL_KEY` |
| Generate video (Veo) | `aifilm-veo` | Google Veo 3.1 image-to-video with native audio | `GEMINI_API_KEY` or Vertex ADC |
| Generate dubbing | `aifilm-tts` | MiniMax TTS from an SRT | `MINIMAX_API_KEY`, `ffmpeg` |
| Lip sync | `aifilm-lipsync` | fal.ai lip sync (sync-v2/v3, kling, omnihuman) | `FAL_KEY`, `ffmpeg` |

## Layout

```
.
├── aifilm/
│   ├── videos.py      # fal.ai video generation
│   ├── veo.py         # Google Veo video generation
│   ├── lipsync.py     # fal.ai lip sync
│   ├── tts.py         # MiniMax TTS
│   └── common.py      # shared path / jobs helpers
├── templates/
│   ├── video_jobs.example.json       # fal.ai jobs
│   ├── video_jobs.veo.example.json   # Veo jobs
│   └── voice_cast.example.json
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Install

1. Python ≥ 3.10
2. ffmpeg / ffprobe (needed for TTS and fal lip sync): `brew install ffmpeg`
3. Install the package:

   ```bash
   pip install -e .
   ```

4. Copy `.env.example` to `.env` in your **project root** (not committed) and fill in keys.

Relative paths are resolved against `--root` (default: current directory). Every tool also loads `<cwd>/.env`.

## Google Veo path (Builder Cup)

Veo 3.1 generates picture **and** synced speech in one call. Put spoken lines in `dialogue` (or quote them in `prompt`). You do **not** need MiniMax + fal lip sync unless you want a different voice after the fact.

### Auth

**Gemini API (AI Studio)** — fastest for local tests:

```bash
GEMINI_API_KEY=your_key
```

**Vertex AI (required if the Cup prototype must run on Google Cloud):**

```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your-gcp-project
GOOGLE_CLOUD_LOCATION=us-central1
gcloud auth application-default login
```

### Jobs file

Copy `templates/video_jobs.veo.example.json`. Same fields as the fal jobs file, plus Veo-only keys:

| Field | Meaning |
| --- | --- |
| `model` | `veo` (quality), `veo-fast`, `veo-lite`, or `veo-text` |
| `image` | First-frame keyframe |
| `end_image` | Optional last frame (interpolation) |
| `reference_images` | Up to 3 character / product references |
| `dialogue` | Spoken line; injected as quoted speech + lip-sync instruction |
| `speaker_direction` | Optional delivery note (`quietly, afraid`) |
| `duration` | `4`, `6`, or `8` seconds (other values snap to the nearest). 1080p / 4K force `8` |
| `resolution` | `720p` (default), `1080p`, `4k` |
| `aspect_ratio` | `16:9` or `9:16` |
| `generate_audio` | Native soundtrack. Leave `true`. Set `false` only if the backend still honors it |

`aifilm-videos` models (`kling-o3`, `seedance`, …) are rejected here on purpose so a mixed jobs file cannot silently hit the wrong API.

### Run

```bash
aifilm-veo --jobs board1/video_jobs.json --dry-run
aifilm-veo --jobs board1/video_jobs.json
aifilm-veo --jobs board1/video_jobs.json --only 1.1 1.6
```

Prompting tips that matter for lip sync:

- Put the exact line in quotes. Veo treats quoted text as dialogue.
- Describe delivery (`whispers`, `stammers`, `too rehearsed`).
- Keep camera and identity locked (`same man as the reference image, static eye-level medium shot`).

Duration / cost notes:

- Veo clips are 4 / 6 / 8 seconds. A `duration: 5` job snaps to 4 (dry-run prints the snap).
- 1080p and 4K only work at 8 seconds.
- Generated files on the Gemini API expire after 2 days on Google's side; `aifilm-veo` downloads them immediately.

## fal.ai path (Seedance / Kling + MiniMax)

### 1. Generate video

Write `video_jobs.json` (see `templates/video_jobs.example.json`):

```bash
aifilm-videos --jobs board1/video_jobs.json --dry-run
aifilm-videos --jobs board1/video_jobs.json
aifilm-videos --jobs board1/video_jobs.json --only 1.1 1.6
```

`image` can be a filename (looked up in `image_dir`) or a relative path (if it contains `/`, it is resolved against `--root`).

### 2. Generate dubbing (TTS)

Write `voice_cast.json` (see `templates/voice_cast.example.json`), then:

```bash
aifilm-tts --srt script.srt --cast voice_cast.json --dry-run
aifilm-tts --srt script.srt --cast voice_cast.json
aifilm-tts --srt script.srt --cast voice_cast.json --only cue01 cue02
```

The tool prints allotted SRT duration vs actual audio duration, plus a suggested lipsync `offset`.

### 3. Lip sync

Add a `lipsync` block to each job (see `templates/video_jobs.example.json`):

```bash
aifilm-lipsync --jobs board1/video_jobs.json --dry-run
aifilm-lipsync --jobs board1/video_jobs.json
aifilm-lipsync --jobs board1/video_jobs.json --only 1.1 --loudnorm
```

### Lip-sync engines

| Engine | Kind | Notes |
| --- | --- | --- |
| `sync-v2` / `sync-v2-pro` / `sync-v3` | video-to-video | Stable; `sync_mode` handles length mismatch |
| `kling` | video-to-video | Cheapest; source clip must be 2–10 s |
| `omnihuman` | image-to-video | Image + audio → talking head, no seam |

Use `audio` for a single speaker. Use `panels` for split-screen (`crop`: `left` / `right` / `top` / `bottom` or `x,y,w,h`).

`aifilm-lipsync --dry-run` prints a USD estimate per shot. Videos are trimmed to `trim_to_seconds` before upload so you are not billed for frames you will cut.

## Which path should I use?

- **English (or any language Veo can speak in-prompt)** → `aifilm-veo`. Native audio is the lip sync.
- **Cantonese MiniMax voices / existing fal jobs** → `aifilm-videos` + `aifilm-tts` + `aifilm-lipsync`.
- **AI Builder Cup** → Vertex-backed `aifilm-veo` deployed on Cloud Run / GCP. The contest requires a Google Cloud stack; fal.ai-only submissions are not eligible.
