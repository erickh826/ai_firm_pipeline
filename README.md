# AI Film Pipeline

一套可複用的 AI 短片配音管線：**fal.ai 圖生影片 → MiniMax 配音 → fal.ai 口型對齊**。
三個工具都是獨立 CLI，可以單獨跑，也可以串成一條完整流程。

## 三個工具

| 工具 | 命令 | 用途 | 依賴 |
| --- | --- | --- | --- |
| 生成影片 | `aifilm-videos` | fal.ai 圖生影片（Kling O3 / Seedance / MiniMax） | `FAL_KEY` |
| 生成配音 | `aifilm-tts` | MiniMax 由 SRT 批次生成粵語配音 | `MINIMAX_API_KEY`、`ffmpeg` |
| 口型對齊 | `aifilm-lipsync` | fal.ai lip sync（sync-v2/v3、kling、omnihuman） | `FAL_KEY`、`ffmpeg` |

## 目錄結構

```
.
├── aifilm/
│   ├── videos.py      # 生成影片
│   ├── lipsync.py     # 口型對齊
│   └── tts.py         # TTS 配音
├── templates/         # 範例 JSON（複製到你的專案再改）
│   ├── video_jobs.example.json
│   └── voice_cast.example.json
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## 安裝

1. Python ≥ 3.10
2. 安裝 ffmpeg / ffprobe：macOS 用 `brew install ffmpeg`
3. 安裝 Python 依賴：
   ```bash
   pip install -e .
   ```
4. 設定金鑰（放在你**專案根目錄**的 `.env`，不要提交）：
   ```bash
   cp .env.example .env
   # 再編輯 .env 填入 FAL_KEY / MINIMAX_API_KEY / MINIMAX_GROUP_ID
   ```

相對路徑一律以 `--root`（預設為當前目錄）為基準；三個工具也都會自動載入
`<工作目錄>/.env`。

## 快速開始

### 1. 生成影片

準備好 keyframe 圖片後，寫一份 `video_jobs.json`（參考 `templates/video_jobs.example.json`），
先 dry-run 驗證路徑，再正式生成：

```bash
aifilm-videos --jobs board1/video_jobs.json --dry-run
aifilm-videos --jobs board1/video_jobs.json
aifilm-videos --jobs board1/video_jobs.json --only 1.1 1.6   # 只生成指定鏡頭
```

`image` 可寫檔名（自動在 `image_dir` 下找）或相對路徑（含 `/` 則以 `--root` 為基準）。

### 2. 生成配音（TTS）

先寫 `voice_cast.json`（聲線 + 生成計劃，參考 `templates/voice_cast.example.json`），
再從 SRT 生成配音：

```bash
aifilm-tts --srt script.srt --cast voice_cast.json --dry-run
aifilm-tts --srt script.srt --cast voice_cast.json
aifilm-tts --srt script.srt --cast voice_cast.json --only cue01 cue02
```

生成完會印出「SRT 分配時長 vs 實際時長」對照與建議 offset，把實際 duration
填回 `lipsync` 設定的 `offset` 後再跑下一步。

### 3. 口型對齊（lip sync）

在 `video_jobs.json` 的每個 job 加上 `lipsync` 區塊（見 `templates/video_jobs.example.json`），
先 dry-run 看預估費用：

```bash
aifilm-lipsync --jobs board1/video_jobs.json --dry-run
aifilm-lipsync --jobs board1/video_jobs.json
aifilm-lipsync --jobs board1/video_jobs.json --only 1.1 --loudnorm
```

## lipsync 引擎

| 引擎 | 型態 | 特色 |
| --- | --- | --- |
| `sync-v2` / `sync-v2-pro` / `sync-v3` | 影片轉影片 | 穩定，支援 `sync_mode` 處理音視頻時長不一致 |
| `kling` | 影片轉影片 | 最平，源片限 2–10 秒 |
| `omnihuman` | 圖生影片 | 圖 + 音直接出片，talking head 專用 |

單人鏡頭用 `audio`；分割畫面用 `panels`（`crop` 支援 `left`/`right`/`top`/`bottom` 或 `x,y,w,h`）。

## 成本提醒

- `aifilm-lipsync --dry-run` 會印出每個鏡頭的預估美金費用，正式跑之前先看一次。
- 引擎價錢是公開牌價，會變動，僅供預估。
- 影片會先 trim 到 `trim_to_seconds` 再上傳，按秒收費時別為要剪走的幀付錢。
