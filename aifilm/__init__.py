"""AI film pipeline: video generation, TTS dubbing, and lip sync.

Reusable CLI tools for producing short videos with dubbed speech:

- ``aifilm.videos``  : fal.ai image-to-video (Kling / Seedance / MiniMax).
- ``aifilm.veo``     : Google Veo image-to-video with native audio and lip sync.
- ``aifilm.lipsync`` : fal.ai lip sync (sync.so / kling / omnihuman).
- ``aifilm.tts``     : MiniMax text-to-speech from an SRT script.

Run them directly:

    python -m aifilm.videos --jobs board1/video_jobs.json --dry-run
    python -m aifilm.veo --jobs board1/video_jobs.json --dry-run
    python -m aifilm.lipsync --jobs board1/video_jobs.json --dry-run
    python -m aifilm.tts --srt script.srt --cast voice_cast.json --dry-run

Or install the package and use the console entry points:

    pip install -e .
    aifilm-veo --jobs board1/video_jobs.json --dry-run
"""

__all__ = ["videos", "veo", "lipsync", "tts"]
