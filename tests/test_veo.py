from __future__ import annotations

import json
from pathlib import Path

import pytest

from aifilm.veo import compose_prompt, resolve_model_key, snap_duration, run
from aifilm import veo as veo_mod


def test_resolve_model_aliases() -> None:
    assert resolve_model_key("veo-3.1") == "veo"
    assert resolve_model_key("veo-3.1-fast") == "veo-fast"
    assert resolve_model_key("veo") == "veo"
    with pytest.raises(ValueError, match="fal.ai"):
        resolve_model_key("kling-o3")


def test_snap_duration_nearest_and_high_res() -> None:
    assert snap_duration("veo", 5, "720p") == 4
    assert snap_duration("veo", 7, "720p") == 6
    assert snap_duration("veo", 8, "720p") == 8
    assert snap_duration("veo", 4, "1080p") == 8
    assert snap_duration("veo", "auto", "720p") == 8


def test_compose_prompt_injects_quoted_dialogue() -> None:
    prompt = compose_prompt(
        {
            "id": "1.1",
            "prompt": "A man sits at a table in a grey room.",
            "dialogue": "I didn't steal anything.",
            "speaker_direction": "defensive, slightly too loud",
        },
        None,
    )
    assert "A man sits at a table" in prompt
    assert '"I didn\'t steal anything."' in prompt
    assert "Lip-sync" in prompt
    assert "defensive, slightly too loud" in prompt


def test_compose_prompt_does_not_duplicate_existing_quote() -> None:
    original = 'A man says "I didn\'t steal anything." looking at the camera.'
    prompt = compose_prompt(
        {"id": "1.1", "prompt": original, "dialogue": "I didn't steal anything."},
        None,
    )
    assert prompt == original


def test_dry_run_prints_jobs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image_dir = tmp_path / "keyframes"
    image_dir.mkdir()
    image = image_dir / "1.1_v1.jpg"
    image.write_bytes(b"\xff\xd8\xff")  # tiny jpeg-like bytes, never decoded in dry-run
    jobs_path = tmp_path / "video_jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "board": "board1",
                "image_dir": "keyframes",
                "output_dir": "video",
                "defaults": {"model": "veo", "duration": 8},
                "jobs": [
                    {
                        "id": "1.1",
                        "image": "1.1_v1.jpg",
                        "prompt": "A woman looks up from her phone.",
                        "dialogue": "This is a scam.",
                        "output_name": "1.1_veo.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run(jobs_path=jobs_path, only=None, dry_run=True, overwrite=False, root=tmp_path, poll_interval=10)

    out = capsys.readouterr().out
    assert "[DRY] 1.1 -> veo" in out
    assert "dialogue: This is a scam." in out
    assert "1.1_veo.mp4" in out
    assert "Lip-sync" in out


def test_dry_run_snaps_duration(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image_dir = tmp_path / "keyframes"
    image_dir.mkdir()
    (image_dir / "shot.jpg").write_bytes(b"\xff\xd8\xff")
    jobs_path = tmp_path / "video_jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "image_dir": "keyframes",
                "output_dir": "video",
                "jobs": [
                    {
                        "id": "x",
                        "model": "veo",
                        "image": "shot.jpg",
                        "prompt": "Idle.",
                        "duration": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run(jobs_path=jobs_path, only=None, dry_run=True, overwrite=False, root=tmp_path, poll_interval=10)
    out = capsys.readouterr().out
    assert "duration: 4s (snapped from 5)" in out


def test_using_vertex_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    assert veo_mod.using_vertex() is False
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    assert veo_mod.using_vertex() is True


def test_dry_run_skips_fal_preset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs_path = tmp_path / "video_jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "image_dir": "keyframes",
                "output_dir": "video",
                "defaults": {"model": "kling-o3"},
                "jobs": [{"id": "1.1", "prompt": "x", "image": "missing.jpg"}],
            }
        ),
        encoding="utf-8",
    )
    run(jobs_path=jobs_path, only=None, dry_run=True, overwrite=False, root=tmp_path, poll_interval=10)
    out = capsys.readouterr().out
    assert "[SKIP] 1.1" in out
    assert "fal.ai" in out
