import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.video.video_compose import VideoCompose


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)


def _run(command: list[str]) -> None:
    subprocess.run(command, capture_output=True, check=True, timeout=120)


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,avg_frame_rate,pix_fmt:format=duration",
         "-of", "json", str(path)],
        capture_output=True, check=True, text=True, timeout=30,
    ).stdout
    return json.loads(out)


def _max_volume(path: Path, start: float = 0.0, duration: float = 60.0) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-t", str(duration),
         "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, timeout=60,
    )
    line = next(l for l in proc.stderr.splitlines() if "max_volume:" in l)
    return float(line.split("max_volume:")[1].split("dB")[0])


def _silent_video(path: Path, seconds: float = 3.0, size: str = "320x180", rate: int = 30) -> Path:
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:s={size}:d={seconds}:r={rate}",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    return path


def _tone(path: Path, seconds: float, freq: int = 440) -> Path:
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}", str(path)])
    return path


# ---------------------------------------------------------------- music bed


def test_music_bed_loops_short_track_to_video_length(tmp_path) -> None:
    video = _silent_video(tmp_path / "v.mp4", seconds=4)
    music = _tone(tmp_path / "m.wav", seconds=1.5)

    result = VideoCompose()._mix_music_into_render(
        video, music, {"fade_in_seconds": 0.2, "fade_out_seconds": 0.5},
    )

    assert result.success, result.error
    assert result.data["kept_render_audio"] is False
    info = _probe(video)
    assert {s["codec_type"] for s in info["streams"]} == {"video", "audio"}
    assert float(info["format"]["duration"]) == pytest.approx(4.0, abs=0.15)
    # Looped, not ending after the 1.5 s source: still sounding at 2.5-3.2 s.
    assert _max_volume(video, start=2.5, duration=0.7) > -30


def test_music_is_ducked_under_existing_narration(tmp_path) -> None:
    video = tmp_path / "narrated.mp4"
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=3:r=30",
          "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)])
    music = _tone(tmp_path / "m.wav", seconds=5, freq=880)

    result = VideoCompose()._mix_music_into_render(video, music, {})

    assert result.success, result.error
    assert result.data["kept_render_audio"] is True
    assert result.data["music_ducked"] is True
    assert result.data["music_volume"] == pytest.approx(0.2)


def test_missing_music_asset_fails_before_render() -> None:
    with pytest.raises(FileNotFoundError, match="bed"):
        VideoCompose._music_config({"audio": {"music": {"asset_id": "bed"}}}, {})


def test_legacy_top_level_music_is_resolved_through_manifest(tmp_path) -> None:
    track = _tone(tmp_path / "t.wav", seconds=1)
    cfg, path = VideoCompose._music_config(
        {"music": {"asset_id": "bed", "volume": 0.5}}, {"bed": {"path": str(track)}},
    )
    assert path == track
    assert cfg["volume"] == 0.5


# ---------------------------------------------------------- normalization


def test_mismatched_clip_is_conformed_and_cut_rewritten(tmp_path) -> None:
    odd = tmp_path / "odd.mkv"
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=s=320x240:d=1:r=25",
          "-c:v", "mpeg4", str(odd)])
    good = _silent_video(tmp_path / "good.mp4", seconds=1, size="640x360")
    cuts = [
        {"id": "a", "source": str(odd), "in_seconds": 0, "out_seconds": 1},
        {"id": "b", "source": str(good), "in_seconds": 0, "out_seconds": 1},
        {"id": "c", "source": str(odd), "in_seconds": 0, "out_seconds": 0.5},
    ]

    report = VideoCompose()._normalize_clip_sources(cuts, tmp_path / "norm", "youtube_shorts")

    assert len(report["normalized"]) == 2  # odd.mkv once, good.mp4 is 640x360 not 1080x1920
    assert cuts[0]["source"] == cuts[2]["source"] != str(odd)
    stream = _probe(Path(cuts[0]["source"]))["streams"][0]
    assert (stream["codec_name"], stream["pix_fmt"]) == ("h264", "yuv420p")
    assert (stream["width"], stream["height"]) == (1080, 1920)
    assert stream["avg_frame_rate"] == "30/1"

    # Second run hits the cache instead of re-encoding.
    again = [dict(cuts[0], source=str(odd))]
    report2 = VideoCompose()._normalize_clip_sources(again, tmp_path / "norm", "youtube_shorts")
    assert report2["normalized"][0]["reason"].endswith("(cached)")


def test_letterbox_bars_are_detected(tmp_path) -> None:
    boxed = tmp_path / "boxed.mp4"
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=s=640x260:d=3:r=30",
          "-vf", "pad=640:360:0:50:black", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(boxed)])

    crop = VideoCompose._detect_letterbox(boxed, 640, 360)

    assert crop is not None
    w, h, _, y = (int(v) for v in crop.removeprefix("crop=").split(":"))
    assert w == 640 and 256 <= h <= 264 and 48 <= y <= 52
    assert VideoCompose._detect_letterbox(_silent_video(tmp_path / "full.mp4"), 320, 180) is None


# ------------------------------------------------------------ end to end


def test_render_normalizes_clips_and_mixes_declared_music(monkeypatch, tmp_path) -> None:
    clip = tmp_path / "clip.webm"
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=s=320x240:d=2:r=24",
          "-c:v", "libvpx-vp9", "-b:v", "200k", str(clip)])
    music = _tone(tmp_path / "bed.wav", seconds=10)
    output = tmp_path / "out" / "final.mp4"
    seen_sources: list[str] = []

    def fake_compose(self, inputs):
        cut = inputs["edit_decisions"]["cuts"][0]
        seen_sources.append(cut["source"])
        _run(["ffmpeg", "-y", "-i", cut["source"], "-an", "-c:v", "copy", inputs["output_path"]])
        from tools.base_tool import ToolResult
        return ToolResult(success=True, data={"output": inputs["output_path"]})

    monkeypatch.setattr(VideoCompose, "_compose", fake_compose)
    monkeypatch.setattr(VideoCompose, "_pre_compose_validation", lambda self, *a, **k: None)
    monkeypatch.setattr(
        VideoCompose, "_run_final_review",
        lambda self, *a, **k: {"status": "pass", "issues_found": []},
    )

    result = VideoCompose().execute({
        "operation": "render",
        "output_path": str(output),
        "profile": "youtube_landscape",
        "edit_decisions": {
            "version": "1.0",
            "render_runtime": "ffmpeg",
            "cuts": [{"id": "c1", "source": "clip", "in_seconds": 0, "out_seconds": 2}],
            "audio": {"music": {"asset_id": "bed", "fade_out_seconds": 0.5}},
        },
        "asset_manifest": {"assets": [
            {"id": "clip", "path": str(clip)},
            {"id": "bed", "path": str(music)},
        ]},
    })

    assert result.success, result.error
    assert Path(seen_sources[0]).parent == output.parent / ".normalized"
    assert result.data["clip_normalization"]["normalized"][0]["source"] == str(clip.resolve())
    assert result.data["music_mix"]["kept_render_audio"] is False
    info = _probe(output)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1920, 1080)
    assert any(s["codec_type"] == "audio" for s in info["streams"])


def test_silence_window_mutes_the_bed(tmp_path) -> None:
    video = _silent_video(tmp_path / "v.mp4", seconds=4)
    music = _tone(tmp_path / "m.wav", seconds=4)
    cfg, path = VideoCompose._music_config(
        {"audio": {"music": {"asset_id": "bed", "fade_in_seconds": 0, "fade_out_seconds": 0}},
         "metadata": {"silence_window": {"start_seconds": 1.5, "end_seconds": 2.5}}},
        {"bed": {"path": str(music)}},
    )

    result = VideoCompose()._mix_music_into_render(video, path, cfg)

    assert result.success, result.error
    assert _max_volume(video, start=1.8, duration=0.4) < -50
    assert _max_volume(video, start=0.5, duration=0.5) > -30
