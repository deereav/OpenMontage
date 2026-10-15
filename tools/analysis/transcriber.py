"""Transcription tool wrapping faster-whisper / WhisperX.

Provides speech-to-text with word-level timestamps and optional speaker
diarization. Falls back gracefully when GPU or diarization dependencies
are not available.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ResumeSupport,
    ToolResult,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class Transcriber(BaseTool):
    name = "transcriber"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "analysis"
    provider = "whisperx"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["python:faster_whisper"]
    install_instructions = (
        "pip install faster-whisper  # CPU mode\n"
        "pip install faster-whisper[gpu]  # GPU mode (requires CUDA)\n"
        "# Diarization: WhisperX pins an older torch/transformers, so install it\n"
        "# in a side venv (.venv-whisperx) or point OPENMONTAGE_WHISPERX_PYTHON at\n"
        "# an interpreter that has it; set HF_TOKEN for the pyannote models."
    )
    agent_skills = ["speech-to-text"]

    capabilities = [
        "transcribe",
        "word_timestamps",
        "diarization",
        "language_detection",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string", "description": "Path to audio or video file"},
            "model_size": {
                "type": "string",
                "enum": ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                "default": "base",
            },
            "language": {"type": "string", "description": "ISO 639-1 language code, or null for auto-detect"},
            "diarize": {"type": "boolean", "default": False},
            "output_dir": {"type": "string", "description": "Directory for output files"},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "segments": {"type": "array"},
            "word_timestamps": {"type": "array"},
            "language": {"type": "string"},
            "duration_seconds": {"type": "number"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2,
        ram_mb=2048,
        vram_mb=0,  # CPU by default; GPU optional
        disk_mb=500,
        network_required=False,
    )

    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["MemoryError"])
    resume_support = ResumeSupport.FROM_START
    idempotency_key_fields = ["input_path", "model_size", "language"]
    side_effects = ["writes transcript JSON to output_dir"]
    fallback = None
    user_visible_verification = [
        "Check transcript text against source audio",
        "Verify word timestamps align with speech",
    ]

    def get_status(self) -> ToolStatus:
        try:
            import faster_whisper  # noqa: F401
            return ToolStatus.AVAILABLE
        except ImportError:
            return ToolStatus.UNAVAILABLE

    def _whisperx_python(self) -> Optional[str]:
        """Interpreter that can import WhisperX, or None.

        Prefers OPENMONTAGE_WHISPERX_PYTHON, then the repo's .venv-whisperx
        side environment, then the current interpreter.
        """
        override = os.environ.get("OPENMONTAGE_WHISPERX_PYTHON")
        if override:
            return override if Path(override).exists() else None

        side_venv = Path(__file__).resolve().parents[2] / ".venv-whisperx"
        for candidate in (side_venv / "Scripts" / "python.exe", side_venv / "bin" / "python"):
            if candidate.exists():
                return str(candidate)

        try:
            import whisperx  # noqa: F401
            return sys.executable
        except ImportError:
            return None

    def _has_diarization(self) -> bool:
        return self._whisperx_python() is not None

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        """Rough estimate: ~0.5x real-time on CPU for 'base' model."""
        return 60.0  # conservative default

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        input_path = Path(inputs["input_path"])
        model_size = inputs.get("model_size", "base")
        language = inputs.get("language")
        diarize = inputs.get("diarize", False)
        output_dir = Path(inputs.get("output_dir", input_path.parent))

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input file not found: {input_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return ToolResult(
                success=False,
                error="faster-whisper is not installed. Run: pip install faster-whisper",
            )

        start = time.time()

        # faster-whisper executes through CTranslate2, so that runtime—not
        # PyTorch—is authoritative for CUDA availability and compute types.
        device = "cpu"
        compute_type = "int8"
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                supported = ctranslate2.get_supported_compute_types("cuda")
                for candidate in ("float16", "int8_float16", "float32"):
                    if candidate in supported:
                        device = "cuda"
                        compute_type = candidate
                        break
        except Exception:
            # Probing is advisory. CPU remains a safe deterministic baseline.
            pass

        def _transcribe_on(selected_device: str, selected_compute_type: str):
            model = WhisperModel(
                model_size,
                device=selected_device,
                compute_type=selected_compute_type,
            )
            segments_iter, transcription_info = model.transcribe(
                str(input_path),
                language=language,
                word_timestamps=True,
                vad_filter=True,
            )

            parsed_segments = []
            parsed_words = []
            # faster-whisper evaluates lazily. Draining the iterator here keeps
            # missing CUDA runtime libraries inside the fallback boundary.
            for seg in segments_iter:
                seg_data = {
                    "id": seg.id,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                }

                if seg.words:
                    words = []
                    for word in seg.words:
                        word_entry = {
                            "word": word.word,
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                            "probability": round(word.probability, 3),
                        }
                        words.append(word_entry)
                        parsed_words.append(word_entry)
                    seg_data["words"] = words

                parsed_segments.append(seg_data)

            return parsed_segments, parsed_words, transcription_info

        gpu_fallback_reason = None
        try:
            segments, word_timestamps, info = _transcribe_on(device, compute_type)
        except Exception as exc:
            if device == "cpu":
                raise
            gpu_fallback_reason = f"{type(exc).__name__}: {exc}"
            device = "cpu"
            compute_type = "int8"
            segments, word_timestamps, info = _transcribe_on(device, compute_type)

        detected_language = language or info.language
        duration = info.duration

        # Optional diarization pass
        diarization = None
        if diarize:
            segments, diarization = self._apply_diarization(
                str(input_path), segments, detected_language
            )

        elapsed = time.time() - start

        result_data = {
            "segments": segments,
            "word_timestamps": word_timestamps,
            "language": detected_language,
            "duration_seconds": round(duration, 3),
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type,
            "gpu_fallback_reason": gpu_fallback_reason,
            "diarization": diarization,
        }

        # Write transcript JSON
        output_path = output_dir / f"{input_path.stem}_transcript.json"
        output_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")

        return ToolResult(
            success=True,
            data=result_data,
            artifacts=[str(output_path)],
            duration_seconds=round(elapsed, 2),
        )

    def _apply_diarization(
        self,
        audio_path: str,
        segments: list[dict],
        language: str,
    ) -> tuple[list[dict], dict]:
        """Apply WhisperX alignment + diarization in a WhisperX interpreter.

        Best-effort: on any failure the original segments are returned, and
        the status dict says why.
        """
        python = self._whisperx_python()
        if python is None:
            return segments, {
                "status": "unavailable",
                "note": "WhisperX not found; see install_instructions",
            }
        if not os.environ.get("HF_TOKEN"):
            return segments, {
                "status": "skipped",
                "note": "HF_TOKEN not set; pyannote diarization models are gated on HuggingFace",
            }

        worker = Path(__file__).with_name("whisperx_diarize_worker.py")
        with tempfile.TemporaryDirectory() as tmp:
            request_path = Path(tmp) / "request.json"
            response_path = Path(tmp) / "response.json"
            request_path.write_text(
                json.dumps(
                    {"audio_path": audio_path, "segments": segments, "language": language}
                ),
                encoding="utf-8",
            )
            try:
                proc = subprocess.run(
                    [python, "-I", str(worker), str(request_path), str(response_path)],
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return segments, {"status": "failed", "note": f"{type(exc).__name__}: {exc}"}

            if proc.returncode != 0 or not response_path.exists():
                tail = (proc.stderr or "").strip().splitlines()[-5:]
                return segments, {"status": "failed", "note": "\n".join(tail)}

            response = json.loads(response_path.read_text(encoding="utf-8"))

        return response["segments"], {"status": response["status"], "note": response["note"]}
