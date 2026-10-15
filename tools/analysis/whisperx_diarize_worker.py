"""WhisperX alignment + diarization worker.

Runs inside whichever interpreter has WhisperX installed — usually the
isolated ``.venv-whisperx`` side environment, because WhisperX pins an older
torch/transformers than the main venv. It must not import anything from
OpenMontage so it can run under that interpreter.

Usage: python whisperx_diarize_worker.py <request.json> <response.json>

Request: {"audio_path": str, "segments": [...], "language": str}
Response: {"segments": [...], "status": "applied", "note": str}
Requires HF_TOKEN (the pyannote diarization models are gated).
"""

from __future__ import annotations

import json
import os
import sys


def main(request_path: str, response_path: str) -> int:
    with open(request_path, encoding="utf-8") as fh:
        request = json.load(fh)

    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = whisperx.load_audio(request["audio_path"])

    align_model, align_metadata = whisperx.load_align_model(
        language_code=request["language"], device=device
    )
    aligned = whisperx.align(
        request["segments"], align_model, align_metadata, audio, device=device
    )

    from whisperx.diarize import DiarizationPipeline

    diarize_model = DiarizationPipeline(token=os.environ["HF_TOKEN"], device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, aligned)
    response = {
        "segments": result["segments"],
        "status": "applied",
        "note": f"diarized on {device}",
    }

    with open(response_path, "w", encoding="utf-8") as fh:
        # WhisperX can return numpy scalars; coerce them to plain floats.
        json.dump(response, fh, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
