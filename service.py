"""Transcription use-case: turns an uploaded audio blob into a response model.

Keeps file materialisation, result mapping, and GPU-cache cleanup out of the
web layer, so the route handler stays a thin HTTP adapter.
"""
import tempfile
from pathlib import Path

from config import settings
from schemas import Segment, TranscribeResponse
from transcriber import clear_cuda_cache, transcribe_audio


async def transcribe_upload(
    audio_bytes: bytes,
    filename: str | None,
    language: str,
    initial_prompt: str = "",
) -> TranscribeResponse:
    # Whisper reads from a path, so the in-memory upload is spooled to a temp
    # file. This is an infrastructure detail hidden from the web layer.
    suffix = Path(filename).suffix if filename else settings.default_fallback_suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    prompt = initial_prompt or settings.default_initial_prompt
    # An empty value or "auto" lets Whisper detect the language itself
    # (passing None). Any other value forces that language.
    lang = None if language.strip().lower() in ("", "auto") else language
    try:
        result = await transcribe_audio(tmp_path, lang, prompt)
        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in result.get("segments", [])
        ]
        # Whisper doesn't report audio length, so derive it from the end of the
        # last segment (0.0 when there's no speech / no segments).
        duration = segments[-1].end if segments else 0.0
        return TranscribeResponse(
            text=result["text"].strip(),
            language=result["language"],
            duration=duration,
            segments=segments,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        clear_cuda_cache()
