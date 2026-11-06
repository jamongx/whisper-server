"""Transcription and diarization use-cases: turn an uploaded audio blob into a
response model.

Keeps file materialisation, result mapping, and GPU-cache cleanup out of the
web layer, so the route handlers stay thin HTTP adapters.
"""
import tempfile
from contextlib import contextmanager
from pathlib import Path

from config import settings
from diarizer import diarize_audio
from schemas import DiarizeResponse, Segment, SpeakerTurn, TranscribeResponse, Word
from transcriber import clear_cuda_cache, transcribe_audio


@contextmanager
def _spooled(audio_bytes: bytes, filename: str | None):
    # The models read from a path, so the in-memory upload is spooled to a temp
    # file. This is an infrastructure detail hidden from the web layer.
    suffix = Path(filename).suffix if filename else settings.default_fallback_suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        yield tmp_path
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        clear_cuda_cache()


async def transcribe_upload(
    audio_bytes: bytes,
    filename: str | None,
    language: str,
    initial_prompt: str = "",
    word_timestamps: bool = False,
) -> TranscribeResponse:
    prompt = initial_prompt or settings.default_initial_prompt
    # An empty value or "auto" lets Whisper detect the language itself
    # (passing None). Any other value forces that language.
    lang = None if language.strip().lower() in ("", "auto") else language
    with _spooled(audio_bytes, filename) as tmp_path:
        result = await transcribe_audio(tmp_path, lang, prompt, word_timestamps)
    segments = [
        Segment(
            start=s["start"],
            end=s["end"],
            text=s["text"].strip(),
            words=[Word(**w) for w in s["words"]] if "words" in s else None,
        )
        for s in result.get("segments", [])
    ]
    # Duration = end of the last segment (0.0 when there's no speech). The
    # SRT timecodes come from segment start/end, so this is informational.
    duration = segments[-1].end if segments else 0.0
    return TranscribeResponse(
        text=result["text"].strip(),
        language=result["language"],
        duration=duration,
        segments=segments,
    )


def validate_speaker_counts(
    num_speakers: int | None, min_speakers: int | None, max_speakers: int | None
) -> None:
    """Reject contradictory hints instead of letting pyannote silently ignore
    them (it drops min/max when num_speakers is given)."""
    for name, value in (("num_speakers", num_speakers), ("min_speakers", min_speakers),
                        ("max_speakers", max_speakers)):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1")
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise ValueError("pass either num_speakers or min_speakers/max_speakers, not both")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ValueError("min_speakers must not exceed max_speakers")


async def diarize_upload(
    audio_bytes: bytes,
    filename: str | None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> DiarizeResponse:
    validate_speaker_counts(num_speakers, min_speakers, max_speakers)
    with _spooled(audio_bytes, filename) as tmp_path:
        result = await diarize_audio(tmp_path, num_speakers, min_speakers, max_speakers)
    segments = [SpeakerTurn(**s) for s in result["segments"]]
    return DiarizeResponse(
        duration=result["duration"],
        num_speakers=len({s.speaker for s in segments}),
        segments=segments,
    )
