"""Model loading, inference, and GPU helpers (faster-whisper / CTranslate2)."""
import asyncio
import logging
import subprocess

from config import settings

import ctranslate2
import numpy as np

logger = logging.getLogger("whisper-server")

device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

_model = None
_model_lock = asyncio.Lock()
# CTranslate2 models are not safe to drive from concurrent requests on a small
# GPU (and parallel jobs only thrash VRAM), so transcriptions run one at a time.
_inference_lock = asyncio.Lock()


async def get_model():
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel
                loop = asyncio.get_running_loop()
                compute_type = settings.whisper_compute_type if device == "cuda" else "int8"
                logger.info("Loading %s (%s, %s)", settings.whisper_model, device, compute_type)
                _model = await loop.run_in_executor(
                    None,
                    lambda: WhisperModel(settings.whisper_model, device=device, compute_type=compute_type),
                )
    return _model


SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    """Decode any media file to mono 16 kHz float32 with ffmpeg. Passing an
    array to faster-whisper bypasses its PyAV decoder, which breaks whenever
    PyAV changes its open() signature."""
    pcm = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
         "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


def _run(model, tmp_path: str, language: str | None, initial_prompt: str | None) -> dict:
    segments, info = model.transcribe(
        load_audio(tmp_path),
        language=language,
        beam_size=settings.transcribe_beam_size,
        best_of=settings.transcribe_best_of,
        initial_prompt=initial_prompt or None,
        vad_filter=settings.transcribe_vad,
        condition_on_previous_text=settings.transcribe_condition_on_previous_text,
    )
    # `segments` is a lazy generator — decoding happens while iterating.
    segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    return {
        "text": "".join(s["text"] for s in segs),
        "language": info.language,
        "duration": info.duration,
        "segments": segs,
    }


async def transcribe_audio(tmp_path: str, language: str | None, initial_prompt: str | None = None) -> dict:
    model = await get_model()
    async with _inference_lock:
        # Run in a thread pool to avoid blocking the event loop
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: _run(model, tmp_path, language, initial_prompt)
        )


def is_cuda_error(exc: Exception) -> bool:
    """Classify an exception as a CUDA/GPU failure. Lives here (the GPU layer)
    so callers don't need to know the device-specific error vocabulary."""
    msg = str(exc).lower()
    return "cuda" in msg or "cublas" in msg or "cudnn" in msg or "device-side" in msg


def clear_cuda_cache() -> None:
    """CTranslate2 manages its own allocator cache; nothing to release per request."""


def cuda_healthy() -> bool:
    """Actively probe the GPU by copying a tiny buffer to the device and back.
    NVML (nvidia-smi) can report "Unknown Error" while CUDA keeps working, so
    a real CUDA round-trip is the only trustworthy signal."""
    if device != "cuda":
        return True  # CPU-only deployments are always "healthy"
    try:
        x = ctranslate2.StorageView.from_array(np.ones((4,), dtype=np.float32))
        y = np.array(x.to_device(ctranslate2.Device.cuda).to_device(ctranslate2.Device.cpu))
        return bool(y.sum() == 4.0)
    except Exception:  # noqa: BLE001
        return False
