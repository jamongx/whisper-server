"""Model loading, inference, and GPU helpers (faster-whisper / CTranslate2)."""
import asyncio
import logging
import subprocess
import sys
import time

from config import settings

import ctranslate2
import numpy as np

logger = logging.getLogger("whisper-server")

device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

_model = None
_model_lock = asyncio.Lock()
# CTranslate2 models are not safe to drive from concurrent requests on a small
# GPU (and parallel jobs only thrash VRAM), so GPU jobs — transcription and
# diarization alike — run one at a time (see run_gpu_job).
_inference_lock = asyncio.Lock()

# Last known GPU state. Kept up to date by real transcriptions and the watchdog,
# so /health can answer without touching the GPU — any CUDA call wakes the card
# out of its low-power state (P8) for ~15-20 s.
_gpu_ok = True
_last_success = 0.0  # time.monotonic() of the last successful GPU job


def gpu_ok() -> bool:
    return _gpu_ok


def set_gpu_ok(ok: bool) -> None:
    global _gpu_ok
    _gpu_ok = ok


def seconds_since_last_success() -> float:
    return time.monotonic() - _last_success if _last_success else float("inf")


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


def _run(
    model, tmp_path: str, language: str | None, initial_prompt: str | None, word_timestamps: bool
) -> dict:
    segments, info = model.transcribe(
        load_audio(tmp_path),
        language=language,
        beam_size=settings.transcribe_beam_size,
        best_of=settings.transcribe_best_of,
        initial_prompt=initial_prompt or None,
        vad_filter=settings.transcribe_vad,
        condition_on_previous_text=settings.transcribe_condition_on_previous_text,
        word_timestamps=word_timestamps,
    )
    # `segments` is a lazy generator — decoding happens while iterating.
    segs = []
    for s in segments:
        seg = {"start": s.start, "end": s.end, "text": s.text}
        if word_timestamps:
            # Words keep Whisper's leading space (" 안녕하세요") so callers can
            # rebuild the text by plain concatenation.
            seg["words"] = [{"start": w.start, "end": w.end, "word": w.word} for w in s.words or []]
        segs.append(seg)
    return {
        "text": "".join(s["text"] for s in segs),
        "language": info.language,
        "duration": info.duration,
        "segments": segs,
    }


async def run_gpu_job(fn):
    """Run a blocking GPU job in the thread pool, one at a time.

    Shared by transcription and diarization so the two never compete for the
    6 GB card. A success records the GPU as healthy (the watchdog then skips its
    probe); a CUDA error is confirmed with an immediate probe. Out-of-memory is
    a property of the input, not a broken GPU, so it leaves the state alone.
    """
    global _last_success
    loop = asyncio.get_running_loop()
    async with _inference_lock:
        try:
            result = await loop.run_in_executor(None, fn)
        except RuntimeError as exc:
            if is_cuda_error(exc) and not is_out_of_memory(exc):
                # Confirm right away instead of waiting for the next watchdog round.
                set_gpu_ok(await loop.run_in_executor(None, cuda_healthy))
            raise
    _last_success = time.monotonic()
    set_gpu_ok(True)
    return result


def gpu_busy() -> bool:
    """True while a transcription or diarization holds the GPU."""
    return _inference_lock.locked()


async def transcribe_audio(
    tmp_path: str,
    language: str | None,
    initial_prompt: str | None = None,
    word_timestamps: bool = False,
) -> dict:
    model = await get_model()
    return await run_gpu_job(lambda: _run(model, tmp_path, language, initial_prompt, word_timestamps))


def is_cuda_error(exc: Exception) -> bool:
    """Classify an exception as a CUDA/GPU failure. Lives here (the GPU layer)
    so callers don't need to know the device-specific error vocabulary."""
    msg = str(exc).lower()
    return "cuda" in msg or "cublas" in msg or "cudnn" in msg or "device-side" in msg


def is_out_of_memory(exc: Exception) -> bool:
    """GPU out-of-memory (torch.OutOfMemoryError, CTranslate2 "out of memory").
    The device still works; the job was just too large for the free VRAM."""
    return "out of memory" in str(exc).lower()


def clear_cuda_cache() -> None:
    """Hand torch's cached VRAM back to the driver after a request.

    CTranslate2 manages its own allocator cache. torch (pyannote) keeps freed
    blocks cached, which would leave the diarization peak resident next to
    Whisper. Only acts once diarization has initialised torch's CUDA context,
    so plain transcriptions never import torch.
    """
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


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
