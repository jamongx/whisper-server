import asyncio

# `config` must be imported before `torch` so PYTORCH_CUDA_ALLOC_CONF is set
# before the CUDA context initialises (see config.py).
from config import settings

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_model_lock = asyncio.Lock()


async def get_model():
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:
                import whisper
                loop = asyncio.get_running_loop()
                _model = await loop.run_in_executor(
                    None, lambda: whisper.load_model(settings.whisper_model, device=device)
                )
    return _model


async def transcribe_audio(tmp_path: str, language: str | None, initial_prompt: str | None = None) -> dict:
    model = await get_model()
    loop = asyncio.get_running_loop()
    # Run in a thread pool to avoid blocking the event loop
    result = await loop.run_in_executor(
        None,
        lambda: model.transcribe(
            tmp_path,
            language=language,
            fp16=settings.transcribe_fp16,
            beam_size=settings.transcribe_beam_size,
            best_of=settings.transcribe_best_of,
            initial_prompt=initial_prompt or None,
        ),
    )
    return result


def is_cuda_error(exc: Exception) -> bool:
    """Classify an exception as a CUDA/GPU failure. Lives here (the GPU layer)
    so callers don't need to know the device-specific error vocabulary."""
    msg = str(exc).lower()
    return "cuda" in msg or "nvml" in msg or "device-side" in msg


def clear_cuda_cache() -> None:
    """Best-effort cache release. Never raises — a dead device must not mask
    the real response or crash the request path."""
    if device != "cuda":
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — device may be lost; watchdog handles recovery
        pass


def cuda_healthy() -> bool:
    """Actively probe the GPU with a tiny op. `torch.cuda.is_available()` can
    return a stale cached True after the device is lost, so we run a real op."""
    if device != "cuda":
        return True  # CPU-only deployments are always "healthy"
    try:
        torch.zeros(1, device="cuda").add_(1.0)
        torch.cuda.synchronize()
        return True
    except Exception:  # noqa: BLE001
        return False
