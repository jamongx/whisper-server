"""
Local Whisper FastAPI server — runs on GPU machine.

Usage:
  bash run.sh

Environment variables (see .env.example):
  WHISPER_MODEL          — faster-whisper model name (default: large-v3)
  WHISPER_COMPUTE_TYPE   — CTranslate2 quantization (default: int8)
  DEFAULT_LANGUAGE       — transcription language (default: en)
  TRANSCRIBE_BEAM_SIZE   — beam search width (default: 5)
  TRANSCRIBE_BEST_OF     — best-of candidates (default: 5)
  TRANSCRIBE_VAD         — skip non-speech with Silero VAD (default: true)
  DEFAULT_INITIAL_PROMPT — vocabulary hint applied when none is supplied
  CUDA_WATCHDOG_INTERVAL — seconds between idle GPU probes (default: 600)
  CUDA_WATCHDOG_RETRY    — re-probe interval once the GPU looks unhealthy (default: 15)
  CUDA_WATCHDOG_FAILURES — consecutive failures before self-restart (default: 3)
  WARMUP_ON_START        — load the model at startup (default: true)
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from schemas import TranscribeResponse
from service import transcribe_upload
from transcriber import get_model, gpu_ok, is_cuda_error, set_gpu_ok
from watchdog import cuda_watchdog

logger = logging.getLogger("whisper-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.warmup_on_start:
        try:
            await get_model()
            logger.info("Model warmed up and ready.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Warmup failed: %s", exc)
            if isinstance(exc, RuntimeError) and is_cuda_error(exc):
                set_gpu_ok(False)  # let the watchdog re-probe and restart if it persists
    watchdog = asyncio.create_task(cuda_watchdog())
    try:
        yield
    finally:
        watchdog.cancel()


app = FastAPI(title=settings.app_title, lifespan=lifespan)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form(default=settings.default_language),
        initial_prompt: str = Form(default=""),
        ):
    audio_bytes = await audio.read()
    try:
        return await transcribe_upload(audio_bytes, audio.filename, language, initial_prompt)
    except RuntimeError as exc:
        if is_cuda_error(exc):
            logger.error("CUDA error during transcription: %s", exc)
            return JSONResponse(
                status_code=503,
                content={"detail": "GPU unavailable; server is recovering, retry shortly."},
            )
        raise


@app.get("/health")
async def health() -> JSONResponse:
    # Last known state only — no GPU call here, so Docker's 30 s healthcheck
    # doesn't keep the card out of its low-power state.
    if gpu_ok():
        return JSONResponse(status_code=200, content={"status": "ok", "cuda": True})
    return JSONResponse(status_code=503, content={"status": "degraded", "cuda": False})
