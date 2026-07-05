"""
Local Whisper FastAPI server — runs on GPU machine.

Usage:
  bash run.sh

Environment variables (see .env):
  WHISPER_MODEL          — model size to load (default: small)
  DEFAULT_LANGUAGE       — transcription language (default: en)
  TRANSCRIBE_FP16        — use fp16 inference (default: false)
  TRANSCRIBE_BEAM_SIZE   — beam search width (default: 1)
  TRANSCRIBE_BEST_OF     — best-of candidates (default: 1)
  DEFAULT_INITIAL_PROMPT — vocabulary hint applied when none is supplied
  CUDA_WATCHDOG_INTERVAL — seconds between GPU health probes (default: 30)
  CUDA_WATCHDOG_FAILURES — consecutive failures before self-restart (default: 3)
  WARMUP_ON_START        — load the model at startup (default: true)
  CUDA_ALLOC_CONF        — PYTORCH_CUDA_ALLOC_CONF value (default: max_split_size_mb:256)
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from schemas import TranscribeResponse
from service import transcribe_upload
from transcriber import cuda_healthy, get_model, is_cuda_error
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
    healthy = await asyncio.get_running_loop().run_in_executor(None, cuda_healthy)
    if healthy:
        return JSONResponse(status_code=200, content={"status": "ok", "cuda": True})
    return JSONResponse(status_code=503, content={"status": "degraded", "cuda": False})
