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
  HF_TOKEN               — Hugging Face token for the gated diarization model;
                           without it the server runs and only /diarize returns 503
  DIARIZE_MODEL          — pyannote pipeline (default: pyannote/speaker-diarization-community-1)
  DIARIZE_SEGMENTATION_BATCH_SIZE / DIARIZE_EMBEDDING_BATCH_SIZE
                         — override the pipeline's batch sizes (default: 0 = pipeline config)
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from diarizer import DiarizationUnavailable
from schemas import DiarizeResponse, TranscribeResponse
from service import diarize_upload, transcribe_upload
from transcriber import get_model, gpu_ok, is_cuda_error, is_out_of_memory, set_gpu_ok
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


def _gpu_error_response(exc: RuntimeError, job: str) -> JSONResponse | None:
    """Map GPU failures to 503; None means "not a GPU error, re-raise"."""
    if is_out_of_memory(exc):
        logger.error("GPU out of memory during %s", job)
        return JSONResponse(
            status_code=503,
            content={"detail": "GPU ran out of memory for this request; try a shorter file."},
        )
    if is_cuda_error(exc):
        logger.error("CUDA error during %s: %s", job, exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "GPU unavailable; server is recovering, retry shortly."},
        )
    return None


# exclude_none keeps the default response identical to before word_timestamps
# existed: segments carry "words" only when it was asked for.
@app.post("/transcribe", response_model=TranscribeResponse, response_model_exclude_none=True)
async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form(default=settings.default_language),
        initial_prompt: str = Form(default=""),
        word_timestamps: bool = Form(default=False),
        ):
    audio_bytes = await audio.read()
    try:
        return await transcribe_upload(
            audio_bytes, audio.filename, language, initial_prompt, word_timestamps
        )
    except RuntimeError as exc:
        if (response := _gpu_error_response(exc, "transcription")) is not None:
            return response
        raise


@app.post("/diarize", response_model=DiarizeResponse)
async def diarize(
        audio: UploadFile = File(...),
        num_speakers: int | None = Form(default=None),
        min_speakers: int | None = Form(default=None),
        max_speakers: int | None = Form(default=None),
        ):
    """Speaker turns for the whole file in one pass, so labels stay consistent
    across a long recording. Send the full file — not the 60-minute chunks."""
    audio_bytes = await audio.read()
    try:
        return await diarize_upload(
            audio_bytes, audio.filename, num_speakers, min_speakers, max_speakers
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except DiarizationUnavailable as exc:
        logger.error("Diarization unavailable: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    except RuntimeError as exc:
        if (response := _gpu_error_response(exc, "diarization")) is not None:
            return response
        raise


@app.get("/health")
async def health() -> JSONResponse:
    # Last known state only — no GPU call here, so Docker's 30 s healthcheck
    # doesn't keep the card out of its low-power state.
    if gpu_ok():
        return JSONResponse(status_code=200, content={"status": "ok", "cuda": True})
    return JSONResponse(status_code=503, content={"status": "degraded", "cuda": False})
