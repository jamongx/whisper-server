"""Speaker diarization (pyannote.audio) — the GPU layer for POST /diarize.

The pipeline is loaded on the first request, never at startup: importing torch
and creating its CUDA context would otherwise cost VRAM and keep the card out
of P8 even on days nobody diarizes. Inference goes through run_gpu_job, so it
never overlaps a transcription.
"""
import asyncio
import logging
import os

import numpy as np

from config import settings
from transcriber import SAMPLE_RATE, load_audio, run_gpu_job

# pyannote.audio 4 ships with usage telemetry switched on (OTLP spans with audio
# duration and speaker counts sent to otel.pyannote.ai). Meeting metadata must
# stay on this machine, so turn it off before pyannote is ever imported. The
# Dockerfile sets the same variable; this covers runs outside the container.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

logger = logging.getLogger("whisper-server")

_pipeline = None
_pipeline_lock = asyncio.Lock()


class DiarizationUnavailable(Exception):
    """Diarization can't run on this server as configured (missing token, model
    terms not accepted, download failed). The message is safe to show clients."""


def _load_pipeline():
    if not settings.hf_token:
        raise DiarizationUnavailable(
            "Speaker diarization is not configured: set HF_TOKEN in .env "
            f"(a Hugging Face token whose account accepted the terms of {settings.diarize_model}) "
            "and restart the server."
        )
    import torch
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    from pyannote.audio import Pipeline

    logger.info("Loading diarization pipeline %s", settings.diarize_model)
    try:
        pipeline = Pipeline.from_pretrained(settings.diarize_model, token=settings.hf_token)
    except HfHubHTTPError as exc:
        # 401/403: bad token or the model's terms not accepted. Log the status
        # only — never the exception text or the token.
        status = getattr(exc.response, "status_code", None)
        logger.error("Hugging Face refused %s (HTTP %s)", settings.diarize_model, status)
        raise DiarizationUnavailable(
            f"Could not download {settings.diarize_model} (HTTP {status}). Check that HF_TOKEN is "
            "valid and that its account accepted the model's terms on huggingface.co, then retry."
        ) from None
    except LocalEntryNotFoundError:
        raise DiarizationUnavailable(
            f"{settings.diarize_model} is not cached and Hugging Face is unreachable "
            "(or HF_HUB_OFFLINE is set). Connect the server once to download it."
        ) from None
    if pipeline is None:  # older pyannote releases returned None on Hub errors
        raise DiarizationUnavailable(f"Could not load {settings.diarize_model}.")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    if settings.diarize_segmentation_batch_size > 0:
        pipeline.segmentation_batch_size = settings.diarize_segmentation_batch_size
    if settings.diarize_embedding_batch_size > 0:
        pipeline.embedding_batch_size = settings.diarize_embedding_batch_size
    logger.info(
        "Diarization pipeline ready (segmentation batch %s, embedding batch %s)",
        pipeline.segmentation_batch_size, pipeline.embedding_batch_size,
    )
    return pipeline


async def get_pipeline():
    global _pipeline
    if _pipeline is None:
        async with _pipeline_lock:
            if _pipeline is None:
                loop = asyncio.get_running_loop()
                _pipeline = await loop.run_in_executor(None, _load_pipeline)
    return _pipeline


def _run(
    pipeline,
    tmp_path: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict:
    import torch

    audio: np.ndarray = load_audio(tmp_path)
    # Hand pyannote the decoded waveform: one ffmpeg decode path for the whole
    # server, and torchcodec's file reader is never involved.
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (channel, time), no copy
    output = pipeline(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    # Exclusive diarization has no overlapping turns — one speaker at any
    # instant, which is what assigning words to speakers needs.
    annotation = output.exclusive_speaker_diarization
    segments = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    return {"duration": len(audio) / SAMPLE_RATE, "segments": segments}


async def diarize_audio(
    tmp_path: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    pipeline = await get_pipeline()
    return await run_gpu_job(
        lambda: _run(pipeline, tmp_path, num_speakers, min_speakers, max_speakers)
    )
