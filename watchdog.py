"""GPU health watchdog — a process-lifecycle concern kept out of the app module.

Every CUDA call pulls the GPU out of its low-power state for ~15-20 s, so the
watchdog probes rarely (CUDA_WATCHDOG_INTERVAL, default 10 min) and skips the
probe when a transcription succeeded within that window. Once the GPU looks
unhealthy (failed probe, or a CUDA error during a request) it re-probes every
CUDA_WATCHDOG_RETRY seconds; after CUDA_WATCHDOG_FAILURES consecutive failures
it exits the process so Docker's restart policy recreates a fresh container
with a working CUDA context (replaces a manual `docker restart`).
"""
import asyncio
import logging
import os

from config import settings
from transcriber import cuda_healthy, gpu_ok, seconds_since_last_success, set_gpu_ok

logger = logging.getLogger("whisper-server")


async def cuda_watchdog() -> None:
    failures = 0
    while True:
        healthy_before = gpu_ok()
        await asyncio.sleep(
            settings.cuda_watchdog_interval if healthy_before else settings.cuda_watchdog_retry
        )
        if gpu_ok() and seconds_since_last_success() < settings.cuda_watchdog_interval:
            failures = 0  # a recent transcription already proved the GPU works
            continue
        healthy = await asyncio.get_running_loop().run_in_executor(None, cuda_healthy)
        set_gpu_ok(healthy)
        if healthy:
            failures = 0
            continue
        failures += 1
        logger.warning(
            "CUDA health probe failed (%d/%d)", failures, settings.cuda_watchdog_failures
        )
        if failures >= settings.cuda_watchdog_failures:
            logger.error("CUDA unavailable — exiting for restart to recover the GPU context.")
            os._exit(1)
