"""GPU health watchdog — a process-lifecycle concern kept out of the app module.

After CUDA_WATCHDOG_FAILURES consecutive probe failures it exits the process so
Docker's restart policy recreates a fresh container with a working CUDA context
(replaces a manual `docker restart`).
"""
import asyncio
import logging
import os

from config import settings
from transcriber import cuda_healthy

logger = logging.getLogger("whisper-server")


async def cuda_watchdog() -> None:
    failures = 0
    while True:
        await asyncio.sleep(settings.cuda_watchdog_interval)
        healthy = await asyncio.get_running_loop().run_in_executor(None, cuda_healthy)
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
