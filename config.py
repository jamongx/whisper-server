import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from environment variables / .env.

    Field names map to upper-case env vars case-insensitively
    (e.g. ``whisper_model`` <- ``WHISPER_MODEL``).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model & decoding
    whisper_model: str = "small"
    default_language: str = "en"
    transcribe_fp16: bool = False
    transcribe_beam_size: int = 1
    transcribe_best_of: int = 1
    cuda_alloc_conf: str = "max_split_size_mb:256"

    # Optional decoding hint biasing the model's vocabulary (e.g. product names).
    # Applied to every request unless the caller passes its own initial_prompt.
    default_initial_prompt: str = ""

    # CUDA watchdog — the container can silently lose GPU access (NVML "Unknown
    # Error") if the host runs `systemd daemon-reload` while the container is up.
    # The watchdog probes the GPU periodically and, after repeated failures, exits
    # the process so Docker's restart policy brings up a fresh (working) container.
    cuda_watchdog_interval: float = 30.0
    cuda_watchdog_failures: int = 3

    # Load the model at startup (warms the cache and surfaces GPU problems early).
    warmup_on_start: bool = True

    # Internal constants (overridable via env, but rarely changed).
    app_title: str = "Local Whisper Server"
    default_fallback_suffix: str = ".audio"


settings = Settings()

# Set the CUDA allocator config BEFORE torch is imported anywhere. torch reads
# PYTORCH_CUDA_ALLOC_CONF when the CUDA context initialises, so this must run
# first — hence every torch-importing module imports `config` above `torch`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", settings.cuda_alloc_conf)
