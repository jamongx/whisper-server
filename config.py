from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from environment variables / .env.

    Field names map to upper-case env vars case-insensitively
    (e.g. ``whisper_model`` <- ``WHISPER_MODEL``).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model & decoding
    # Any faster-whisper model name (small, medium, large-v3, large-v3-turbo, ...)
    # or a path / Hugging Face repo of a CTranslate2-converted model.
    whisper_model: str = "large-v3"
    # CTranslate2 quantization: int8 | int8_float16 | float16 | float32
    whisper_compute_type: str = "int8"
    default_language: str = "en"
    transcribe_beam_size: int = 5
    transcribe_best_of: int = 5
    # Silero VAD skips non-speech (silence, room noise) — fewer hallucinations.
    transcribe_vad: bool = True
    # Feeding the previous window's text back in helps consistency but lets a
    # hallucination loop persist; False is the safer default for long talks.
    transcribe_condition_on_previous_text: bool = False

    # Optional decoding hint biasing the model's vocabulary (e.g. product names).
    # Applied to every request unless the caller passes its own initial_prompt.
    default_initial_prompt: str = ""

    # CUDA watchdog — the container can silently lose GPU access if the host
    # runs `systemd daemon-reload` while the container is up. The watchdog
    # probes the GPU periodically and, after repeated failures, exits the
    # process so Docker's restart policy brings up a fresh (working) container.
    # Probing wakes the GPU from its low-power state, so keep the interval long;
    # a successful transcription counts as a probe.
    cuda_watchdog_interval: float = 600.0
    cuda_watchdog_retry: float = 15.0  # re-probe interval once the GPU looks unhealthy
    cuda_watchdog_failures: int = 3

    # Load the model at startup (warms the cache and surfaces GPU problems early).
    warmup_on_start: bool = True

    # Internal constants (overridable via env, but rarely changed).
    app_title: str = "Local Whisper Server"
    default_fallback_suffix: str = ".audio"


settings = Settings()
