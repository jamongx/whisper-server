# whisper-server

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![faster-whisper](https://img.shields.io/badge/faster--whisper-CTranslate2-412991.svg)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900.svg?logo=nvidia&logoColor=white)

A lightweight local STT server running [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, int8) behind FastAPI, built for a laptop with a GTX 1660 Ti (6GB VRAM) on Ubuntu.

## Specs

| | |
|---|---|
| GPU | NVIDIA GTX 1660 Ti 6GB |
| OS | Ubuntu 24.04 |
| Runtime | Docker |
| Default model | large-v3-turbo (int8) |

## Project structure

```
whisper_server/
├── main.py            # FastAPI app and routes (thin web adapter)
├── service.py         # Transcription use-case (upload → response)
├── transcriber.py     # Model loading, inference, and GPU helpers
├── watchdog.py        # GPU health watchdog (self-restart on CUDA loss)
├── schemas.py         # Pydantic models
├── config.py          # Settings (pydantic-settings)
├── .env               # Local environment variables (not committed)
├── .env.example       # Template for .env
├── Dockerfile
├── docker-compose.yml
└── run.sh
```

## Requirements

- Docker
- NVIDIA driver
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Installing NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

```env
WHISPER_MODEL=large-v3-turbo # faster-whisper name: small | medium | large-v3 | large-v3-turbo | ...
WHISPER_COMPUTE_TYPE=int8    # int8 | int8_float16 | float16 | float32
DEFAULT_LANGUAGE=en          # default language code; use "auto" (or leave empty) to auto-detect
TRANSCRIBE_BEAM_SIZE=5       # higher = more accurate but slower
TRANSCRIBE_BEST_OF=5         # higher = more accurate but slower
TRANSCRIBE_VAD=true          # skip non-speech with Silero VAD (fewer hallucinations on silence)
TRANSCRIBE_CONDITION_ON_PREVIOUS_TEXT=false

# Optional decoding hint biasing the model's vocabulary (e.g. product names).
# Applied to every request unless the caller passes its own initial_prompt.
DEFAULT_INITIAL_PROMPT=

# CUDA watchdog — probes the GPU periodically and self-restarts on repeated failure.
CUDA_WATCHDOG_INTERVAL=30    # seconds between GPU health probes
CUDA_WATCHDOG_FAILURES=3     # consecutive failures before self-restart

# Load the model at startup (warms the cache and surfaces GPU problems early).
WARMUP_ON_START=true
```

## Model choice (GTX 1660 Ti, Korean talks)

Measured on 4-minute Korean seminar clips, beam 5, VAD on:

| Model (int8) | Speed | VRAM | GPU ≥95% / peak temp |
|---|---|---|---|
| large-v3-turbo | ~12x real time | 1.3 GB | 7 s of 40 s / 80°C |
| large-v3 | ~5x real time | 2.5 GB | 30 s of 90 s / 82°C |

Quality is close; large-v3 is marginally cleaner. Both are far better than openai-whisper `small`.
Models are cached in the `hf-cache` volume, so switching `WHISPER_MODEL` only needs `docker compose up -d --force-recreate`.

Requests are serialized (one transcription at a time) — concurrent jobs only thrash a 6 GB GPU.

## Running

```bash
bash run.sh
```

This builds the image and starts the container in the background. Re-running the same command after code changes will trigger a rebuild automatically.

## API

Swagger UI is available at `http://localhost:47321/docs` once the server is up.

### `POST /transcribe`

Transcribes an audio file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | file | Audio file (mp3, wav, m4a, etc.) |
| `language` | string | Language code (default: `en`). Use `auto` or an empty value to auto-detect the language |
| `initial_prompt` | string | Optional decoding hint biasing the vocabulary (falls back to `DEFAULT_INITIAL_PROMPT`) |

**Response**

```json
{
  "text": "Hello, world!",
  "language": "en",
  "duration": 3.5,
  "segments": [
    { "start": 0.0, "end": 3.5, "text": "Hello, world!" }
  ]
}
```

### `GET /health`

Actively probes the GPU with a tiny CUDA round-trip (NVML/`nvidia-smi` can report errors while CUDA still works). Returns `200` when CUDA is available, `503` when the device is lost.

```json
// 200 OK
{ "status": "ok", "cuda": true }

// 503 Service Unavailable
{ "status": "degraded", "cuda": false }
```

## Useful commands

```bash
docker compose logs -f    # tail logs
docker compose down       # stop the server
docker compose restart    # restart the server
```

## License

Released under the [MIT License](LICENSE).
