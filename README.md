# whisper-server

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![faster-whisper](https://img.shields.io/badge/faster--whisper-CTranslate2-412991.svg)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900.svg?logo=nvidia&logoColor=white)

A lightweight local STT server running [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, int8) behind FastAPI, built for a laptop with a GTX 1660 Ti (6GB VRAM) on Ubuntu. Optional speaker diarization runs [pyannote.audio](https://github.com/pyannote/pyannote-audio) on the same GPU — everything stays on the machine.

> 📖 Write-up: [Generate Speaker-Labeled Subtitles with Whisper and pyannote.audio on Ubuntu 24.04](https://jamongx.com/whisper-pyannote-speaker-labeled-subtitles-ubuntu-24-04/) — how I use this server in practice.

## Specs

| | |
|---|---|
| GPU | NVIDIA GTX 1660 Ti 6GB |
| OS | Ubuntu 24.04 |
| Runtime | Docker |
| Default model | large-v3-turbo (int8) |
| Diarization | pyannote/speaker-diarization-community-1 (optional) |

## Project structure

```
whisper_server/
├── main.py            # FastAPI app and routes (thin web adapter)
├── service.py         # Transcription / diarization use-cases (upload → response)
├── transcriber.py     # Whisper loading, inference, shared GPU lock and helpers
├── diarizer.py        # pyannote diarization pipeline (lazy-loaded)
├── watchdog.py        # GPU health watchdog (self-restart on CUDA loss)
├── schemas.py         # Pydantic models
├── config.py          # Settings (pydantic-settings)
├── scripts/
│   └── merge_diarization.py  # Reference client merge: words + speakers → SRT
├── tests/             # pytest (merge logic, API shape)
├── .env               # Local environment variables (not committed)
├── .env.example       # Template for .env
├── Dockerfile
├── docker-compose.yml
└── run.sh
```

## Requirements

- Docker
- NVIDIA driver — see [How to Install NVIDIA Driver on Ubuntu 24.04 (GTX 1660 Ti Laptop)](https://jamongx.com/ubuntu-24-04-nvidia-driver-install-gtx-1660-ti/) for the exact setup this server was built on
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
CUDA_WATCHDOG_INTERVAL=600   # seconds between idle GPU probes (each probe wakes the GPU ~20 s)
CUDA_WATCHDOG_RETRY=15       # re-probe interval once the GPU looks unhealthy
CUDA_WATCHDOG_FAILURES=3     # consecutive failures before self-restart

# Speaker diarization (POST /diarize). The model is gated: accept its terms on
# huggingface.co with the account that owns the token. Leave HF_TOKEN empty to
# run without diarization (/diarize then returns 503). Never commit the value.
HF_TOKEN=                    # or keep it in ~/.config/secrets/whisper_server.env (see below)
DIARIZE_MODEL=pyannote/speaker-diarization-community-1
DIARIZE_SEGMENTATION_BATCH_SIZE=0  # 0 = pipeline default; larger = faster, more VRAM
DIARIZE_EMBEDDING_BATCH_SIZE=0

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

Requests are serialized (one GPU job at a time, transcription and diarization share the lock) — concurrent jobs only thrash a 6 GB GPU.

When idle the GPU drops to P8 (~8 W) with the model still loaded in VRAM, so the first request after a pause is not slowed down.

## Running

```bash
bash run.sh
```

This builds the image and starts the container in the background. Re-running the same command after code changes will trigger a rebuild automatically.

The first start downloads the model from Hugging Face (internet required; ~1.6 GB for large-v3-turbo, ~3 GB for large-v3), so `/health` may take a few minutes to come up. Later starts load it from the `hf-cache` volume.

## API

Swagger UI is available at `http://localhost:47321/docs` once the server is up.

### `POST /transcribe`

Transcribes an audio file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | file | Audio file (mp3, wav, m4a, etc.) |
| `language` | string | Language code (default: `en`). Use `auto` or an empty value to auto-detect the language |
| `initial_prompt` | string | Optional decoding hint biasing the vocabulary (falls back to `DEFAULT_INITIAL_PROMPT`) |
| `word_timestamps` | bool | Default `false`. When `true`, each segment also carries `words: [{start, end, word}]` (slower; used to merge with `/diarize`) |

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

With `word_timestamps=true` a segment looks like this (the default response has no `words` key at all). `word` keeps Whisper's leading space, so concatenating words rebuilds the text:

```json
{ "start": 0.0, "end": 1.2, "text": "안녕하세요 여러분",
  "words": [ { "start": 0.0, "end": 0.6, "word": " 안녕하세요" },
             { "start": 0.6, "end": 1.2, "word": " 여러분" } ] }
```

### `POST /diarize`

Finds who spoke when. Send the **whole recording in one request** — not the 60-minute chunks used for `/transcribe` — so speaker labels stay consistent across the file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | file | Audio file (mp3, wav, m4a, etc.) |
| `num_speakers` | int | Optional exact speaker count |
| `min_speakers` / `max_speakers` | int | Optional bounds (cannot be combined with `num_speakers`) |

**Response** — non-overlapping turns (pyannote's *exclusive* diarization: one speaker at any instant):

```json
{
  "duration": 5520.4,
  "num_speakers": 4,
  "segments": [
    { "start": 0.53, "end": 7.91, "speaker": "SPEAKER_00" },
    { "start": 7.91, "end": 12.40, "speaker": "SPEAKER_02" }
  ]
}
```

| Status | When |
|---|---|
| `400` | Contradictory speaker counts |
| `503` | `HF_TOKEN` missing, model terms not accepted, or download failed (the `detail` says which); GPU out of memory; CUDA lost |

## Speaker diarization

**Setup.** Accept the terms of [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) on Hugging Face, create a read token, and store it outside the repo:

```bash
mkdir -p -m 700 ~/.config/secrets
(umask 077; echo "HF_TOKEN=hf_..." > ~/.config/secrets/whisper_server.env)
```

`docker-compose.yml` loads that file after `.env` when it exists (`HF_TOKEN` in `.env` works too). Without it the server starts normally and only `/diarize` answers 503. The pipeline (a few tens of MB) is downloaded on the first `/diarize` call into the `hf-cache` volume and loaded on first use — never at startup, so days without diarization keep the GPU idle exactly as before.

**Why community-1.** It is pyannote's current open pipeline (pyannote.audio 4, CC-BY-4.0) and runs fully locally; it improves speaker counting and assignment over 3.1 and provides exclusive diarization for aligning with transcripts. `speaker-diarization-precision` is newer but runs on pyannoteAI's cloud, so it is not an option for confidential recordings.

**Privacy.** pyannote.audio 4 sends usage telemetry (audio duration, speaker counts) to `otel.pyannote.ai` by default. The image sets `PYANNOTE_METRICS_ENABLED=false`, and `diarizer.py` sets it again before importing pyannote, so nothing leaves the machine. Audio is never uploaded. After the first download you may also set `HF_HUB_OFFLINE=1` to stop all Hub requests (only once both the Whisper and pyannote models are in the cache).

**Client workflow.**

```bash
# 1. Speakers for the whole recording (one request)
curl -F audio=@meeting.m4a -F min_speakers=3 -F max_speakers=6 \
     http://SERVER:47321/diarize > diarization.json
# 2. Words for each 60-minute chunk
curl -F audio=@part1.m4a -F language=ko -F word_timestamps=true \
     http://SERVER:47321/transcribe > part1.json
# 3. Merge on the client: PATH@SECONDS shifts each chunk to its start time
python scripts/merge_diarization.py diarization.json part1.json@0 part2.json@3600 -o meeting.srt
```

The merge assigns each word to the speaker turn it overlaps longest, starts a new cue whenever the speaker changes, and writes cues as `[SPEAKER_00] text`. The server never merges; the script is a reference implementation (standard library only).

**Long requests.** Diarizing a 1–2 hour recording takes minutes and holds the GPU lock, so a `/transcribe` sent meanwhile waits for it. Give clients a generous read timeout (e.g. `curl --max-time 3600`). The upload is read into memory, and very long files (3–4 h) need several GB of RAM for clustering — measure before relying on them.

### `GET /health`

Returns the last known GPU state without touching the GPU — any CUDA call wakes the card out of its low-power state (P8, ~8 W) for 15–20 s, and Docker polls this endpoint every 30 s. The state is kept current by real transcriptions/diarizations and by the watchdog, which probes with a tiny CUDA round-trip (NVML/`nvidia-smi` can report errors while CUDA still works) only every `CUDA_WATCHDOG_INTERVAL` seconds (default 10 min), only when no GPU job succeeded in that window, and never while one is running. A CUDA error during a request triggers an immediate re-probe; after `CUDA_WATCHDOG_FAILURES` failed probes the process exits and Docker restarts it. Returns `200` when CUDA is available, `503` when the device is lost.

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

# tests (no GPU needed)
pip install fastapi==0.111.1 pydantic-settings python-multipart numpy pytest httpx
python -m pytest
```

## Related reading

Field notes from building and running this, on [jamongx.com](https://jamongx.com):

- [Generate Speaker-Labeled Subtitles with Whisper and pyannote.audio on Ubuntu 24.04](https://jamongx.com/whisper-pyannote-speaker-labeled-subtitles-ubuntu-24-04/)
- [How to Install NVIDIA Driver on Ubuntu 24.04 (GTX 1660 Ti Laptop)](https://jamongx.com/ubuntu-24-04-nvidia-driver-install-gtx-1660-ti/)

## License

Released under the [MIT License](LICENSE).
