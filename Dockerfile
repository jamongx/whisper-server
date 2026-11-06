FROM python:3.12-slim

WORKDIR /app

# ffmpeg decodes uploads to 16 kHz PCM (see transcriber.load_audio)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
# Drop the poetry/pip download cache in the same layer — with torch it is ~8 GB.
# (/root/.cache/huggingface is the hf-cache volume, mounted at runtime.)
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --without dev && \
    rm -rf /root/.cache/pypoetry /root/.cache/pip

# CTranslate2 loads cuBLAS/cuDNN dynamically; point it at the pip-installed NVIDIA libs.
# torch (diarization) pins the same CUDA 12 / cuDNN 9 wheels, so both engines
# load one copy of each library.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

# pyannote.audio sends usage telemetry to otel.pyannote.ai by default; meeting
# metadata stays on this machine.
ENV PYANNOTE_METRICS_ENABLED=false

COPY . .

EXPOSE 47321

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "47321"]
