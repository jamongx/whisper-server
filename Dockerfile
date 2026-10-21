FROM python:3.12-slim

WORKDIR /app

# ffmpeg decodes uploads to 16 kHz PCM (see transcriber.load_audio)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

# CTranslate2 loads cuBLAS/cuDNN dynamically; point it at the pip-installed NVIDIA libs.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

COPY . .

EXPOSE 47321

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "47321"]
