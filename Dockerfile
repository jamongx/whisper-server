FROM python:3.12-slim

WORKDIR /app

# whisper가 요구하는 ffmpeg 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

COPY . .

EXPOSE 47321

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "47321"]
