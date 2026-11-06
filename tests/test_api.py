import asyncio

import pytest
from fastapi.testclient import TestClient

import diarizer
import main
import service
import transcriber

client = TestClient(main.app)  # no `with`: lifespan (warmup, watchdog) never runs

LEGACY_SEGMENT_KEYS = {"start", "end", "text"}
LEGACY_RESPONSE_KEYS = {"text", "language", "duration", "segments"}


def fake_transcription(word_timestamps):
    seg = {"start": 0.0, "end": 1.0, "text": " 안녕하세요"}
    if word_timestamps:
        seg["words"] = [{"start": 0.0, "end": 1.0, "word": " 안녕하세요"}]
    return {"text": seg["text"], "language": "ko", "duration": 1.0, "segments": [seg]}


@pytest.fixture
def fake_transcribe(monkeypatch):
    calls = []

    async def transcribe_audio(tmp_path, language, prompt, word_timestamps=False):
        calls.append(word_timestamps)
        return fake_transcription(word_timestamps)

    monkeypatch.setattr(service, "transcribe_audio", transcribe_audio)
    return calls


def post_audio(path, **form):
    return client.post(path, files={"audio": ("a.wav", b"RIFF", "audio/wav")}, data=form)


# --- /transcribe ---------------------------------------------------------------

def test_transcribe_default_response_shape_is_unchanged(fake_transcribe):
    r = post_audio("/transcribe", language="ko")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == LEGACY_RESPONSE_KEYS
    assert [set(s) for s in body["segments"]] == [LEGACY_SEGMENT_KEYS]  # no "words": null
    assert fake_transcribe == [False]


def test_transcribe_word_timestamps(fake_transcribe):
    r = post_audio("/transcribe", language="ko", word_timestamps="true")
    assert r.status_code == 200
    assert r.json()["segments"][0]["words"] == [{"start": 0.0, "end": 1.0, "word": " 안녕하세요"}]
    assert fake_transcribe == [True]


def test_transcribe_out_of_memory_is_503(monkeypatch):
    async def transcribe_audio(*args, **kwargs):
        raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch.setattr(service, "transcribe_audio", transcribe_audio)
    r = post_audio("/transcribe")
    assert r.status_code == 503
    assert "out of memory" in r.json()["detail"]


# --- /diarize ------------------------------------------------------------------

def test_diarize_without_token_is_503_and_names_the_variable(monkeypatch):
    monkeypatch.setattr(diarizer.settings, "hf_token", "")
    monkeypatch.setattr(diarizer, "_pipeline", None)
    r = post_audio("/diarize")
    assert r.status_code == 503
    assert "HF_TOKEN" in r.json()["detail"]


@pytest.mark.parametrize("form", [
    {"num_speakers": "3", "min_speakers": "2"},
    {"min_speakers": "5", "max_speakers": "3"},
    {"num_speakers": "0"},
])
def test_diarize_rejects_contradictory_speaker_counts(form):
    r = post_audio("/diarize", **form)
    assert r.status_code == 400


def test_diarize_success(monkeypatch):
    seen = {}

    async def diarize_audio(tmp_path, num, lo, hi):
        seen.update(num=num, lo=lo, hi=hi)
        return {"duration": 12.5, "segments": [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01"},
            {"start": 8.0, "end": 12.0, "speaker": "SPEAKER_00"},
        ]}

    monkeypatch.setattr(service, "diarize_audio", diarize_audio)
    r = post_audio("/diarize", min_speakers="3", max_speakers="6")
    assert r.status_code == 200
    body = r.json()
    assert body["num_speakers"] == 2 and body["duration"] == 12.5
    assert body["segments"][1] == {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01"}
    assert seen == {"num": None, "lo": 3, "hi": 6}


def test_diarize_cuda_error_is_503(monkeypatch):
    async def diarize_audio(*args):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(service, "diarize_audio", diarize_audio)
    r = post_audio("/diarize")
    assert r.status_code == 503
    assert "recovering" in r.json()["detail"]


# --- shared GPU runner -----------------------------------------------------------

def test_out_of_memory_keeps_gpu_marked_healthy(monkeypatch):
    probes = []
    monkeypatch.setattr(transcriber, "cuda_healthy", lambda: probes.append(1) or False)
    transcriber.set_gpu_ok(True)

    def job():
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(RuntimeError):
        asyncio.run(transcriber.run_gpu_job(job))
    assert transcriber.gpu_ok() and probes == []


def test_cuda_error_probes_and_marks_gpu(monkeypatch):
    monkeypatch.setattr(transcriber, "cuda_healthy", lambda: False)
    transcriber.set_gpu_ok(True)

    def job():
        raise RuntimeError("CUDA error: unknown error")

    with pytest.raises(RuntimeError):
        asyncio.run(transcriber.run_gpu_job(job))
    assert not transcriber.gpu_ok()
    transcriber.set_gpu_ok(True)


def test_gpu_busy_while_job_runs():
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        loop = asyncio.get_running_loop()

        def job():
            loop.call_soon_threadsafe(started.set)
            asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
            return "done"

        task = asyncio.create_task(transcriber.run_gpu_job(job))
        await started.wait()
        busy_during = transcriber.gpu_busy()
        release.set()
        assert await task == "done"
        return busy_during, transcriber.gpu_busy()

    assert asyncio.run(scenario()) == (True, False)


def test_clear_cuda_cache_does_not_import_torch(monkeypatch):
    monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)
    transcriber.clear_cuda_cache()
    assert "torch" not in __import__("sys").modules
