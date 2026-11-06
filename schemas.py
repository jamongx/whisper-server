from pydantic import BaseModel


class Word(BaseModel):
    start: float
    end: float
    word: str


class Segment(BaseModel):
    start: float
    end: float
    text: str
    # Only filled when /transcribe is called with word_timestamps=true. The route
    # drops None fields, so default responses keep their original shape.
    words: list[Word] | None = None


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration: float | None = None
    segments: list[Segment] = []


class SpeakerTurn(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizeResponse(BaseModel):
    duration: float
    num_speakers: int
    segments: list[SpeakerTurn] = []
