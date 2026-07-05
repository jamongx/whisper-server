from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration: float | None = None
    segments: list[Segment] = []
