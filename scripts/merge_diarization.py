#!/usr/bin/env python3
"""Merge /transcribe word timestamps with /diarize speaker turns into an SRT.

Reference implementation for clients — the server never merges. Standard
library only, so it runs anywhere.

    # 1. Diarize the whole recording once (speaker labels must not reset per chunk)
    curl -F audio=@meeting.m4a -F min_speakers=3 -F max_speakers=6 \\
         http://SERVER:47321/diarize > diarization.json
    # 2. Transcribe each 60-minute chunk with word timestamps
    curl -F audio=@part1.m4a -F language=ko -F word_timestamps=true \\
         http://SERVER:47321/transcribe > part1.json
    # 3. Merge; PATH@SECONDS shifts a chunk to its start time in the recording
    python scripts/merge_diarization.py diarization.json part1.json@0 part2.json@3600 -o meeting.srt

Each word goes to the speaker turn it overlaps longest (nearest turn if it
falls in a gap). A cue ends where the speaker changes, at a Whisper segment
boundary, after a long pause, or when it grows too long. Cues read
"[SPEAKER_00] text".
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Word:
    start: float
    end: float
    text: str  # keeps Whisper's leading space, so cue text is plain concatenation
    segment: int  # index of the Whisper segment it came from
    speaker: str | None = None


@dataclass
class Cue:
    start: float
    end: float
    speaker: str
    text: str


class SpeakerIndex:
    """Speaker lookup over turns sorted by start. Turns from exclusive
    diarization don't overlap, but the lookup stays correct if they do."""

    def __init__(self, turns: list[Turn]):
        self.turns = sorted(turns, key=lambda t: (t.start, t.end))
        self.starts = [t.start for t in self.turns]
        # max_end[i] = latest end among turns[0..i] — bounds the backward scan;
        # max_end_at[i] = index of that turn.
        self.max_end: list[float] = []
        self.max_end_at: list[int] = []
        for i, t in enumerate(self.turns):
            if not self.max_end or t.end > self.max_end[-1]:
                self.max_end.append(t.end)
                self.max_end_at.append(i)
            else:
                self.max_end.append(self.max_end[-1])
                self.max_end_at.append(self.max_end_at[-1])

    def speaker_at(self, start: float, end: float) -> str | None:
        """Speaker overlapping [start, end] the longest; ties go to the earlier
        turn. With no overlap, the turn closest in time. None if no turns."""
        if not self.turns:
            return None
        best, best_overlap = None, 0.0
        i = bisect.bisect_left(self.starts, end) - 1  # turns starting before `end`
        while i >= 0 and self.max_end[i] > start:
            t = self.turns[i]
            overlap = min(end, t.end) - max(start, t.start)
            if overlap > 0 and overlap >= best_overlap:  # >= : earlier turn wins ties
                best, best_overlap = t, overlap
            i -= 1
        if best is not None:
            return best.speaker
        return self._nearest(start, end).speaker

    def _nearest(self, start: float, end: float) -> Turn:
        def gap(t: Turn) -> float:
            return max(t.start - end, start - t.end, 0.0)

        i = bisect.bisect_left(self.starts, end)
        candidates = []
        if i > 0:  # of the turns before the word, the one ending latest
            candidates.append(self.turns[self.max_end_at[i - 1]])
        if i < len(self.turns):  # the first turn after it
            candidates.append(self.turns[i])
        return min(candidates, key=lambda t: (gap(t), t.start))


def load_turns(diarization: dict) -> list[Turn]:
    return [Turn(float(s["start"]), float(s["end"]), str(s["speaker"]))
            for s in diarization.get("segments", [])]


def load_words(transcript: dict, offset: float = 0.0, segment_base: int = 0) -> list[Word]:
    """Flatten a /transcribe response into words shifted by `offset` seconds.
    A segment without word timestamps becomes one pseudo-word, so older
    responses still merge (at segment granularity)."""
    words: list[Word] = []
    for n, seg in enumerate(transcript.get("segments", [])):
        seg_id = segment_base + n
        seg_words = seg.get("words")
        if seg_words:
            for w in seg_words:
                if w.get("start") is None or w.get("end") is None or not w.get("word", "").strip():
                    continue
                words.append(Word(w["start"] + offset, w["end"] + offset, w["word"], seg_id))
        elif seg.get("text", "").strip():
            words.append(Word(seg["start"] + offset, seg["end"] + offset, " " + seg["text"].strip(), seg_id))
    return words


def assign_speakers(words: list[Word], turns: list[Turn], unknown: str = "UNKNOWN") -> None:
    index = SpeakerIndex(turns)
    for w in words:
        w.speaker = index.speaker_at(w.start, w.end) or unknown


def build_cues(
    words: list[Word],
    max_chars: int = 80,
    max_duration: float = 7.0,
    max_gap: float = 1.5,
) -> list[Cue]:
    cues: list[Cue] = []
    current: list[Word] = []

    def flush() -> None:
        if current:
            text = "".join(w.text for w in current).strip()
            cues.append(Cue(current[0].start, current[-1].end, current[0].speaker, text))
            current.clear()

    for w in words:
        if current:
            prev = current[-1]
            text_len = len("".join(x.text for x in current).strip()) + len(w.text)
            if (
                w.speaker != prev.speaker
                or w.segment != prev.segment
                or w.start - prev.end > max_gap
                or w.end - current[0].start > max_duration
                or text_len > max_chars
            ):
                flush()
        current.append(w)
    flush()
    return cues


def _timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{n}\n{_timestamp(c.start)} --> {_timestamp(c.end)}\n[{c.speaker}] {c.text}\n"
        for n, c in enumerate(cues, 1)
    ]
    return "\n".join(blocks)


def merge(diarization: dict, transcripts: list[tuple[dict, float]], **cue_options) -> str:
    """Full pipeline: transcripts are (response, offset_seconds) pairs."""
    words: list[Word] = []
    segment_base = 0
    for transcript, offset in transcripts:
        words.extend(load_words(transcript, offset, segment_base))
        segment_base += len(transcript.get("segments", []))
    words.sort(key=lambda w: (w.start, w.end))
    assign_speakers(words, load_turns(diarization))
    return to_srt(build_cues(words, **cue_options))


def _parse_chunk(arg: str) -> tuple[Path, float]:
    path, sep, offset = arg.rpartition("@")
    if not sep:
        return Path(arg), 0.0
    try:
        return Path(path), float(offset)
    except ValueError:
        return Path(arg), 0.0  # an "@" that is part of the file name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diarization", type=Path, help="JSON response of POST /diarize")
    parser.add_argument("transcripts", nargs="+", help="JSON response(s) of POST /transcribe, as PATH[@OFFSET_SECONDS]")
    parser.add_argument("-o", "--output", type=Path, help="SRT path (default: stdout)")
    parser.add_argument("--max-chars", type=int, default=80)
    parser.add_argument("--max-duration", type=float, default=7.0)
    parser.add_argument("--max-gap", type=float, default=1.5)
    args = parser.parse_args(argv)

    diarization = json.loads(args.diarization.read_text(encoding="utf-8"))
    transcripts = []
    for chunk in args.transcripts:
        path, offset = _parse_chunk(chunk)
        transcripts.append((json.loads(path.read_text(encoding="utf-8")), offset))
    srt = merge(diarization, transcripts, max_chars=args.max_chars,
                max_duration=args.max_duration, max_gap=args.max_gap)
    if args.output:
        args.output.write_text(srt, encoding="utf-8")
    else:
        sys.stdout.write(srt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
