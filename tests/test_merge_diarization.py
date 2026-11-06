import merge_diarization as m


def turns(*spec):
    return [m.Turn(s, e, spk) for s, e, spk in spec]


def word(start, end, text, segment=0):
    return m.Word(start, end, text, segment)


# --- speaker assignment: overlap -------------------------------------------

def test_word_goes_to_speaker_with_longest_overlap():
    index = m.SpeakerIndex(turns((0.0, 2.3, "A"), (2.3, 5.0, "B")))
    assert index.speaker_at(2.0, 2.5) == "A"  # 0.3 s with A, 0.2 s with B
    assert index.speaker_at(2.2, 2.8) == "B"  # 0.1 s with A, 0.5 s with B


def test_word_inside_a_single_turn():
    index = m.SpeakerIndex(turns((0.0, 1.0, "A"), (1.0, 9.0, "B"), (9.0, 10.0, "C")))
    assert index.speaker_at(4.0, 4.4) == "B"


def test_overlapping_turns_still_pick_longest_overlap():
    # Non-exclusive input: a long A turn spans a short B turn.
    index = m.SpeakerIndex(turns((0.0, 10.0, "A"), (3.0, 3.5, "B")))
    assert index.speaker_at(3.0, 4.0) == "A"  # 1.0 s with A, 0.5 s with B
    assert index.speaker_at(3.1, 3.4) in ("A", "B")


# --- speaker assignment: boundaries ------------------------------------------

def test_equal_overlap_goes_to_earlier_turn():
    index = m.SpeakerIndex(turns((0.0, 2.0, "A"), (2.0, 4.0, "B")))
    assert index.speaker_at(1.5, 2.5) == "A"


def test_word_touching_a_boundary_is_not_an_overlap():
    index = m.SpeakerIndex(turns((0.0, 2.0, "A"), (2.0, 4.0, "B")))
    assert index.speaker_at(2.0, 2.4) == "B"  # touches A's end only
    assert index.speaker_at(1.6, 2.0) == "A"  # touches B's start only


def test_word_in_gap_goes_to_nearest_turn():
    index = m.SpeakerIndex(turns((0.0, 2.0, "A"), (5.0, 7.0, "B")))
    assert index.speaker_at(2.2, 2.5) == "A"
    assert index.speaker_at(4.5, 4.8) == "B"


def test_word_before_first_or_after_last_turn():
    index = m.SpeakerIndex(turns((3.0, 4.0, "A"), (5.0, 6.0, "B")))
    assert index.speaker_at(0.0, 0.5) == "A"
    assert index.speaker_at(9.0, 9.5) == "B"


def test_nearest_considers_long_earlier_turn():
    # B starts after A but ends earlier; a word after both is closest to A.
    index = m.SpeakerIndex(turns((0.0, 10.0, "A"), (1.0, 2.0, "B"), (20.0, 21.0, "C")))
    assert index.speaker_at(11.0, 11.5) == "A"


def test_no_turns_gives_unknown():
    words = [word(0.0, 0.5, " 안녕")]
    m.assign_speakers(words, [])
    assert words[0].speaker == "UNKNOWN"


# --- cue building: speaker changes ---------------------------------------------

def test_cue_splits_where_speaker_changes():
    words = [word(0.0, 0.4, " 안녕하세요"), word(0.5, 0.9, " 반갑습니다"),
             word(1.0, 1.4, " 네"), word(1.5, 1.9, " 좋아요")]
    m.assign_speakers(words, turns((0.0, 0.95, "SPEAKER_00"), (0.95, 2.0, "SPEAKER_01")))
    cues = m.build_cues(words)
    assert [(c.speaker, c.text) for c in cues] == [
        ("SPEAKER_00", "안녕하세요 반갑습니다"),
        ("SPEAKER_01", "네 좋아요"),
    ]
    assert (cues[0].start, cues[0].end) == (0.0, 0.9)
    assert (cues[1].start, cues[1].end) == (1.0, 1.9)


def test_speaker_change_inside_a_whisper_segment():
    words = [word(0.0, 1.0, " 그러니까", 0), word(1.0, 2.0, " 아니요", 0)]
    m.assign_speakers(words, turns((0.0, 1.0, "A"), (1.0, 2.0, "B")))
    assert [c.speaker for c in m.build_cues(words)] == ["A", "B"]


def test_same_speaker_back_to_back_turns_stay_one_cue():
    words = [word(0.0, 0.4, " 하나"), word(0.5, 0.9, " 둘")]
    m.assign_speakers(words, turns((0.0, 0.45, "A"), (0.45, 1.0, "A")))
    assert len(m.build_cues(words)) == 1


def test_cue_splits_on_segment_gap_and_length():
    words = [word(0.0, 0.5, " 가", 0), word(0.6, 1.0, " 나", 1),  # new segment
             word(3.0, 3.5, " 다", 1),                              # 2 s pause
             word(3.6, 11.0, " 라", 1)]                             # > 7 s cue
    for w in words:
        w.speaker = "A"
    assert [c.text for c in m.build_cues(words)] == ["가", "나", "다", "라"]


def test_cue_splits_when_too_long():
    words = [word(i * 0.1, i * 0.1 + 0.05, " 가나다라마") for i in range(20)]
    for w in words:
        w.speaker = "A"
    cues = m.build_cues(words, max_chars=20)
    assert all(len(c.text) <= 20 for c in cues)
    assert "".join(c.text.replace(" ", "") for c in cues) == "가나다라마" * 20


# --- end to end ----------------------------------------------------------------

def test_merge_applies_chunk_offsets_and_formats_srt():
    diarization = {"segments": [
        {"start": 0.0, "end": 3601.0, "speaker": "SPEAKER_00"},
        {"start": 3601.0, "end": 3700.0, "speaker": "SPEAKER_01"},
    ]}
    part1 = {"segments": [{"start": 1.0, "end": 2.0, "text": "첫 번째",
                           "words": [{"start": 1.0, "end": 1.4, "word": " 첫"},
                                     {"start": 1.5, "end": 2.0, "word": " 번째"}]}]}
    part2 = {"segments": [{"start": 1.5, "end": 3.0, "text": "두 번째",
                           "words": [{"start": 1.5, "end": 2.0, "word": " 두"},
                                     {"start": 2.1, "end": 3.0, "word": " 번째"}]}]}
    srt = m.merge(diarization, [(part2, 3600.0), (part1, 0.0)])
    assert srt == (
        "1\n00:00:01,000 --> 00:00:02,000\n[SPEAKER_00] 첫 번째\n"
        "\n"
        "2\n01:00:01,500 --> 01:00:03,000\n[SPEAKER_01] 두 번째\n"
    )


def test_segments_without_words_fall_back_to_segment_level():
    diarization = {"segments": [{"start": 0.0, "end": 5.0, "speaker": "A"},
                                {"start": 5.0, "end": 9.0, "speaker": "B"}]}
    transcript = {"segments": [{"start": 0.5, "end": 4.0, "text": " 하나 "},
                               {"start": 5.5, "end": 8.0, "text": "둘"}]}
    srt = m.merge(diarization, [(transcript, 0.0)])
    assert "[A] 하나" in srt and "[B] 둘" in srt


def test_words_without_timestamps_are_skipped():
    transcript = {"segments": [{"start": 0, "end": 1, "text": "x", "words": [
        {"start": None, "end": None, "word": " x"}, {"start": 0.1, "end": 0.2, "word": " y"}]}]}
    assert [w.text for w in m.load_words(transcript)] == [" y"]


def test_parse_chunk():
    assert m._parse_chunk("part2.json@3600") == (m.Path("part2.json"), 3600.0)
    assert m._parse_chunk("part1.json") == (m.Path("part1.json"), 0.0)
    assert m._parse_chunk("me@home.json") == (m.Path("me@home.json"), 0.0)


def test_timestamp_format():
    assert m._timestamp(3725.4567) == "01:02:05,457"
    assert m._timestamp(-0.1) == "00:00:00,000"
