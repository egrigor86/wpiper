"""Settings and data containers for the audiobook builder."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BookConfig:
    # ---- voice (single narrator now; `speakers` is the scaffold for several voices later)
    voice: str = ""                                   # path to <name>.onnx
    speakers: Dict[str, str] = field(default_factory=dict)   # speaker tag -> voice path (unused for now)
    noise_scale: Optional[float] = None               # None = use the voice's own defaults
    length_scale: Optional[float] = None              # >1 slower
    noise_w: Optional[float] = None
    threads: int = 0                                  # onnxruntime intra-op threads, 0 = all cores
    providers: Tuple[str, ...] = ("CPUExecutionProvider",)

    # ---- text -> units
    max_words: int = 60                               # a sentence longer than this is split at ; : , or a conjunction
    min_words: int = 4                                # smallest piece produced by a split
    wrap: str = "auto"                                # auto | hard | lines  (how single newlines are read)
    speak_headings: bool = True
    pronunciations: str = ""                          # path to a JSON {word: replacement}
    caps_headings: str = "auto"                       # ALL-CAPS title lines: auto | chapters | inline | off
    number_headings: bool = True                      # lines holding only 1, 2, 3 ... or I, II, III ... are chapters
    number_label: str = "Chapter"                     # spoken before such a number ("" = just the number)
    drop_page_numbers: bool = True                    # other digit-only lines are page numbers: not spoken

    # ---- silences (ms)
    pause_sentence: int = 350
    pause_fragment: int = 150                         # between pieces of one long sentence
    pause_paragraph: int = 800
    pause_heading: int = 900
    pause_scene: int = 1200
    pause_chapter: int = 2000
    pause_scale: float = 1.0

    # ---- audio
    trim_db: float = -45.0                            # silence threshold relative to each clip's peak
    trim_head_ms: int = 20
    trim_tail_ms: int = 60
    peak_target: float = 0.9                          # master peak level (fraction of full scale)

    # ---- subtitles
    srt_max_chars: int = 42
    srt_max_lines: int = 2
    srt_max_sec: float = 7.0
    srt_min_sec: float = 1.0
    srt_linger_ms: int = 150
    srt_offset_ms: int = 0

    # ---- output
    title: str = ""
    author: str = ""
    formats: Tuple[str, ...] = ("wav",)               # wav flac mp3 opus m4b
    split_chapters: bool = False
    loudnorm: bool = False
    bitrate: str = ""                                 # e.g. 64k (mp3/opus/m4b); empty = sensible default
    ffmpeg: str = ""                                  # path; empty = $FFMPEG_EXE or PATH
    limit: int = 0                                    # only the first N units (smoke tests)


@dataclass
class Chapter:
    index: int
    title: str
    text: str
    has_heading: bool = False                         # title was found in / given to the text and should be spoken


@dataclass
class Unit:
    """One synthesis atom: a sentence (or a piece of an over-long sentence)."""
    idx: int
    chapter: int
    para: int
    sent: int                                         # pieces of the same sentence share this id
    kind: str                                         # sentence | heading | scene_break
    text: str                                         # display text (subtitles)
    spoken: str                                       # cleaned text sent to the TTS ('' for scene breaks)
    speaker: str = "narrator"
    ends_sentence: bool = True
    ends_para: bool = False
    ends_chapter: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Placed:
    unit: Unit
    start: int                                        # sample offsets in the master timeline
    end: int                                          # end of the speech (before the pause that follows)


UnitList = List[Unit]
