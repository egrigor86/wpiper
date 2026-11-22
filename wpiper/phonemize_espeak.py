"""Phonemization with espeak-ng.

Two backends, same output format:
  * "bridge": the prebuilt Piper espeakbridge.pyd (exact Piper phonemes).
  * "ctypes": the installed eSpeak NG DLL, used when the bridge is missing.
    It lacks espeak_TextToPhonemesWithTerminator, so clause terminators are
    recovered from the text espeak consumed. It differs from Piper in a few
    places (e.g. no 'ʲ' glide between vowels), so phoneme ids from the two
    backends should not be mixed in one dataset.
"""

import ctypes
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Union

AUDIO_OUTPUT_SYNCHRONOUS = 2
ESPEAK_CHARS_UTF8 = 1
ESPEAK_PHONEMES_IPA = 0x02
EE_OK = 0

ESPEAK_DATA_DIR = Path(__file__).parent / "espeak-ng-data"

_ESPEAK_DIRS = [Path(r"C:\Program Files\eSpeak NG"), Path(r"C:\Program Files (x86)\eSpeak NG")]
if "ESPEAK_NG_DIR" in os.environ:
    _ESPEAK_DIRS.insert(0, Path(os.environ["ESPEAK_NG_DIR"]))

_SENTENCE_END = ".?!"
_CLAUSE_END = ".?!,:;"


class EspeakPhonemizer:
    """Phonemizer that uses espeak-ng."""

    def __init__(
        self,
        espeak_data_dir: Union[str, Path] = ESPEAK_DATA_DIR,
        backend: Optional[str] = None,
        espeak_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """backend: "bridge", "ctypes", or None to prefer the bridge."""
        self._voice: Optional[str] = None
        self._bridge = None
        self._data_dir = Path(espeak_data_dir)

        if backend in (None, "bridge"):
            try:
                from . import espeakbridge

                espeakbridge.initialize(str(espeak_data_dir))
                self._bridge = espeakbridge
            except ImportError:
                if backend == "bridge":
                    raise

        if self._bridge is None:
            self._init_ctypes(espeak_dir, espeak_data_dir)

        self.backend = "bridge" if self._bridge is not None else "ctypes"

    def _init_ctypes(self, espeak_dir, espeak_data_dir) -> None:
        dirs = [Path(espeak_dir)] if espeak_dir else _ESPEAK_DIRS
        for d in dirs:
            if (d / "libespeak-ng.dll").exists():
                break
        else:
            raise FileNotFoundError(
                "libespeak-ng.dll not found; install eSpeak NG or set ESPEAK_NG_DIR"
            )

        os.add_dll_directory(str(d))
        self._lib = ctypes.CDLL(str(d / "libespeak-ng.dll"))
        self._lib.espeak_Initialize.restype = ctypes.c_int
        self._lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        self._lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        self._lib.espeak_TextToPhonemes.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
        ]

        rate = self._lib.espeak_Initialize(
            AUDIO_OUTPUT_SYNCHRONOUS, 0, str(espeak_data_dir).encode("utf-8"), 0
        )
        if rate < 0:
            raise RuntimeError("Failed to initialize espeak-ng")

    def _set_voice_raw(self, voice: str) -> None:
        if self._bridge is not None:
            self._bridge.set_voice(voice)
        elif self._lib.espeak_SetVoiceByName(voice.encode("utf-8")) != EE_OK:
            raise RuntimeError(f"Failed to set voice: {voice}")

    def _voice_file_for_language(self, code: str) -> Optional[str]:
        """Name of the voice file under lang/ that declares 'language <code>'.

        espeak looks voices up by file name, so codes such as "en-gb" (file
        lang/gmw/en) or "fr-fr" (file lang/roa/fr) fail by name alone. The
        file with the lowest priority number for the code wins.
        """
        code = code.lower()
        best = None
        for path in (self._data_dir / "lang").rglob("*"):
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line in lines:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "language" and parts[1].lower() == code:
                    prio = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 5
                    if best is None or prio < best[0]:
                        best = (prio, path.name)
        return best[1] if best else None

    def _set_voice(self, voice: str) -> None:
        if voice == self._voice:
            return
        try:
            self._set_voice_raw(voice)
        except RuntimeError:
            alias = self._voice_file_for_language(voice)
            if not alias or alias.lower() == voice.lower():
                raise
            self._set_voice_raw(alias)
        self._voice = voice

    def _clauses(self, text: str) -> list[tuple[str, str, bool]]:
        """Text to (phonemes, terminator, end_of_sentence) per clause."""
        if self._bridge is not None:
            return self._bridge.get_phonemes(text)

        data = text.encode("utf-8")
        buf = ctypes.create_string_buffer(data)
        base = ctypes.addressof(buf)
        ptr = ctypes.c_void_p(base)

        clauses: list[tuple[str, str, bool]] = []
        start = 0
        while ptr.value is not None:
            phonemes = self._lib.espeak_TextToPhonemes(
                ctypes.byref(ptr), ESPEAK_CHARS_UTF8, ESPEAK_PHONEMES_IPA
            )
            at_end = ptr.value is None
            end = len(data) if at_end else ptr.value - base
            consumed = data[start:end].decode("utf-8", errors="ignore")
            start = end

            # espeak stops one character past the terminator (after skipping
            # whitespace), so drop that look-ahead character before looking.
            if not at_end:
                consumed = consumed[:-1]
            consumed = consumed.rstrip()

            terminator = consumed[-1] if consumed and consumed[-1] in _CLAUSE_END else ""
            clauses.append(
                (
                    (phonemes or b"").decode("utf-8"),
                    terminator,
                    terminator in _SENTENCE_END,
                )
            )

        return clauses

    def phonemize(self, voice: str, text: str) -> list[list[str]]:
        """Text to phonemes grouped by sentence."""
        self._set_voice(voice)

        all_phonemes: list[list[str]] = []
        sentence_phonemes: list[str] = []

        for phonemes_str, terminator_str, end_of_sentence in self._clauses(text):
            # Filter out (lang) switch (flags).
            phonemes_str = re.sub(r"\([^)]+\)", "", phonemes_str)

            # Keep punctuation even though it's not technically a phoneme
            phonemes_str += terminator_str
            if terminator_str in (",", ":", ";"):
                # Not a sentence boundary
                phonemes_str += " "

            # Decompose phonemes into UTF-8 codepoints.
            sentence_phonemes.extend(list(unicodedata.normalize("NFD", phonemes_str)))

            if end_of_sentence:
                all_phonemes.append(sentence_phonemes)
                sentence_phonemes = []

        if sentence_phonemes:
            all_phonemes.append(sentence_phonemes)

        return all_phonemes
