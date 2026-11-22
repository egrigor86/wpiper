"""Large-dataset format: one memory-mapped int16 audio file plus small index files.

Layout of a prepared dataset directory (written by wpiper.prepare_big):
    audio.bin       all clips back to back, int16 mono at config sample rate
    phonemes.bin    all phoneme-id sequences back to back, int16
    index.npz       audio_off, audio_len, ph_off, ph_len (per clip)
    meta.tsv        idx <tab> source id <tab> text
    config.json     Piper config (same as the small-dataset format)
    manifest.json   how the data was made (args, counts, versions, rejects)
    split.json      {"train": [idx...], "val": [...], "test": [...]} (wpiper.make_split)

Spectrograms are not stored; training computes them on the GPU.
"""

import json
import random
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .dataset import Batch, Utterance

INT16_SCALE = 32768.0


def is_big_format(data_dir: Path) -> bool:
    return (Path(data_dir) / "audio.bin").exists()


class BigData:
    """Read access to a prepared dataset; cheap to pickle (memmap opens lazily)."""

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        idx = np.load(self.dir / "index.npz")
        self.audio_off, self.audio_len = idx["audio_off"], idx["audio_len"]
        self.ph_off, self.ph_len = idx["ph_off"], idx["ph_len"]
        self.phonemes = np.fromfile(self.dir / "phonemes.bin", dtype=np.int16)
        self.sample_rate = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))["audio"]["sample_rate"]
        self._audio: Optional[np.memmap] = None
        self._meta: Optional[list] = None

    def __len__(self) -> int:
        return len(self.audio_len)

    @property
    def audio(self) -> np.memmap:
        if self._audio is None:
            self._audio = np.memmap(self.dir / "audio.bin", dtype=np.int16, mode="r")
        return self._audio

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_audio"] = None
        return state

    def meta(self) -> list:
        """[(source id, text)] per clip."""
        if self._meta is None:
            self._meta = []
            for line in (self.dir / "meta.tsv").read_text(encoding="utf-8").splitlines():
                _i, src, text = line.split("\t", 2)
                self._meta.append((src, text))
        return self._meta

    def clip(self, i: int):
        a = self.audio[self.audio_off[i] : self.audio_off[i] + self.audio_len[i]]
        p = self.phonemes[self.ph_off[i] : self.ph_off[i] + self.ph_len[i]]
        return a, p

    def utterance(self, i: int) -> Utterance:
        a, p = self.clip(i)
        src, text = self.meta()[i]
        return Utterance(
            utt_id=str(i), text=text,
            phoneme_ids=torch.from_numpy(p.astype(np.int64)),
            audio=torch.from_numpy(a.astype(np.float32) / INT16_SCALE),
            spec=None,
        )


class BigDataset(Dataset):
    def __init__(self, data: BigData, indices: List[int]):
        self.data, self.indices = data, list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        i = self.indices[k]
        a, p = self.data.clip(i)
        return (
            torch.from_numpy(p.astype(np.int64)),
            torch.from_numpy(a.astype(np.float32) / INT16_SCALE),
        )

    def frames(self, hop: int) -> np.ndarray:
        return self.data.audio_len[self.indices] // hop


class BucketBatchSampler(Sampler):
    """Batches of similar-length clips under a padded-frame budget.

    A batch holds at most `max_batch` clips and at most `max_frames` padded
    frames (clips x longest clip), so long clips automatically get smaller
    batches and memory stays flat. Shuffling is deterministic per epoch.
    """

    def __init__(self, frames: np.ndarray, max_frames: int, max_batch: int,
                 shuffle: bool, seed: int = 0, chunk: int = 4096):
        self.frames, self.max_frames, self.max_batch = frames, max_frames, max_batch
        self.shuffle, self.seed, self.chunk = shuffle, seed, chunk
        self.epoch = 0
        self._len: Optional[int] = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.frames)))
        if self.shuffle:
            rng.shuffle(order)
        batches: List[List[int]] = []
        for start in range(0, len(order), self.chunk):
            part = sorted(order[start : start + self.chunk], key=lambda i: self.frames[i])
            cur: List[int] = []
            for i in part:
                longest = max(int(self.frames[i]), int(self.frames[cur[-1]]) if cur else 0)
                if cur and ((len(cur) + 1) * longest > self.max_frames or len(cur) >= self.max_batch):
                    batches.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        if self._len is None:
            self._len = len(self._batches())
        return self._len


class BigCollate:
    def __init__(self, segment_size: int, hop_length: int):
        self.segment_size, self.hop = segment_size, hop_length

    def __call__(self, items) -> Batch:
        n = len(items)
        max_ph = max(p.size(0) for p, _ in items)
        # Audio cannot be smaller than the training segment
        max_audio = max(self.segment_size, max(a.size(0) for _, a in items))
        phonemes = torch.zeros(n, max_ph, dtype=torch.long)
        audio = torch.zeros(n, 1, max_audio)
        ph_len, a_len = torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)
        for k, (p, a) in enumerate(items):
            phonemes[k, : p.size(0)] = p
            audio[k, 0, : a.size(0)] = a
            ph_len[k], a_len[k] = p.size(0), a.size(0)
        return Batch(phonemes, ph_len, None, a_len // self.hop, audio, a_len)


def load_split(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "split.json").read_text(encoding="utf-8"))
