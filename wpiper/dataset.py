"""Dataset and batching for the cache written by prepare_dataset."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, Dataset


@dataclass
class Utterance:
    utt_id: str
    text: str
    phoneme_ids: LongTensor
    audio: FloatTensor
    spec: Optional[FloatTensor]


@dataclass
class Batch:
    phoneme_ids: LongTensor
    phoneme_lengths: LongTensor
    spectrograms: Optional[FloatTensor]  # None: the trainer computes it on the GPU
    spectrogram_lengths: LongTensor
    audios: FloatTensor
    audio_lengths: LongTensor
    speaker_ids: Optional[LongTensor] = None

    def to(self, device) -> "Batch":
        move = lambda t: None if t is None else t.to(device, non_blocking=True)  # noqa: E731
        return Batch(
            move(self.phoneme_ids), move(self.phoneme_lengths), move(self.spectrograms),
            move(self.spectrogram_lengths), move(self.audios), move(self.audio_lengths),
        )


class VitsDataset(Dataset):
    """Keeps the whole (small) dataset in memory."""

    def __init__(self, data_dir: Path, utt_ids: Sequence[str]):
        self.utts = []
        for utt_id in utt_ids:
            d = torch.load(Path(data_dir) / "cache" / f"{utt_id}.pt", weights_only=True)
            self.utts.append(
                Utterance(utt_id, d["text"], d["phoneme_ids"], d["audio"], d["spec"])
            )

    def __len__(self) -> int:
        return len(self.utts)

    def __getitem__(self, idx: int) -> Utterance:
        return self.utts[idx]


class UtteranceCollate:
    def __init__(self, segment_size: int):
        self.segment_size = segment_size

    def __call__(self, utterances: Sequence[Utterance]) -> Batch:
        utts = sorted(utterances, key=lambda u: u.spec.size(1), reverse=True)
        n = len(utts)
        num_bins = utts[0].spec.size(0)

        # Audio cannot be smaller than the training segment
        max_audio = max(self.segment_size, max(u.audio.size(0) for u in utts))
        phonemes = LongTensor(n, max(u.phoneme_ids.size(0) for u in utts)).zero_()
        spec = FloatTensor(n, num_bins, max(u.spec.size(1) for u in utts)).zero_()
        audio = FloatTensor(n, 1, max_audio).zero_()
        phoneme_lengths, spec_lengths, audio_lengths = (LongTensor(n) for _ in range(3))

        for i, u in enumerate(utts):
            phonemes[i, : u.phoneme_ids.size(0)] = u.phoneme_ids
            spec[i, :, : u.spec.size(1)] = u.spec
            audio[i, 0, : u.audio.size(0)] = u.audio
            phoneme_lengths[i] = u.phoneme_ids.size(0)
            spec_lengths[i] = u.spec.size(1)
            audio_lengths[i] = u.audio.size(0)

        return Batch(phonemes, phoneme_lengths, spec, spec_lengths, audio, audio_lengths)


def make_big_loaders(data_dir: Path, batch_size: int, segment_size: int, hop_length: int,
                     num_workers: int, max_batch_frames: int, seed: int):
    from .bigdata import BigCollate, BigData, BigDataset, BucketBatchSampler, load_split

    data = BigData(data_dir)
    split = load_split(data_dir)
    collate = BigCollate(segment_size, hop_length)

    def loader(indices, shuffle):
        ds = BigDataset(data, indices)
        sampler = BucketBatchSampler(ds.frames(hop_length), max_batch_frames, batch_size, shuffle, seed)
        kwargs = dict(num_workers=num_workers, persistent_workers=True, prefetch_factor=4) if num_workers else {}
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, pin_memory=True, **kwargs)

    train_loader, val_loader = loader(split["train"], True), loader(split["val"], False)
    return train_loader, val_loader, [data.utterance(i) for i in split["test"]]


def make_loaders(data_dir: Path, batch_size: int, segment_size: int, num_workers: int = 0,
                 hop_length: int = 256, max_batch_frames: int = 24000, seed: int = 1234):
    """Returns (train_loader, val_loader, test_utterances).

    Picks the format by looking in data_dir: the large format (audio.bin) gets
    length-bucketed batches under a frame budget; the small per-utterance cache
    gets plain shuffled batches of batch_size.
    """
    from .bigdata import is_big_format

    data_dir = Path(data_dir)
    if is_big_format(data_dir):
        return make_big_loaders(data_dir, batch_size, segment_size, hop_length,
                                num_workers, max_batch_frames, seed)
    split = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    collate = UtteranceCollate(segment_size)

    train = VitsDataset(data_dir, split["train"])
    val = VitsDataset(data_dir, split["val"])
    test = VitsDataset(data_dir, split["test"])

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test.utts
