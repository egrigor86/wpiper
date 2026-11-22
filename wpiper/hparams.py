"""Model hyper-parameters (Piper "medium" defaults) and model construction."""

from dataclasses import asdict, dataclass, field
from typing import Optional

from .vits.models import MultiPeriodDiscriminator, SynthesizerTrn


# Architecture changes per quality, as in piper_train. Anything not listed keeps the
# Hparams default (the "medium" model, which is what every run before presets used).
QUALITY_OVERRIDES = {
    "x-low": dict(hidden_channels=96, inter_channels=96, filter_channels=384),
    "low": {},
    "medium": {},
    "high": dict(
        resblock="1", resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates=(8, 8, 2, 2), upsample_initial_channel=512, upsample_kernel_sizes=(16, 16, 4, 4),
    ),
}
QUALITIES = tuple(QUALITY_OVERRIDES)
# Sample rate each quality is normally trained at (Piper convention).
QUALITY_SAMPLE_RATE = {"x-low": 16000, "low": 16000, "medium": 22050, "high": 22050}


@dataclass
class Hparams:
    sample_rate: int = 22050
    num_symbols: int = 256
    num_speakers: int = 1
    # audio
    resblock: str = "2"
    resblock_kernel_sizes: tuple = (3, 5, 7)
    resblock_dilation_sizes: tuple = ((1, 2), (2, 6), (3, 12))
    upsample_rates: tuple = (8, 8, 4)
    upsample_initial_channel: int = 256
    upsample_kernel_sizes: tuple = (16, 16, 8)
    # mel
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None
    # model
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    use_spectral_norm: bool = False
    gin_channels: int = 0
    use_sdp: bool = True
    segment_size: int = 8192
    # training
    learning_rate: float = 2e-4
    learning_rate_d: float = 1e-4
    betas: tuple = (0.8, 0.99)
    betas_d: tuple = (0.5, 0.9)
    eps: float = 1e-9
    lr_decay: float = 0.999875
    lr_decay_d: float = 0.9999
    c_mel: int = 45
    c_kl: float = 1.0

    quality: str = "medium"

    @staticmethod
    def for_quality(quality: str = "medium", **kwargs) -> "Hparams":
        """Piper's size presets. "medium" (and "low") are the defaults above."""
        if quality not in QUALITIES:
            raise ValueError(f"unknown quality {quality!r}; choose from {', '.join(QUALITIES)}")
        return Hparams(quality=quality, **{**QUALITY_OVERRIDES[quality], **kwargs})

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Hparams":
        def tup(v):
            return tuple(tup(i) for i in v) if isinstance(v, list) else v

        fields = Hparams.__dataclass_fields__
        return Hparams(**{k: tup(v) for k, v in data.items() if k in fields})


def build_models(hp: Hparams) -> tuple[SynthesizerTrn, MultiPeriodDiscriminator]:
    if hp.num_speakers > 1 and hp.gin_channels <= 0:
        hp.gin_channels = 512

    model_g = SynthesizerTrn(
        n_vocab=hp.num_symbols,
        spec_channels=hp.filter_length // 2 + 1,
        segment_size=hp.segment_size // hp.hop_length,
        inter_channels=hp.inter_channels,
        hidden_channels=hp.hidden_channels,
        filter_channels=hp.filter_channels,
        n_heads=hp.n_heads,
        n_layers=hp.n_layers,
        kernel_size=hp.kernel_size,
        p_dropout=hp.p_dropout,
        resblock=hp.resblock,
        resblock_kernel_sizes=hp.resblock_kernel_sizes,
        resblock_dilation_sizes=hp.resblock_dilation_sizes,
        upsample_rates=hp.upsample_rates,
        upsample_initial_channel=hp.upsample_initial_channel,
        upsample_kernel_sizes=hp.upsample_kernel_sizes,
        n_speakers=hp.num_speakers,
        gin_channels=hp.gin_channels,
        use_sdp=hp.use_sdp,
    )
    model_d = MultiPeriodDiscriminator(use_spectral_norm=hp.use_spectral_norm)
    return model_g, model_d
