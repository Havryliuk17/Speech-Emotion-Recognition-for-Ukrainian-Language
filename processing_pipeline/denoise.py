"""Denoise audio clips using Meta's Denoiser (dns64 checkpoint)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

from config import DENOISER_MODEL, DRY_WET, SAMPLE_RATE, SNR_THRESHOLD

log = logging.getLogger(__name__)


def load_denoiser(model_name: str = DENOISER_MODEL) -> torch.nn.Module:
    """Load and return the pretrained Meta Denoiser model."""
    from denoiser import pretrained

    loader = getattr(pretrained, model_name)
    model = loader()
    model.eval()
    log.info("Loaded denoiser model: %s", model_name)
    return model


def denoise_clip(wav_path: Path, model: torch.nn.Module, dry_wet: float = DRY_WET) -> None:
    """Denoise a single WAV file in-place."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    wav_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        denoised = model(wav_tensor).squeeze(0)

    if dry_wet > 0:
        original = torch.from_numpy(audio).unsqueeze(0)
        denoised = dry_wet * original + (1 - dry_wet) * denoised

    peak = denoised.abs().max()
    if peak > 0.99:
        denoised = denoised / peak * 0.99

    sf.write(str(wav_path), denoised.squeeze(0).numpy(), SAMPLE_RATE, subtype="PCM_16")


def denoise_low_snr(
    df: pd.DataFrame,
    threshold: float = SNR_THRESHOLD,
) -> pd.DataFrame:
    """Denoise clips below threshold SNR; adds was_denoised boolean column."""
    mask = df["snr_db"] < threshold
    to_denoise = df.loc[mask]

    df["was_denoised"] = False

    if to_denoise.empty:
        log.info("No clips below %.1f dB – nothing to denoise", threshold)
        return df

    log.info("Denoising %d / %d clips (SNR < %.1f dB) ...", len(to_denoise), len(df), threshold)
    model = load_denoiser()

    for i in tqdm(to_denoise.index, desc="Denoising"):
        wav_path = Path(df.at[i, "wav_path"])
        if not wav_path.exists():
            continue
        denoise_clip(wav_path, model)
        df.at[i, "was_denoised"] = True

    log.info("Denoised %d clips", df["was_denoised"].sum())
    return df
