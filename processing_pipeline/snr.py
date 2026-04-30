"""WADA-SNR estimation (Kim & Stern, Interspeech 2008)."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import SAMPLE_RATE

log = logging.getLogger(__name__)

# Lookup table: Gamma-distribution statistic → SNR in dB (speech shape = 0.4)
_DB_VALS = np.arange(-20, 101)
_G_VALS = np.array([
    0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186,
    0.40999006, 0.41027138, 0.41052627, 0.41101024, 0.41143264,
    0.41231718, 0.41337272, 0.41526426, 0.41781920, 0.42077252,
    0.42452799, 0.42918886, 0.43510373, 0.44234195, 0.45161485,
    0.46221153, 0.47491647, 0.48883809, 0.50509236, 0.52353709,
    0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496,
    0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148,
    0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418,
    0.95655449, 0.98353490, 1.01047155, 1.03620950, 1.06136425,
    1.08579312, 1.10948190, 1.13277995, 1.15472826, 1.17627308,
    1.19703503, 1.21671694, 1.23535898, 1.25364313, 1.27103891,
    1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935,
    1.36057270, 1.37345513, 1.38577122, 1.39733504, 1.40856397,
    1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831,
    1.46669568, 1.47486938, 1.48269965, 1.49034339, 1.49748214,
    1.50435106, 1.51076426, 1.51698915, 1.52290970, 1.52857800,
    1.53389835, 1.53912110, 1.54390650, 1.54858517, 1.55310776,
    1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767,
    1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681,
    1.59162477, 1.59419690, 1.59693155, 1.59944600, 1.60185011,
    1.60408668, 1.60627134, 1.60826199, 1.61004547, 1.61192472,
    1.61369656, 1.61534074, 1.61688905, 1.61838916, 1.61985374,
    1.62135878, 1.62268119, 1.62390423, 1.62513143, 1.62632463,
    1.62740270, 1.62842767, 1.62945532, 1.63033070, 1.63128026,
    1.63204102,
])

_BLOCK_SIZE = 100_000


def _wada_snr_block(wav: np.ndarray, eps: float = 1e-10) -> tuple[float, float]:
    """Return (signal_energy, noise_energy) for one block."""
    wav = wav.astype(np.float64)
    wav -= wav.mean()

    energy = float(np.sum(wav ** 2))
    if energy < eps:
        return 0.0, 0.0

    abs_wav = np.clip(np.abs(wav / np.abs(wav).max()), eps, None)

    v1 = max(eps, abs_wav.mean())
    v3 = np.log(v1) - np.log(abs_wav).mean()

    below = np.where(_G_VALS < v3)[0]
    if len(below) == 0:
        snr_db = float(_DB_VALS[0])
    elif below[-1] == len(_DB_VALS) - 1:
        snr_db = float(_DB_VALS[-1])
    else:
        snr_db = float(_DB_VALS[int(below[-1]) + 1])

    factor = 10.0 ** (snr_db / 10.0)
    noise_energy = energy / (1.0 + factor)
    signal_energy = energy * factor / (1.0 + factor)
    return signal_energy, noise_energy


def wada_snr(wav: np.ndarray) -> float:
    """Estimate SNR in dB for wav using WADA, processed in 100k-sample blocks."""
    acc_signal = 0.0
    acc_noise = 0.0

    for start in range(0, len(wav), _BLOCK_SIZE):
        block = wav[start : start + _BLOCK_SIZE]
        if len(block) < 64:
            continue
        sig, noise = _wada_snr_block(block)
        acc_signal += sig
        acc_noise += noise

    if acc_noise < 1e-20:
        return float("nan")

    return round(10.0 * np.log10(acc_signal / acc_noise), 2)


def compute_snr_column(
    df: pd.DataFrame,
    sr: int = SAMPLE_RATE,
    column: str = "snr_db",
) -> pd.DataFrame:
    """Add snr_db column to df with WADA-SNR for each clip."""
    snr_values: list[float] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="WADA-SNR"):
        wav_path = Path(row["wav_path"])
        if not wav_path.exists():
            snr_values.append(float("nan"))
            continue
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
        snr_values.append(wada_snr(y))

    df[column] = snr_values
    valid = pd.Series(snr_values).dropna()
    log.info(
        "SNR computed: n=%d  mean=%.1f dB  median=%.1f dB  min=%.1f dB",
        len(valid), valid.mean(), valid.median(), valid.min(),
    )
    return df
