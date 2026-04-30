"""EBU R128 loudness normalisation for audio clips."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

from config import SAMPLE_RATE, TARGET_LUFS

log = logging.getLogger(__name__)


def normalize_clip(
    src: Path,
    dst: Path,
    meter: pyln.Meter,
    target_lufs: float = TARGET_LUFS,
) -> dict:
    """Normalise src to target_lufs and write to dst; returns original and achieved LUFS."""
    data, sr = sf.read(str(src), dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)

    original_lufs = meter.integrated_loudness(data)

    if np.isinf(original_lufs) or np.isnan(original_lufs):
        sf.write(str(dst), data, sr, subtype="PCM_16")
        return {"lufs_original": float("nan"), "lufs_normalized": float("nan")}

    normalised = pyln.normalize.loudness(data, original_lufs, target_lufs)

    peak = np.max(np.abs(normalised))
    if peak > 1.0:
        normalised = normalised / peak * 0.99

    sf.write(str(dst), normalised, sr, subtype="PCM_16")

    normalised_lufs = meter.integrated_loudness(normalised)
    return {
        "lufs_original": round(original_lufs, 2),
        "lufs_normalized": round(normalised_lufs, 2),
    }


def normalize_all(
    df: pd.DataFrame,
    output_dir: Path,
    video_name: str,
    target_lufs: float = TARGET_LUFS,
) -> pd.DataFrame:
    """Normalise all clips in df and write to output_dir; adds lufs and final_wav_path columns."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meter = pyln.Meter(SAMPLE_RATE)

    lufs_orig: list[float] = []
    lufs_norm: list[float] = []
    final_paths: list[str] = []

    for seq, (i, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc="Normalising")
    ):
        src = Path(row["wav_path"])
        fname = f"{video_name}_spk{row['speaker_id']}_{seq:04d}.wav"
        dst = output_dir / fname

        if not src.exists():
            lufs_orig.append(float("nan"))
            lufs_norm.append(float("nan"))
            final_paths.append("")
            continue

        info = normalize_clip(src, dst, meter, target_lufs)
        lufs_orig.append(info["lufs_original"])
        lufs_norm.append(info["lufs_normalized"])
        final_paths.append(str(dst))

    df["lufs_original"] = lufs_orig
    df["lufs_normalized"] = lufs_norm
    df["final_wav_path"] = final_paths

    valid = pd.Series(lufs_orig).dropna()
    log.info(
        "Normalised %d clips to %.1f LUFS (median original: %.1f LUFS)",
        len(valid), target_lufs, valid.median(),
    )
    return df
