"""PANNs-based audio event classification and noise filtering."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import PANNS_REJECT_LABELS, PANNS_SCORE_THRESHOLD, SAMPLE_RATE

log = logging.getLogger(__name__)


def classify_clips(
    df: pd.DataFrame,
    device: str = "cpu",
    sr: int = SAMPLE_RATE,
) -> pd.DataFrame:
    """Run PANNs AudioTagging on each clip and store top-2 predictions."""
    from panns_inference import AudioTagging, labels as panns_labels

    for col in ("panns_top1_label", "panns_top1_score",
                "panns_top2_label", "panns_top2_score"):
        df[col] = np.nan

    log.info("Loading PANNs AudioTagging (device=%s) ...", device)
    tagger = AudioTagging(checkpoint_path=None, device=device)

    for i in tqdm(df.index, desc="PANNs classify"):
        wav_path = Path(df.at[i, "wav_path"])
        if not wav_path.exists():
            log.warning("File missing, skipped: %s", wav_path)
            continue

        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
        clipwise, _ = tagger.inference(y[np.newaxis, :])
        scores = clipwise[0]

        top2 = np.argsort(scores)[-2:][::-1]
        df.at[i, "panns_top1_label"] = panns_labels[int(top2[0])]
        df.at[i, "panns_top1_score"] = float(scores[top2[0]])
        df.at[i, "panns_top2_label"] = panns_labels[int(top2[1])]
        df.at[i, "panns_top2_score"] = float(scores[top2[1]])

    log.info("PANNs classification done for %d clips", len(df))
    return df


def mark_noisy_clips(
    df: pd.DataFrame,
    reject_labels: set[str] = PANNS_REJECT_LABELS,
    score_threshold: float = PANNS_SCORE_THRESHOLD,
) -> pd.DataFrame:
    """Add is_noisy column; flags clips where a reject label appears in top-2 above threshold."""
    def _is_noisy(row: pd.Series) -> bool:
        for lbl_col, score_col in (
            ("panns_top1_label", "panns_top1_score"),
            ("panns_top2_label", "panns_top2_score"),
        ):
            label = str(row.get(lbl_col, ""))
            score = float(row.get(score_col, 0.0) or 0.0)
            if label in reject_labels and score >= score_threshold:
                return True
        return False

    df["is_noisy"] = df.apply(_is_noisy, axis=1)
    n_noisy = df["is_noisy"].sum()
    log.info("Noisy clips marked: %d / %d (threshold=%.2f)", n_noisy, len(df), score_threshold)
    return df
