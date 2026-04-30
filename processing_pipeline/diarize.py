"""Speaker diarisation with pyannote, merging short same-speaker gaps."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pyannote.audio import Pipeline

from config import MERGE_COLLAR, PYANNOTE_MODEL, SAMPLE_RATE

log = logging.getLogger(__name__)


def _merge_segments(
    raw: list[tuple[str, float, float]],
    collar: float,
) -> list[tuple[str, float, float]]:
    """Merge consecutive same-speaker segments when the gap <= collar."""
    merged: list[tuple[str, float, float]] = []
    for spk, start, end in raw:
        if merged and merged[-1][0] == spk and (start - merged[-1][2]) <= collar:
            merged[-1] = (spk, merged[-1][1], max(merged[-1][2], end))
        else:
            merged.append((spk, start, end))
    return merged


def run_diarization(
    audio_path: Path,
    outdir: Path,
    hf_token: str,
    merge_collar: float = MERGE_COLLAR,
) -> pd.DataFrame:
    """Run pyannote diarisation and export per-speaker WAV chunks."""
    outdir = Path(outdir)
    chunks_dir = outdir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    y, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    total_dur = len(y) / sr
    log.info(
        "Audio loaded: sr=%d  duration=%.1fs  samples=%s",
        sr, total_dur, f"{len(y):,}",
    )

    t0 = time.time()
    log.info("Loading pyannote pipeline: %s", PYANNOTE_MODEL)
    pipe = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=hf_token)
    if pipe is None:
        raise RuntimeError("Pyannote pipeline failed to initialise")

    log.info("Running diarisation ...")
    annotation = pipe(str(audio_path))
    log.info("Diarisation finished in %.1fs", time.time() - t0)

    raw: list[tuple[str, float, float]] = []
    for seg, _, spk in annotation.itertracks(yield_label=True):
        raw.append((spk, float(seg.start), float(seg.end)))
    raw.sort(key=lambda x: (x[1], x[2]))

    merged = _merge_segments(raw, merge_collar)
    log.info(
        "Segments: %d raw -> %d merged (collar=%.2fs)",
        len(raw), len(merged), merge_collar,
    )

    speaker_map = {spk: idx for idx, spk in enumerate(sorted({s for s, _, _ in merged}))}

    rows = []
    for seg_idx, (spk, start, end) in enumerate(merged):
        s_smp = max(0, int(round(start * sr)))
        e_smp = min(len(y), int(round(end * sr)))
        if e_smp <= s_smp:
            continue

        spk_id = speaker_map[spk]
        duration = end - start
        fname = f"{seg_idx:06d}_spk{spk_id}_{int(start)}_{int(end)}.wav"
        fpath = chunks_dir / fname

        sf.write(str(fpath), y[s_smp:e_smp], sr)

        rows.append({
            "speaker_id": spk_id,
            "wav_path": str(fpath),
            "duration": round(duration, 3),
            "start_time": round(start, 3),
            "end_time": round(end, 3),
        })

    df = pd.DataFrame(rows)
    log.info("Exported %d chunks to %s", len(df), chunks_dir)

    audio_path.unlink(missing_ok=True)
    log.info("Deleted intermediate audio: %s", audio_path)

    return df
