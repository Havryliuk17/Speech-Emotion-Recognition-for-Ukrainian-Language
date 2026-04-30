"""End-to-end audio dataset preprocessing pipeline.

Usage:
    python pipeline.py --url "https://youtu.be/..." --output results/
"""

from __future__ import annotations

import json
import logging
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

import config
from download import convert_to_16k_mono, download_audio
from diarize import run_diarization
from overlap import mark_overlaps
from noise_filter import classify_clips, mark_noisy_clips
from snr import compute_snr_column
from denoise import denoise_low_snr
from normalize import normalize_all

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _mark_short_and_weak(
    df: pd.DataFrame,
    min_dur: float,
    min_speaker_total: float,
) -> pd.DataFrame:
    """Add is_short and is_weak_speaker flag columns."""
    df["is_short"] = df["duration"].astype(float) < min_dur

    kept = df[~(df["is_overlap"] | df["is_short"])]
    totals = kept.groupby("speaker_id")["duration"].sum()
    weak = set(totals[totals < min_speaker_total].index)

    df["is_weak_speaker"] = df["speaker_id"].isin(weak)

    log.info("Short clips: %d", df["is_short"].sum())
    log.info(
        "Weak speakers (< %.0fs total): %s",
        min_speaker_total,
        sorted(weak) if weak else "none",
    )
    return df


def _delete_wav_files(df: pd.DataFrame) -> int:
    """Delete WAV files for rows in df. Returns count removed."""
    removed = 0
    for path_str in df["wav_path"]:
        p = Path(path_str)
        if p.exists():
            p.unlink()
            removed += 1
    return removed


def run_pipeline(
    url: str,
    output_dir: str | Path,
    base_outdir: str | Path = "data",
    hf_token: str | None = None,
    panns_device: str = "cpu",
    gemini_api_key: str | None = None,
) -> Path:
    """Execute the full preprocessing pipeline for a single video."""
    t_start = time.time()
    output_dir = Path(output_dir)
    hf_token = hf_token or config.HF_TOKEN

    if not hf_token:
        raise ValueError("HF_TOKEN is required (set env var or pass explicitly)")

    log.info("STEP 1/10 — Download audio")
    raw_audio, meta = download_audio(url, outdir=base_outdir)
    folder = Path(meta["folder"])
    video_name = folder.name

    log.info("STEP 2/10 — Convert to 16 kHz mono")
    audio_16k = convert_to_16k_mono(raw_audio, folder)

    log.info("STEP 3/10 — Pyannote diarisation")
    work_dir = folder / "processed_data"
    df = run_diarization(audio_16k, work_dir, hf_token)
    total_after_diar = len(df)

    # annotation phase — add flags without deleting anything
    log.info("STEP 4/10 — Mark overlapping segments")
    df = mark_overlaps(df)

    log.info("STEP 5/10 — Mark short clips & weak speakers")
    df = _mark_short_and_weak(
        df,
        min_dur=config.MIN_CLIP_DURATION,
        min_speaker_total=config.MIN_SPEAKER_TOTAL,
    )

    log.info("STEP 6/10 — PANNs noise classification")
    already_flagged = df["is_overlap"] | df["is_short"] | df["is_weak_speaker"]
    eligible_df = df[~already_flagged].copy()
    eligible_df = classify_clips(eligible_df, device=panns_device)
    eligible_df = mark_noisy_clips(eligible_df)

    for col in ("panns_top1_label", "panns_top1_score",
                "panns_top2_label", "panns_top2_score", "is_noisy"):
        df[col] = pd.NA
    df.update(eligible_df)

    log.info("STEP 7/10 — Compute WADA-SNR")
    df = compute_snr_column(df)

    annotated_csv = work_dir / "annotated.csv"
    df.to_csv(annotated_csv, index=False)
    log.info("Audit trail saved: %s (%d rows)", annotated_csv, len(df))

    # filtering phase — drop flagged rows, denoise, export
    log.info("STEP 8/10 — Drop flagged clips")
    drop_mask = (
        df["is_overlap"]
        | df["is_short"]
        | df["is_weak_speaker"]
        | df["is_noisy"].fillna(False)
    )
    n_overlap = int(df["is_overlap"].sum())
    n_short = int(df["is_short"].sum())
    n_weak = int(df["is_weak_speaker"].sum())
    n_noisy = int(df["is_noisy"].fillna(False).sum())

    dropped_df = df[drop_mask]
    _delete_wav_files(dropped_df)
    df = df[~drop_mask].copy().reset_index(drop=True)
    log.info("Kept %d / %d clips after basic + noise filtering", len(df), total_after_diar)

    log.info("STEP 9/10 — Denoise low-SNR clips & re-check")
    df = denoise_low_snr(df, threshold=config.SNR_THRESHOLD)

    denoised_count = int(df["was_denoised"].sum())

    if denoised_count > 0:
        log.info("Re-computing SNR for %d denoised clips ...", denoised_count)
        import librosa
        from snr import wada_snr

        for i in df.index[df["was_denoised"]]:
            wav_path = Path(df.at[i, "wav_path"])
            if wav_path.exists():
                y, _ = librosa.load(str(wav_path), sr=config.SAMPLE_RATE, mono=True)
                df.at[i, "snr_db"] = wada_snr(y)

    still_bad = (df["snr_db"] < config.SNR_THRESHOLD) & df["was_denoised"]
    n_still_bad = int(still_bad.sum())
    if n_still_bad > 0:
        _delete_wav_files(df[still_bad])
        df = df[~still_bad].copy().reset_index(drop=True)
        log.info("Dropped %d clips still below %.1f dB after denoising", n_still_bad, config.SNR_THRESHOLD)

    log.info("STEP 10/10 — Loudness normalisation")
    clips_dir = output_dir / "clips"
    df = normalize_all(df, clips_dir, video_name, target_lufs=config.TARGET_LUFS)

    metadata = df[[
        "speaker_id", "duration", "start_time", "end_time",
        "snr_db", "was_denoised", "lufs_original", "lufs_normalized",
        "final_wav_path",
    ]].copy()
    metadata.insert(0, "video_name", video_name)
    metadata.rename(columns={"duration": "clip_duration"}, inplace=True)

    metadata_csv = output_dir / "metadata.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_csv, index=False)
    log.info("Metadata CSV: %s (%d clips)", metadata_csv, len(metadata))

    if gemini_api_key:
        log.info("Running ASR transcription on %d clips ...", len(metadata))
        from asr_pipeline import run_asr
        run_asr(metadata_csv, gemini_api_key, path_col="final_wav_path")

    report = {
        "source_url": url,
        "video_name": video_name,
        "total_segments_after_diarization": total_after_diar,
        "removed_overlap": n_overlap,
        "removed_short": n_short,
        "removed_weak_speaker": n_weak,
        "removed_noisy_panns": n_noisy,
        "denoised_count": denoised_count,
        "removed_low_snr_after_denoise": n_still_bad,
        "final_clip_count": len(df),
        "final_total_duration_sec": round(float(df["duration"].sum()), 2),
        "speakers_retained": sorted(df["speaker_id"].unique().tolist()),
        "elapsed_sec": round(time.time() - t_start, 1),
        "config": {
            "sample_rate": config.SAMPLE_RATE,
            "min_clip_duration": config.MIN_CLIP_DURATION,
            "min_speaker_total": config.MIN_SPEAKER_TOTAL,
            "merge_collar": config.MERGE_COLLAR,
            "snr_threshold": config.SNR_THRESHOLD,
            "target_lufs": config.TARGET_LUFS,
            "panns_score_threshold": config.PANNS_SCORE_THRESHOLD,
            "panns_reject_labels": sorted(config.PANNS_REJECT_LABELS),
            "model_versions": config.MODEL_VERSIONS,
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Report: %s", report_path)

    audio_16k.unlink(missing_ok=True)

    log.info("Pipeline complete for '%s': %d clips in %.0fs", video_name, len(df), time.time() - t_start)
    return output_dir


def main() -> None:
    parser = ArgumentParser(description="Audio dataset preprocessing pipeline")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--output", required=True, help="Output directory for final clips + CSV")
    parser.add_argument("--workdir", default="data", help="Working directory for intermediate files")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for PANNs")
    parser.add_argument("--gemini-key", default=None, help="Gemini API key for ASR (or set GEMINI_API_KEY env var)")
    args = parser.parse_args()

    run_pipeline(
        url=args.url,
        output_dir=args.output,
        base_outdir=args.workdir,
        hf_token=args.hf_token,
        panns_device=args.device,
        gemini_api_key=args.gemini_key or os.environ.get("GEMINI_API_KEY"),
    )


if __name__ == "__main__":
    main()
