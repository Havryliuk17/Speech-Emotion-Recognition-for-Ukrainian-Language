#!/usr/bin/env python3
"""Transcribe audio clips with Gemini and write a text column to the CSV.

Standalone usage:
    export GEMINI_API_KEY="your-key-here"
    python asr_pipeline.py
"""

import base64
import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")

FIELDNAMES = ["filename", "emotion", "duration_seconds", "text"]

_PROMPT = (
    "Transcribe the following Ukrainian audio clip. "
    "Return ONLY the transcribed text, nothing else. "
    "If the audio is unclear or has no speech, return an empty string."
)


def run_asr(
    csv_path: str | Path,
    api_key: str,
    path_col: str = "final_wav_path",
    clips_dir: str | None = None,
) -> None:
    """Transcribe clips listed in csv_path and save results in a text column.

    path_col is the CSV column that holds the audio path.
    If clips_dir is given, it is prepended to the value in path_col;
    otherwise the value is used as a full path.
    """
    csv_path = str(csv_path)
    client = genai.Client(api_key=api_key)

    rows = []
    fieldnames = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if "text" not in row:
                row["text"] = ""
            rows.append(row)

    if "text" not in fieldnames:
        fieldnames.append("text")

    pending = [(i, r) for i, r in enumerate(rows) if not r.get("text")]
    total = len(rows)

    print(f"\nTotal clips: {total}, already transcribed: {total - len(pending)}, remaining: {len(pending)}\n")
    if not pending:
        print("All clips already transcribed.")
        return

    def _save():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    for count, (idx, row) in enumerate(pending, 1):
        raw_path = row[path_col]
        filepath = os.path.join(clips_dir, raw_path) if clips_dir else raw_path

        # for standalone dataset.csv: also try clips/{emotion}/filename
        if not os.path.exists(filepath) and clips_dir and "emotion" in row:
            filepath = os.path.join(clips_dir, row["emotion"], raw_path)

        if not os.path.exists(filepath):
            print(f"  [{count}/{len(pending)}] SKIP (missing): {raw_path}")
            continue

        print(f"  [{count}/{len(pending)}] Transcribing {os.path.basename(filepath)}...", end=" ", flush=True)

        try:
            with open(filepath, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[{"parts": [
                    {"text": _PROMPT},
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                ]}],
            )
            text = response.text.strip() if response.text else ""
        except Exception as e:
            print(f"ERROR: {e}")
            text = ""
            time.sleep(2)

        rows[idx]["text"] = text
        print(f"'{text[:60]}{'...' if len(text) > 60 else ''}'")

        if count % 10 == 0:
            _save()
            print(f"  -- Saved progress ({count}/{len(pending)}) --")

        time.sleep(0.5)

    _save()
    print(f"\nDone. {total} clips processed. Results saved to {csv_path}")


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY environment variable.")
        sys.exit(1)

    run_asr(CSV_PATH, api_key, path_col="filename", clips_dir=CLIPS_DIR)


if __name__ == "__main__":
    main()
