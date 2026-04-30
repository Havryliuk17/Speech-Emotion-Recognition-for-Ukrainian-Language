#!/usr/bin/env python3
"""Browser UI for reviewing and correcting ASR transcriptions.

Usage:
    python review_text.py
"""

import csv
import json
import os
import webbrowser
import wave
from threading import Timer

from flask import Flask, redirect, render_template, request, send_from_directory, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")
REVIEWED_PATH = os.path.join(BASE_DIR, "reviewed_clips.json")

FIELDNAMES = ["filename", "emotion", "duration_seconds", "text"]

app = Flask(__name__)


def load_dataset():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_dataset(rows):
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_reviewed():
    if os.path.exists(REVIEWED_PATH):
        with open(REVIEWED_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_reviewed(reviewed):
    with open(REVIEWED_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(reviewed), f, ensure_ascii=False, indent=2)


def get_clip_path(row):
    """Return full path to the audio file (flat or nested under clips/{emotion}/)."""
    flat = os.path.join(CLIPS_DIR, row["filename"])
    if os.path.exists(flat):
        return flat
    return os.path.join(CLIPS_DIR, row["emotion"], row["filename"])


def get_duration(filepath):
    try:
        with wave.open(filepath, "r") as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:
        return 0.0


@app.route("/")
def index():
    rows = load_dataset()
    reviewed = load_reviewed()
    total = len(rows)

    pending = [(i, r) for i, r in enumerate(rows) if r["filename"] not in reviewed]
    done = total - len(pending)

    if not pending:
        return render_template(
            "review.html",
            clip=None,
            done=done,
            total=total,
            edited_count=len(reviewed),
        )

    idx, row = pending[0]
    filepath = get_clip_path(row)
    duration = get_duration(filepath)
    flat_path = os.path.join(CLIPS_DIR, row["filename"])
    audio_url = row["filename"] if os.path.exists(flat_path) else f"{row['emotion']}/{row['filename']}"

    return render_template(
        "review.html",
        clip=row["filename"],
        clip_index=idx,
        emotion=row["emotion"],
        text=row.get("text", ""),
        duration=duration,
        audio_url=audio_url,
        done=done,
        total=total,
        remaining=len(pending),
    )


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(CLIPS_DIR, filename)


@app.route("/confirm", methods=["POST"])
def confirm():
    clip_name = request.form.get("clip_name")
    if clip_name:
        reviewed = load_reviewed()
        reviewed.add(clip_name)
        save_reviewed(reviewed)
    return redirect(url_for("index"))


@app.route("/update_text", methods=["POST"])
def update_text():
    clip_name = request.form.get("clip_name")
    new_text = request.form.get("new_text", "").strip()
    clip_index = request.form.get("clip_index")

    if clip_name and clip_index is not None:
        rows = load_dataset()
        idx = int(clip_index)
        if 0 <= idx < len(rows) and rows[idx]["filename"] == clip_name:
            rows[idx]["text"] = new_text
            save_dataset(rows)

        reviewed = load_reviewed()
        reviewed.add(clip_name)
        save_reviewed(reviewed)

    return redirect(url_for("index"))


@app.route("/skip", methods=["POST"])
def skip():
    clip_name = request.form.get("clip_name")
    if clip_name:
        reviewed = load_reviewed()
        reviewed.add(clip_name)
        save_reviewed(reviewed)
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = 5051
    Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\n  Text Review UI running at http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
