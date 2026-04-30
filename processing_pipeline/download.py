"""Download YouTube audio and convert to 16 kHz mono WAV."""

from __future__ import annotations

import logging
import re
import subprocess
import unicodedata
from pathlib import Path

import yt_dlp

from config import SAMPLE_RATE

log = logging.getLogger(__name__)


def slugify(text: str, max_len: int = 80) -> str:
    """Filesystem-safe version of text."""
    text = unicodedata.normalize("NFC", text).strip()
    text = re.sub(r'[\/\\:*?"<>|]', "", text)
    text = re.sub(r"\s+", "_", text)
    text = text[:max_len].strip("._-")
    return text or "video"


def download_audio(url: str, outdir: Path) -> tuple[Path, dict]:
    """Download best audio from url into outdir; return (path, metadata)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with yt_dlp.YoutubeDL({"noplaylist": True, "quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title", "")
    vid_id = info.get("id", "unknown")
    folder = outdir / slugify(title)
    folder.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "outtmpl": {"default": str(folder / "%(id)s.%(ext)s")},
        "format": "bestaudio/b",
        "writeinfojson": False,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 30,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    ext = info.get("ext", "webm")
    raw_audio = folder / f"{vid_id}.{ext}"

    meta = {
        "video_id": vid_id,
        "title": title,
        "channel": info.get("uploader", ""),
        "folder": str(folder),
    }
    log.info("Downloaded %s -> %s", url, raw_audio)
    return raw_audio, meta


def convert_to_16k_mono(src: Path, outdir: Path) -> Path:
    """Re-encode src to 16 kHz mono WAV via ffmpeg, then delete src."""
    outdir = Path(outdir)
    dst = outdir / "audio_16k.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        str(dst),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg failed:\n%s", result.stderr)
        raise RuntimeError(f"ffmpeg conversion failed for {src}")

    src.unlink(missing_ok=True)
    log.info("Converted to 16 kHz mono: %s (%.1f MB)", dst, dst.stat().st_size / 1e6)
    return dst
