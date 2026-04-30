"""Pipeline configuration — thresholds, model IDs, and constants."""

import os

HF_TOKEN = os.getenv("HF_TOKEN")

SAMPLE_RATE = 16_000

PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
MERGE_COLLAR = 0.15  # max gap (s) to merge consecutive same-speaker segments

MIN_CLIP_DURATION = 2.5   # segments shorter than this are dropped
MIN_SPEAKER_TOTAL = 30.0  # speakers with less total speech are dropped

PANNS_SCORE_THRESHOLD = 0.3

PANNS_REJECT_LABELS = {
    "Music",
    "Laughter",
    "Chuckle, chortle",
    "Crying, sobbing",
    "Screaming",
    "Applause",
    "Clapping",
    "Cheering",
    "Crowd",
    "Noise",
    "Environmental noise",
    "Static",
    "Hiss",
    "Buzz",
}

SNR_THRESHOLD = 10.0  # clips below this are denoised; still below after → deleted

DENOISER_MODEL = "dns64"
DRY_WET = 0.0  # 0.0 = fully denoised

TARGET_LUFS = -23.0  # EBU R128

MODEL_VERSIONS = {
    "pyannote": "pyannote/speaker-diarization-3.1",
    "panns": "Cnn14_mAP=0.431.pth",
    "denoiser": "facebook/denoiser – dns64 (v0.1.5)",
    "pyloudnorm": "0.1.1",
}
