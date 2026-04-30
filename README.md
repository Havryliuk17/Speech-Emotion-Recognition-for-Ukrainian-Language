# UA-SER: Speech Emotion Recognition for Ukrainian Language

This repository contains the full research codebase for the thesis *"Speech Emotion Recognition for Ukrainian Language"*, including the data collection pipeline, dataset analysis, and model fine-tuning experiments.

---

## Quick Links

| Resource | Location |
|---|---|
| Dataset (UA-SER) | [https://huggingface.co/OlhaHavryliuk/wav2vec_ser] |
| Best model checkpoints | [https://drive.google.com/drive/folders/1YdSuqI0HnzgiIf9QaiRIMM5HHhI63OFQ?usp=sharing] |

---

## What's in This Repo

```
.
├── processing_pipeline/        # End-to-end data collection and preprocessing scripts
├── dataset_analysis/           # Exploratory notebooks (acoustic + text/cross-modal)
│   └── results/                # Pre-computed CSVs and figures used in the thesis
├── final_notebooks/            # Model fine-tuning and evaluation notebooks
└── dataset.csv                 # Full dataset metadata (952 samples, 400 speakers)
```

---

## Dataset — UA-SER

UA-SER is a Ukrainian-language speech emotion dataset built from publicly available YouTube content. Audio was sourced, diarized, filtered, denoised, and manually transcribed to produce 952 clips across four emotion classes.

| Emotion | Samples |
|---|---|
| Angry | 259 |
| Happy | 244 |
| Sad | 227 |
| Neutral | 222 |
| **Total** | **952** |

- **Speakers:** 400 unique speakers
- **Split:** 771 train / 181 test (speaker-stratified)
- **Duration:** 0.9 – 3.0 seconds per clip (avg ≈ 1.7 s)
- **Language:** Ukrainian, with full transcriptions included

> The dataset (audio files + metadata) is hosted on Hugging Face: [link above](#quick-links).  
> The `dataset.csv` file in this repo contains the metadata only (filename, emotion label, duration, transcription, speaker ID, split).

---

## Data Collection Pipeline

The `processing_pipeline/` directory holds the complete automated pipeline used to build UA-SER. Each step is a standalone script and can be run independently or via the main orchestrator.

```
pipeline.py           ← run this to execute all steps end-to-end
│
├── download.py       # Download audio from YouTube (yt-dlp), convert to 16 kHz mono WAV
├── diarize.py        # Speaker diarization (pyannote/speaker-diarization-3.1)
├── overlap.py        # Detect and discard overlapping speaker segments
├── noise_filter.py   # Reject non-speech events via PANNs (music, laughter, crowd, etc.)
├── snr.py            # Estimate signal-to-noise ratio (WADA-SNR)
├── denoise.py        # Denoise with Meta's Denoiser (dns64 checkpoint)
├── normalize.py      # Loudness normalization to EBU R128 −23 LUFS
├── asr_pipeline.py   # Transcribe clips via Google Gemini API
├── review_text.py    # Flask web UI for reviewing and correcting transcriptions
├── create_speaker_split.py  # Build speaker-stratified train/test split
└── config.py         # All thresholds and model IDs in one place
```

**Required environment variables:**

```bash
export HF_TOKEN=your_huggingface_token      # for pyannote diarization
export GEMINI_API_KEY=your_gemini_api_key   # only needed for ASR step
export CLAUDE_API_KEY=your_claude_api_key   # only needed for text analysis step
```

**Main dependencies:** `yt-dlp`, `pyannote.audio`, `panns-inference`, `denoiser`, `pyloudnorm`, `transformers`, `torch`, `librosa`, `soundfile`, `pandas`, `flask`

---

## Dataset Analysis

The `dataset_analysis/` notebooks explore the dataset before modeling. Both notebooks save figures to `dataset_analysis/results/` — these are the plots used in the thesis.

| Notebook | What it covers |
|---|---|
| `acoustic_feature_analysis.ipynb` | MFCCs, spectral features, pitch, energy, VAD-space analysis |
| `text_crossmodal_analysis.ipynb` | TF-IDF discriminative words, LLM-based text emotion classification, cross-modal agreement |

Pre-computed results (so you don't need to re-run everything):

- `results/features.csv` — acoustic feature vectors for all 952 clips
- `results/text_emotion_llm_results.csv` — per-clip LLM emotion predictions
- `results/vad_results.csv` — voice activity detection outputs

---

## Model Experiments

The `final_notebooks/` directory contains the three core experiments from the thesis. Each notebook is self-contained and includes training, evaluation, and a results summary.

| Notebook | Model | Description |
|---|---|---|
| `pretrained_ser_eval.ipynb` | Various | Zero-shot evaluation of off-the-shelf SER models on UA-SER |
| `finetune_wav2vec2.ipynb` | XLS-R 300M (Ukrainian) | Ablation study: fine-tuning a multilingual wav2vec2 model |
| `finetune_emotion2vec.ipynb` | emotion2vec+ base | Ablation study: fine-tuning an emotion-specialized model |

> Checkpoints for the best-performing model configurations are saved on Google Drive: [link above](#quick-links).  
> To load a checkpoint locally, download it from Drive and update the model path at the top of the relevant notebook.

---

## Reproducing the Results

If you want to reproduce the thesis results without re-collecting data:

1. Download the dataset from Hugging Face and place audio files in `data/audio/`
2. The metadata is already in `dataset.csv`
3. Run the analysis notebooks in `dataset_analysis/` (pre-computed results are included so this is optional)
4. Run the fine-tuning notebooks in `final_notebooks/` — or download the checkpoints from Google Drive to skip training

If you want to collect new data from scratch, configure `processing_pipeline/config.py` and run:

```bash
python processing_pipeline/pipeline.py
```

---

## Thesis Context

This project was developed as part of a bachelor's/master's thesis on speech emotion recognition for low-resource Slavic languages. Ukrainian has very limited SER resources, so a core contribution of this work is UA-SER itself — a balanced, speaker-diverse dataset with clean audio and human-verified transcriptions.

The modeling section explores whether models pretrained on other languages transfer to Ukrainian, and how much fine-tuning on a small domain-specific dataset helps.
