from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATASET_CSV = ROOT / "dataset.csv"
SPEAKER_CSV = ROOT / "speaker_assignments.csv"
OUTPUT_CSV = ROOT / "speaker_split.csv"

SEED = 42


def load_data() -> pd.DataFrame:
    ds = pd.read_csv(DATASET_CSV)
    sa = pd.read_csv(SPEAKER_CSV)
    df = ds.merge(sa[["filename", "speaker_cluster"]], on="filename", how="inner")
    df["gender"] = (
        df["filename"]
        .str.extract(r"_(\d)\.wav$")[0]
        .astype(int)
        .map({0: "male", 1: "female"})
    )
    return df


def build_speaker_profiles(df: pd.DataFrame) -> pd.DataFrame:
    emotions = sorted(df["emotion"].unique())
    genders = sorted(df["gender"].unique())
    rows = []
    for spk, grp in df.groupby("speaker_cluster"):
        row = {"speaker_cluster": spk, "size": len(grp)}
        for e in emotions:
            row[f"emo_{e}"] = (grp["emotion"] == e).sum()
        for g in genders:
            row[f"gen_{g}"] = (grp["gender"] == g).sum()
        rows.append(row)
    return pd.DataFrame(rows)


def distribution_distance(
    current_counts: dict[str, int],
    target_proportions: dict[str, float],
    total: int,
) -> float:
    if total == 0:
        return 0.0
    dist = 0.0
    for key, target_p in target_proportions.items():
        observed_p = current_counts.get(key, 0) / total
        diff = observed_p - target_p
        dist += diff ** 2 / max(target_p, 1e-9)
    return dist


def _score_candidate(
    test_emo: dict[str, int],
    test_gen: dict[str, int],
    row: pd.Series,
    emotions: list[str],
    genders: list[str],
    corpus_emo_props: dict[str, float],
    corpus_gen_props: dict[str, float],
    test_size: int,
) -> tuple[float, dict[str, int], dict[str, int], int]:
    new_size = test_size + int(row["size"])
    new_emo = {e: test_emo[e] + int(row[f"emo_{e}"]) for e in emotions}
    new_gen = {g: test_gen[g] + int(row[f"gen_{g}"]) for g in genders}
    emo_dist = distribution_distance(new_emo, corpus_emo_props, new_size)
    gen_dist = distribution_distance(new_gen, corpus_gen_props, new_size)
    return emo_dist + 0.5 * gen_dist, new_emo, new_gen, new_size


def greedy_split(
    df: pd.DataFrame,
    test_ratio: float = 0.20,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    total_clips = len(df)
    target_test = int(round(total_clips * test_ratio))
    max_test = int(total_clips * (test_ratio + 0.03))

    emotions = sorted(df["emotion"].unique())
    genders = sorted(df["gender"].unique())

    corpus_emo_props = {e: (df["emotion"] == e).sum() / total_clips for e in emotions}
    corpus_gen_props = {g: (df["gender"] == g).sum() / total_clips for g in genders}

    profiles = build_speaker_profiles(df)
    profiles = profiles.sample(frac=1, random_state=rng).reset_index(drop=True)
    profiles = profiles.sort_values("size", ascending=False).reset_index(drop=True)

    test_speakers: set[int] = set()
    test_size = 0
    test_emo: dict[str, int] = {e: 0 for e in emotions}
    test_gen: dict[str, int] = {g: 0 for g in genders}
    assigned: set[int] = set()
    candidates = list(profiles.itertuples(index=False))

    while test_size < target_test:
        best_score = float("inf")
        best_idx = -1
        best_emo = best_gen = None
        best_new_size = 0

        for i, row_tuple in enumerate(candidates):
            if i in assigned:
                continue
            row = pd.Series(row_tuple._asdict())
            if test_size + int(row["size"]) > max_test:
                continue
            score, new_emo, new_gen, new_size = _score_candidate(
                test_emo, test_gen, row, emotions, genders,
                corpus_emo_props, corpus_gen_props, test_size,
            )
            if score < best_score:
                best_score = score
                best_idx = i
                best_emo = new_emo
                best_gen = new_gen
                best_new_size = new_size

        if best_idx == -1:
            break

        row = pd.Series(candidates[best_idx]._asdict())
        test_speakers.add(int(row["speaker_cluster"]))
        test_size = best_new_size
        test_emo = best_emo
        test_gen = best_gen
        assigned.add(best_idx)

    df = df.copy()
    df["split"] = df["speaker_cluster"].apply(lambda s: "test" if s in test_speakers else "train")
    return df


def validate_and_report(df: pd.DataFrame) -> None:
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    train_spk = set(train["speaker_cluster"].unique())
    test_spk = set(test["speaker_cluster"].unique())
    overlap = train_spk & test_spk
    assert len(overlap) == 0, f"Speaker overlap: {overlap}"

    total = len(df)
    print(f"\nTrain: {len(train)} clips ({len(train)/total:.1%}),  {len(train_spk)} speakers")
    print(f"Test:  {len(test)} clips ({len(test)/total:.1%}),  {len(test_spk)} speakers")
    print(f"Speaker overlap: {len(overlap)}\n")

    emotions = sorted(df["emotion"].unique())
    print(f"  {'Emotion':<10} {'Corpus':>8} {'Train':>8} {'Test':>8} {'Test%':>7}")
    print(f"  {'-'*43}")
    for e in emotions:
        c = (df["emotion"] == e).sum()
        tr = (train["emotion"] == e).sum()
        te = (test["emotion"] == e).sum()
        print(f"  {e:<10} {c:>8d} {tr:>8d} {te:>8d} {te/c:>6.1%}")

    genders = sorted(df["gender"].unique())
    print(f"\n  {'Gender':<10} {'Corpus':>8} {'Train':>8} {'Test':>8} {'Test%':>7}")
    print(f"  {'-'*43}")
    for g in genders:
        c = (df["gender"] == g).sum()
        tr = (train["gender"] == g).sum()
        te = (test["gender"] == g).sum()
        print(f"  {g:<10} {c:>8d} {tr:>8d} {te:>8d} {te/c:>6.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = load_data()
    print(f"{len(df)} clips, {df['speaker_cluster'].nunique()} speaker clusters")

    df = greedy_split(df, test_ratio=args.test_ratio, seed=args.seed)
    validate_and_report(df)

    out = df[["filename", "split"]].sort_values("filename")
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV.name} ({len(out)} rows)")


if __name__ == "__main__":
    main()
