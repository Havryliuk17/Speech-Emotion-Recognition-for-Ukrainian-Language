"""Sweep-line overlap detection for diarised segments."""

from __future__ import annotations

import logging
from typing import Set

import pandas as pd

log = logging.getLogger(__name__)


def find_overlaps(df: pd.DataFrame) -> Set[int]:
    """Return row indices of segments that overlap with at least one other."""
    events: list[tuple[float, int, int]] = []
    for i, row in df.iterrows():
        s, e = float(row["start_time"]), float(row["end_time"])
        if e <= s:
            continue
        events.append((s, 0, int(i)))
        events.append((e, 1, int(i)))
    events.sort()

    active: set[int] = set()
    overlapped: set[int] = set()

    for _, ev_type, idx in events:
        if ev_type == 0:
            active.add(idx)
            if len(active) >= 2:
                overlapped.update(active)
        else:
            if len(active) >= 2:
                overlapped.update(active)
            active.discard(idx)

    return overlapped


def mark_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_overlap column to df. Returns the same DataFrame."""
    overlap_idx = find_overlaps(df)
    df["is_overlap"] = False
    if overlap_idx:
        df.loc[list(overlap_idx), "is_overlap"] = True
    log.info("Overlap: %d / %d segments marked", len(overlap_idx), len(df))
    return df
