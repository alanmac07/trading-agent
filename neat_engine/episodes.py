"""
neat_engine/episodes.py

Slices the training data into overlapping windows per ticker. This is
the cheap, label-free stand-in for "market regimes": rather than hand-
labeling bull/bear/sideways periods, we just make sure every genome
sees many different multi-month slices across every available ticker,
so it can't overfit to one continuous run.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Episode:
    ticker: str
    df: pd.DataFrame  # already sliced + index reset


def build_episodes(train_csv_path: str, window: int = 250, stride: int = 180) -> list[Episode]:
    raw = pd.read_csv(train_csv_path, parse_dates=["Date"])
    episodes: list[Episode] = []

    for ticker, group in raw.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        n = len(group)
        if n < window:
            episodes.append(Episode(ticker=ticker, df=group))
            continue

        start = 0
        while start + window <= n:
            episodes.append(Episode(ticker=ticker, df=group.iloc[start:start + window].reset_index(drop=True)))
            start += stride

    return episodes


def sample_episodes_for_generation(all_episodes: list[Episode], k: int, rng) -> list[Episode]:
    """
    Evaluating every genome on ALL episodes every generation gets expensive
    fast (pop_size x n_episodes simulations per generation). Sampling k
    episodes per generation keeps cost bounded while still rotating through
    every ticker/window across the run, so genomes still get judged on
    breadth over time, just not all of it every single generation.
    """
    if k >= len(all_episodes):
        return list(all_episodes)
    return list(rng.sample(all_episodes, k))
