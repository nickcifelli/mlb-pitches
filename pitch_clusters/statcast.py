"""Fetch and cache pitch-level Statcast data via pybaseball.

Month-by-month chunking keeps requests under Baseball Savant's row limits
and allows incremental caching. Data is stored as parquet for fast I/O. This
is the first step of the pipeline — run it before pitch_clusters.fit, since
fit.py reads from data/processed/, which only this module populates.

Usage:
    python -m pitch_clusters.statcast --years 2021 2022 2023 2024
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _REPO_ROOT / "data" / "raw"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"


def get_statcast_month(year: int, month: int) -> pd.DataFrame:
    """Fetch one calendar month of Statcast pitch data, with parquet caching."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = RAW_DIR / f"statcast_pitches_{year}_{month:02d}.parquet"

    if parquet_path.exists():
        logger.info(f"[cache hit] {parquet_path}")
        return pd.read_parquet(parquet_path)

    from pybaseball import statcast
    from pybaseball import cache as pb_cache
    pb_cache.enable()

    start = f"{year}-{month:02d}-01"
    if month == 12:
        end = f"{year}-12-31"
    else:
        end = (pd.Timestamp(f"{year}-{month + 1:02d}-01") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"[fetching] Statcast {start} to {end}")
    df = statcast(start_dt=start, end_dt=end, verbose=True)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info(f"[saved] {parquet_path}  ({len(df):,} rows)")
    return df


def get_statcast_season(year: int, months: range | list[int] | None = None) -> pd.DataFrame:
    """Fetch a full season of Statcast data, stitched from monthly chunks.

    Default months: March (3) through October (10).
    Saves to data/processed/statcast_pitches_{year}.parquet.
    """
    if months is None:
        months = range(3, 11)

    months = list(months)
    frames = []
    for i, m in enumerate(months):
        chunk = get_statcast_month(year, m)
        if len(chunk) > 0:
            frames.append(chunk)
        if i < len(months) - 1:
            time.sleep(2)  # be polite to Baseball Savant between requests

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"statcast_pitches_{year}.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    logger.info(f"[saved] {out}  ({len(df):,} rows)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache Statcast pitch data by season")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    args = parser.parse_args()

    # Application entrypoint, not library code — configure logging so fetch
    # progress is actually visible (the module itself only calls logger.info,
    # which is silent without a handler).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    for year in args.years:
        get_statcast_season(year)


if __name__ == "__main__":
    main()
