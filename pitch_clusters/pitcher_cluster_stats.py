"""Aggregate raw pitches into per-(pitcher, hand, year, cluster) counts.

This is the shared fact table both pitch_clusters.shrinkage and
pitch_clusters.arsenal_fingerprints build on: real pitch/outcome counts per
pitcher per cluster, at full dataset scale (not the light sample
visualizations.py uses for scatter plots). Cached to
data/derived/pitcher_cluster_counts.parquet since it's the expensive step
(one full pass over the raw data) that everything downstream reuses.

Usage:
    python -m pitch_clusters.pitcher_cluster_stats [--years 2021 2022 2023 2024]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pitch_clusters.assign import get_assigner
from pitch_clusters.fit import (
    CLUSTER_FEATURES,
    _SWING_DESCRIPTIONS,
    _WHIFF_DESCRIPTIONS,
    _ZONE_NUMS,
    load_processed_years,
)
from pitch_clusters.fit import OUTPUT_DIR as MODEL_DIR
from pitch_clusters.fit import _REPO_ROOT

DERIVED_DIR = _REPO_ROOT / "data" / "derived"
COUNTS_PATH = DERIVED_DIR / "pitcher_cluster_counts.parquet"

_LOAD_COLUMNS = [
    "pitcher", "player_name", "p_throws", "game_year",
    *CLUSTER_FEATURES,
    "description", "zone", "launch_speed", "bb_type",
]


def load_full_pitch_data(years: list[int]) -> pd.DataFrame:
    """Load full (unsampled) pitch-level data for the given years.

    No per-year subsampling (unlike visualizations.load_pitch_sample) — accurate
    per-pitcher counts need every row.
    """
    return load_processed_years(years, _LOAD_COLUMNS)


def build_pitcher_cluster_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Assign clusters and aggregate to per-(pitcher, hand, year, cluster) counts.

    Drops rows with missing clustering features *before* assigning clusters,
    matching fit._prep_hand's convention, so cluster labels here stay
    consistent with how the model was fit (assigner.assign() on its own
    would silently fold incomplete rows into cluster 7).
    """
    clean = df.dropna(subset=CLUSTER_FEATURES).copy()
    n_dropped = len(df) - len(clean)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} pitches with NaN features "
              f"({100 * n_dropped / len(df):.1f}%)")

    clean["cluster"] = get_assigner().assign(clean).to_numpy()

    desc = clean["description"].to_numpy()
    is_swing = np.isin(desc, list(_SWING_DESCRIPTIONS))
    is_whiff = np.isin(desc, list(_WHIFF_DESCRIPTIONS))
    is_cs = desc == "called_strike"
    in_zone = np.isin(clean["zone"].to_numpy(), _ZONE_NUMS)
    bb_type = clean["bb_type"].to_numpy()
    is_bip = pd.notna(bb_type)
    is_gb = bb_type == "ground_ball"
    ev = clean["launch_speed"].to_numpy(dtype=np.float64)
    has_ev = is_bip & ~np.isnan(ev)
    ev_filled = np.where(has_ev, ev, 0.0)

    frame = pd.DataFrame({
        "pitcher": clean["pitcher"].to_numpy(),
        "player_name": clean["player_name"].to_numpy(),
        "hand": clean["p_throws"].to_numpy(),
        "year": clean["game_year"].to_numpy(),
        "cluster": clean["cluster"].to_numpy(),
        "is_swing": is_swing.astype(np.int64),
        "is_whiff": is_whiff.astype(np.int64),
        "is_cs": is_cs.astype(np.int64),
        "in_zone": in_zone.astype(np.int64),
        "is_bip": is_bip.astype(np.int64),
        "is_gb": is_gb.astype(np.int64),
        "has_ev": has_ev.astype(np.int64),
        "ev": ev_filled,
        "ev_sq": ev_filled * ev_filled,
    })

    counts = frame.groupby(["pitcher", "hand", "year", "cluster"], as_index=False).agg(
        player_name=("player_name", "first"),
        n_pitches=("is_swing", "size"),
        n_swings=("is_swing", "sum"),
        n_whiffs=("is_whiff", "sum"),
        n_called_strikes=("is_cs", "sum"),
        n_in_zone=("in_zone", "sum"),
        n_bip=("is_bip", "sum"),
        n_gb=("is_gb", "sum"),
        n_ev=("has_ev", "sum"),
        sum_ev=("ev", "sum"),
        sum_ev_sq=("ev_sq", "sum"),
    )
    counts["n_csw"] = counts["n_whiffs"] + counts["n_called_strikes"]
    return counts[[
        "pitcher", "player_name", "hand", "year", "cluster",
        "n_pitches", "n_swings", "n_whiffs", "n_called_strikes", "n_csw",
        "n_in_zone", "n_bip", "n_gb", "n_ev", "sum_ev", "sum_ev_sq",
    ]]


def load_or_build_counts(years: list[int], rebuild: bool = False) -> pd.DataFrame:
    """Load the cached counts table, building (and caching) it if needed."""
    if not rebuild and COUNTS_PATH.exists():
        return pd.read_parquet(COUNTS_PATH)

    print(f"Building pitcher/cluster counts for years {years}...")
    df = load_full_pitch_data(years)
    counts = build_pitcher_cluster_counts(df)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_parquet(COUNTS_PATH, index=False)
    print(f"  saved {COUNTS_PATH} ({len(counts):,} rows)")
    return counts


def _cross_check_league_avgs(counts: pd.DataFrame) -> None:
    """Sanity check: re-aggregate this table's rates per (hand, cluster) and
    compare against cluster_meta.json's league_avgs for the same clusters.
    Small deltas are expected (this table pools whatever years were passed on
    the CLI, which may differ from the years the model itself was fit on);
    large deltas would mean the aggregation logic has drifted from fit.py's.
    """
    import json

    with open(MODEL_DIR / "cluster_meta.json") as f:
        meta = json.load(f)

    print("\nCross-check vs. cluster_meta.json league_avgs (hand, cluster: whiff / csw / zone / gb):")
    for hand, meta_key in [("R", "rhp"), ("L", "lhp")]:
        hand_counts = counts[counts["hand"] == hand]
        league_avgs = meta[meta_key]["league_avgs"]
        for cluster, g in hand_counts.groupby("cluster"):
            if str(cluster) not in league_avgs:
                continue
            ref = league_avgs[str(cluster)]
            n_swings, n_pitches, n_bip = g["n_swings"].sum(), g["n_pitches"].sum(), g["n_bip"].sum()
            whiff = g["n_whiffs"].sum() / n_swings if n_swings else float("nan")
            csw = g["n_csw"].sum() / n_pitches if n_pitches else float("nan")
            zone = g["n_in_zone"].sum() / n_pitches if n_pitches else float("nan")
            gb = g["n_gb"].sum() / n_bip if n_bip else float("nan")
            print(
                f"  {hand} C{cluster}: "
                f"{whiff:.3f} vs {ref['whiff_rate']:.3f} / "
                f"{csw:.3f} vs {ref['csw_rate']:.3f} / "
                f"{zone:.3f} vs {ref['zone_pct']:.3f} / "
                f"{gb:.3f} vs {ref['gb_rate']:.3f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Build per-pitcher, per-cluster pitch/outcome counts")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    args = parser.parse_args()

    counts = load_or_build_counts(args.years, rebuild=True)
    print(f"\n{len(counts):,} (pitcher, hand, year, cluster) rows, "
          f"{counts['pitcher'].nunique():,} unique pitchers")

    _cross_check_league_avgs(counts)


if __name__ == "__main__":
    main()
