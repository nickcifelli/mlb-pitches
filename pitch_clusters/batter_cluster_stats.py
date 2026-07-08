"""Aggregate raw pitches into per-(batter, stand, pitcher-hand, year, cluster) counts.

Mirrors pitch_clusters.pitcher_cluster_stats at the batter grain: the same
cluster assignments apply, since archetypes are pitch-shape based, not
pitcher-specific — a batter's performance against "cluster 3" is just as
well-defined as a pitcher's. Split by the *pitcher's* throwing hand rather
than the batter's own, since that's what determines which fitted GMM (and
which cluster_meta.json league averages) apply to a given pitch; the batter's
own stand is kept as a separate grouping key alongside it.

Metrics lean plate-discipline and contact-quality/value (chase rate,
zone-contact rate, hard-hit%, barrel%, wOBA/xwOBA against) rather than the
pitcher whiff/CSW/zone/GB set, though several overlap where the underlying
event is identical (whiff, CSW, zone%, GB%, EV-against) — those are kept with
the same names as pitcher_cluster_stats for easy comparison.

Usage:
    python -m pitch_clusters.batter_cluster_stats [--years 2021 2022 2023 2024]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pitch_clusters.assign import get_assigner
from pitch_clusters.batter_names import get_batter_names
from pitch_clusters.fit import (
    CLUSTER_FEATURES,
    _REPO_ROOT,
    _SWING_DESCRIPTIONS,
    _WHIFF_DESCRIPTIONS,
    _ZONE_NUMS,
    load_processed_years,
)

DERIVED_DIR = _REPO_ROOT / "data" / "derived"
COUNTS_PATH = DERIVED_DIR / "batter_cluster_counts.parquet"

_HARD_HIT_MIN_EV = 95.0
_BARREL_MIN_EV = 98.0
_BARREL_LA_RANGE = (8.0, 32.0)  # standard Statcast barrel proxy: EV >= 98 and LA in [8, 32]

_LOAD_COLUMNS = [
    "batter", "stand", "p_throws", "game_year",
    *CLUSTER_FEATURES,
    "description", "zone", "launch_speed", "launch_angle", "bb_type",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
]

_COUNT_COLS = [
    "n_pitches", "n_swings", "n_whiffs", "n_called_strikes", "n_csw",
    "n_in_zone", "n_out_zone", "n_chases", "n_zone_swings", "n_zone_contact",
    "n_bip", "n_gb", "n_hard_hit", "n_barrels",
    "n_ev", "sum_ev", "sum_ev_sq",
    "n_woba", "sum_woba", "sum_woba_sq",
    "n_xwoba", "sum_xwoba", "sum_xwoba_sq",
]


def load_full_pitch_data(years: list[int]) -> pd.DataFrame:
    """Load full (unsampled) pitch-level data for the given years."""
    return load_processed_years(years, _LOAD_COLUMNS)


def build_batter_cluster_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Assign clusters and aggregate to per-(batter, stand, pitcher-hand, year, cluster) counts.

    Drops rows with missing clustering features *before* assigning clusters,
    matching fit._prep_hand's convention, so cluster labels here stay
    consistent with how the model was fit.
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
    is_chase = is_swing & ~in_zone
    is_zone_swing = is_swing & in_zone
    is_zone_contact = is_zone_swing & ~is_whiff

    bb_type = clean["bb_type"].to_numpy()
    is_bip = pd.notna(bb_type)
    is_gb = bb_type == "ground_ball"

    ev = clean["launch_speed"].to_numpy(dtype=np.float64)
    la = clean["launch_angle"].to_numpy(dtype=np.float64)
    has_ev = is_bip & ~np.isnan(ev)
    ev_filled = np.where(has_ev, ev, 0.0)
    is_hard_hit = has_ev & (ev >= _HARD_HIT_MIN_EV)
    is_barrel = (
        has_ev & ~np.isnan(la) & (ev >= _BARREL_MIN_EV)
        & (la >= _BARREL_LA_RANGE[0]) & (la <= _BARREL_LA_RANGE[1])
    )

    woba_denom = clean["woba_denom"].to_numpy(dtype=np.float64)
    has_woba = ~np.isnan(woba_denom) & (woba_denom > 0)
    woba_value = clean["woba_value"].to_numpy(dtype=np.float64)
    woba_value_filled = np.where(has_woba, woba_value, 0.0)

    xwoba = clean["estimated_woba_using_speedangle"].to_numpy(dtype=np.float64)
    has_xwoba = ~np.isnan(xwoba)
    xwoba_filled = np.where(has_xwoba, xwoba, 0.0)

    frame = pd.DataFrame({
        "batter": clean["batter"].to_numpy(),
        "stand": clean["stand"].to_numpy(),
        "pitcher_hand": clean["p_throws"].to_numpy(),
        "year": clean["game_year"].to_numpy(),
        "cluster": clean["cluster"].to_numpy(),
        "is_swing": is_swing.astype(np.int64),
        "is_whiff": is_whiff.astype(np.int64),
        "is_cs": is_cs.astype(np.int64),
        "in_zone": in_zone.astype(np.int64),
        "is_chase": is_chase.astype(np.int64),
        "is_zone_swing": is_zone_swing.astype(np.int64),
        "is_zone_contact": is_zone_contact.astype(np.int64),
        "is_bip": is_bip.astype(np.int64),
        "is_gb": is_gb.astype(np.int64),
        "is_hard_hit": is_hard_hit.astype(np.int64),
        "is_barrel": is_barrel.astype(np.int64),
        "has_ev": has_ev.astype(np.int64),
        "ev": ev_filled,
        "ev_sq": ev_filled * ev_filled,
        "has_woba": has_woba.astype(np.int64),
        "woba_value": woba_value_filled,
        "woba_value_sq": woba_value_filled * woba_value_filled,
        "has_xwoba": has_xwoba.astype(np.int64),
        "xwoba": xwoba_filled,
        "xwoba_sq": xwoba_filled * xwoba_filled,
    })

    counts = frame.groupby(
        ["batter", "stand", "pitcher_hand", "year", "cluster"], as_index=False
    ).agg(
        n_pitches=("is_swing", "size"),
        n_swings=("is_swing", "sum"),
        n_whiffs=("is_whiff", "sum"),
        n_called_strikes=("is_cs", "sum"),
        n_in_zone=("in_zone", "sum"),
        n_chases=("is_chase", "sum"),
        n_zone_swings=("is_zone_swing", "sum"),
        n_zone_contact=("is_zone_contact", "sum"),
        n_bip=("is_bip", "sum"),
        n_gb=("is_gb", "sum"),
        n_hard_hit=("is_hard_hit", "sum"),
        n_barrels=("is_barrel", "sum"),
        n_ev=("has_ev", "sum"),
        sum_ev=("ev", "sum"),
        sum_ev_sq=("ev_sq", "sum"),
        n_woba=("has_woba", "sum"),
        sum_woba=("woba_value", "sum"),
        sum_woba_sq=("woba_value_sq", "sum"),
        n_xwoba=("has_xwoba", "sum"),
        sum_xwoba=("xwoba", "sum"),
        sum_xwoba_sq=("xwoba_sq", "sum"),
    )
    counts["n_csw"] = counts["n_whiffs"] + counts["n_called_strikes"]
    counts["n_out_zone"] = counts["n_pitches"] - counts["n_in_zone"]

    names = get_batter_names(counts["batter"].unique().tolist())
    counts = counts.merge(names, on="batter", how="left")
    counts["player_name"] = counts["player_name"].fillna(
        "batter #" + counts["batter"].astype(str)
    )

    return counts[[
        "batter", "player_name", "stand", "pitcher_hand", "year", "cluster",
        *_COUNT_COLS,
    ]]


def load_or_build_counts(years: list[int], rebuild: bool = False) -> pd.DataFrame:
    """Load the cached counts table, building (and caching) it if needed."""
    if not rebuild and COUNTS_PATH.exists():
        return pd.read_parquet(COUNTS_PATH)

    print(f"Building batter/cluster counts for years {years}...")
    df = load_full_pitch_data(years)
    counts = build_batter_cluster_counts(df)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_parquet(COUNTS_PATH, index=False)
    print(f"  saved {COUNTS_PATH} ({len(counts):,} rows)")
    return counts


def _cross_check_league_avgs(counts: pd.DataFrame) -> None:
    """Sanity check: the whiff/CSW/zone/GB/EV rates pooled across all batters
    for a given (pitcher hand, cluster) here are the same underlying pitches
    pitcher_cluster_stats aggregates from the other side, so they should land
    close to cluster_meta.json's league_avgs. Large deltas would mean the two
    aggregation pipelines have drifted apart.
    """
    import json

    from pitch_clusters.fit import OUTPUT_DIR as MODEL_DIR

    with open(MODEL_DIR / "cluster_meta.json") as f:
        meta = json.load(f)

    print("\nCross-check vs. cluster_meta.json league_avgs (hand, cluster: whiff / csw / zone / gb):")
    for hand, meta_key in [("R", "rhp"), ("L", "lhp")]:
        hand_counts = counts[counts["pitcher_hand"] == hand]
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
    parser = argparse.ArgumentParser(description="Build per-batter, per-cluster pitch/outcome counts")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    args = parser.parse_args()

    counts = load_or_build_counts(args.years, rebuild=True)
    print(f"\n{len(counts):,} (batter, stand, pitcher_hand, year, cluster) rows, "
          f"{counts['batter'].nunique():,} unique batters")

    _cross_check_league_avgs(counts)


if __name__ == "__main__":
    main()
