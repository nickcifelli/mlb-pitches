"""Empirical-Bayes shrinkage of per-batter, per-cluster performance rates.

Mirrors pitch_clusters.shrinkage at the batter grain, reusing its
shrink_metric/_rate_prior/_normal_prior — the Beta-Binomial and Normal-Normal
update rules and K-estimation are identical, only the metric set and mu
source differ. Unlike the pitcher side (whose mu comes from
cluster_meta.json, fixed at model-fit time), every metric's league-average mu
here is pooled directly from this run's own counts table. That's what lets
batter-only metrics with no cluster_meta.json entry (chase rate, barrel rate,
wOBA, xwOBA, ...) work the same way as the metrics that happen to share a
name with the pitcher side (whiff/CSW/zone/GB/EV-against) — one mu-sourcing
codepath for the whole metric set instead of two.

Usage:
    python -m pitch_clusters.batter_shrinkage [--years 2021 2022 2023 2024]
        [--grain career|season] [--min-n-for-prior 20] [--batter "name"]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pitch_clusters.assign import get_assigner
from pitch_clusters.batter_cluster_stats import (
    _COUNT_COLS,
    DERIVED_DIR,
    load_or_build_counts,
)
from pitch_clusters.shrinkage import _MIN_N_FOR_PRIOR, _normal_prior, _rate_prior, shrink_metric

SHRUNK_PATH = DERIVED_DIR / "batter_cluster_shrunk.parquet"

RATE_METRICS: dict[str, tuple[str, str]] = {
    "whiff_rate": ("n_whiffs", "n_swings"),
    "csw_rate": ("n_csw", "n_pitches"),
    "zone_pct": ("n_in_zone", "n_pitches"),
    "chase_rate": ("n_chases", "n_out_zone"),
    "zone_contact_rate": ("n_zone_contact", "n_zone_swings"),
    "gb_rate": ("n_gb", "n_bip"),
    "hard_hit_rate": ("n_hard_hit", "n_bip"),
    "barrel_rate": ("n_barrels", "n_bip"),
}
# metric -> (n_col, sum_col, sumsq_col), for the Normal-Normal continuous metrics
CONTINUOUS_METRICS: dict[str, tuple[str, str, str]] = {
    "avg_ev": ("n_ev", "sum_ev", "sum_ev_sq"),
    "woba": ("n_woba", "sum_woba", "sum_woba_sq"),
    "xwoba": ("n_xwoba", "sum_xwoba", "sum_xwoba_sq"),
}
ALL_METRICS = [*RATE_METRICS, *CONTINUOUS_METRICS]

_DEFAULT_K = {
    "whiff_rate": 80.0,
    "csw_rate": 120.0,
    "zone_pct": 150.0,
    "chase_rate": 100.0,
    "zone_contact_rate": 150.0,
    "gb_rate": 60.0,
    "hard_hit_rate": 80.0,
    "barrel_rate": 80.0,
    "avg_ev": 50.0,
    "woba": 100.0,
    "xwoba": 80.0,
}


def _pooled_mu(cluster_counts: pd.DataFrame, num_col: str, denom_col: str) -> float:
    """League-average rate/mean for this (hand, cluster), pooled across every
    batter row rather than sourced from cluster_meta.json."""
    denom = cluster_counts[denom_col].sum()
    return float(cluster_counts[num_col].sum() / denom) if denom > 0 else 0.0


def fit_shrinkage_priors(counts: pd.DataFrame, hand: str, min_n: int = _MIN_N_FOR_PRIOR) -> dict[int, dict[str, dict]]:
    """Estimate the empirical-Bayes prior (K, mu) per (cluster, metric) for one
    pitcher hand.

    `counts` should already be at the grain the priors will be applied at (one
    row per qualifying batter-unit per cluster); the cross-batter variance
    used to estimate K is taken over those rows.
    """
    assigner = get_assigner()
    hand_counts = counts[counts["pitcher_hand"] == hand]

    priors: dict[int, dict[str, dict]] = {}
    for cluster in range(assigner.n_components()):
        cluster_counts = hand_counts[hand_counts["cluster"] == cluster]

        metrics = {}
        for metric, (num_col, denom_col) in RATE_METRICS.items():
            mu = _pooled_mu(cluster_counts, num_col, denom_col)
            metrics[metric] = _rate_prior(cluster_counts, num_col, denom_col, mu, _DEFAULT_K[metric], min_n)
        for metric, (n_col, sum_col, sumsq_col) in CONTINUOUS_METRICS.items():
            mu = _pooled_mu(cluster_counts, sum_col, n_col)
            metrics[metric] = _normal_prior(cluster_counts, n_col, sum_col, sumsq_col, mu, _DEFAULT_K[metric], min_n)
        priors[cluster] = metrics
    return priors


def _collapse_to_career(counts: pd.DataFrame) -> pd.DataFrame:
    agg = {col: (col, "sum") for col in _COUNT_COLS}
    return counts.groupby(["batter", "stand", "pitcher_hand", "cluster"], as_index=False).agg(
        player_name=("player_name", "first"), **agg,
    )


def compute_shrunk_stats(
    counts: pd.DataFrame, grain: str = "career", min_n: int = _MIN_N_FOR_PRIOR
) -> pd.DataFrame:
    """Add raw_<metric>, shrunk_<metric>, K_<metric> columns for all 11 metrics.

    grain="career" (default) sums each batter's counts across years per
    (batter, stand, pitcher_hand, cluster) before shrinking — maximizes sample
    size, the best default for "how good is this batter against this pitch
    shape, really." grain="season" shrinks each (..., year, cluster) row
    directly instead, for recency-sensitive uses.
    """
    if grain == "career":
        work = _collapse_to_career(counts)
    elif grain == "season":
        work = counts.copy()
    else:
        raise ValueError(f"grain must be 'career' or 'season', got {grain!r}")

    for hand in ("R", "L"):
        priors = fit_shrinkage_priors(work, hand, min_n=min_n)
        mask = work["pitcher_hand"] == hand
        clusters = work.loc[mask, "cluster"]

        for metric, (num_col, denom_col) in RATE_METRICS.items():
            K = clusters.map(lambda c, m=metric: priors[c][m]["K"]).to_numpy(dtype=np.float64)
            mu = clusters.map(lambda c, m=metric: priors[c][m]["mu"]).to_numpy(dtype=np.float64)
            numerator = work.loc[mask, num_col].to_numpy(dtype=np.float64)
            n = work.loc[mask, denom_col].to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                work.loc[mask, f"raw_{metric}"] = np.where(n > 0, numerator / n, np.nan)
            work.loc[mask, f"shrunk_{metric}"] = shrink_metric(numerator, n, K, mu)
            work.loc[mask, f"K_{metric}"] = K

        for metric, (n_col, sum_col, _sumsq_col) in CONTINUOUS_METRICS.items():
            K = clusters.map(lambda c, m=metric: priors[c][m]["K"]).to_numpy(dtype=np.float64)
            mu = clusters.map(lambda c, m=metric: priors[c][m]["mu"]).to_numpy(dtype=np.float64)
            sum_val = work.loc[mask, sum_col].to_numpy(dtype=np.float64)
            n_val = work.loc[mask, n_col].to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                work.loc[mask, f"raw_{metric}"] = np.where(n_val > 0, sum_val / n_val, np.nan)
            work.loc[mask, f"shrunk_{metric}"] = shrink_metric(sum_val, n_val, K, mu)
            work.loc[mask, f"K_{metric}"] = K

    return work


def _print_priors_diagnostic(work: pd.DataFrame, min_n: int) -> None:
    print(f"\nFitted empirical-Bayes priors (min_n={min_n}):")
    header = f"  {'hand':>4} {'cluster':>7} {'metric':>18} {'K':>8} {'mu':>7} {'n_qual':>7}  fallback"
    print(header)
    for hand in ("R", "L"):
        priors = fit_shrinkage_priors(work, hand, min_n=min_n)
        for cluster, metrics in priors.items():
            for metric, p in metrics.items():
                print(f"  {hand:>4} {cluster:>7} {metric:>18} {p['K']:>8.1f} "
                      f"{p['mu']:>7.3f} {p['n_qualified']:>7}  {p['fallback']}")


def _print_batter_report(shrunk: pd.DataFrame, name_query: str) -> None:
    matches = shrunk[shrunk["player_name"].str.contains(name_query, case=False, na=False, regex=False)]
    if matches.empty:
        print(f"\nNo batter matching {name_query!r} found.")
        return

    unique_batters = matches[["batter", "player_name"]].drop_duplicates()
    if len(unique_batters) > 1:
        print(f"\nMultiple batters match {name_query!r}, be more specific:")
        for _, row in unique_batters.iterrows():
            print(f"  {row['player_name']} (id={row['batter']})")
        return

    name = unique_batters.iloc[0]["player_name"]
    sort_cols = ["stand", "pitcher_hand", "cluster", "year"] if "year" in matches.columns else \
        ["stand", "pitcher_hand", "cluster"]
    print(f"\n{name} — raw vs. shrunk:")
    for _, row in matches.sort_values(sort_cols).iterrows():
        year_tag = f" ({int(row['year'])})" if "year" in matches.columns else ""
        print(f"  vs {row['pitcher_hand']}HP, C{int(row['cluster'])}{year_tag} "
              f"(bats {row['stand']}): n={int(row['n_pitches'])}")
        for metric in ALL_METRICS:
            raw, shrunk_val, K = row[f"raw_{metric}"], row[f"shrunk_{metric}"], row[f"K_{metric}"]
            raw_s = f"{raw:.3f}" if pd.notna(raw) else "n/a"
            print(f"      {metric:>18}: raw={raw_s}  shrunk={shrunk_val:.3f}  (K={K:.1f})")


def main():
    parser = argparse.ArgumentParser(description="Empirical-Bayes shrinkage of per-batter, per-cluster performance rates")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument("--grain", choices=["career", "season"], default="career")
    parser.add_argument("--min-n-for-prior", type=int, default=_MIN_N_FOR_PRIOR)
    parser.add_argument("--batter", type=str, default=None,
                         help="Print raw-vs-shrunk stats for a batter (name substring match)")
    parser.add_argument("--rebuild-counts", action="store_true")
    args = parser.parse_args()

    counts = load_or_build_counts(args.years, rebuild=args.rebuild_counts)
    print(f"Loaded {len(counts):,} count rows for {counts['batter'].nunique():,} batters")

    work = _collapse_to_career(counts) if args.grain == "career" else counts
    _print_priors_diagnostic(work, args.min_n_for_prior)

    shrunk = compute_shrunk_stats(counts, grain=args.grain, min_n=args.min_n_for_prior)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    shrunk.to_parquet(SHRUNK_PATH, index=False)
    print(f"\nsaved {SHRUNK_PATH} ({len(shrunk):,} rows, grain={args.grain})")

    if args.batter:
        _print_batter_report(shrunk, args.batter)


if __name__ == "__main__":
    main()
