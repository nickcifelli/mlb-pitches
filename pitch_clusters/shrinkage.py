"""Empirical-Bayes shrinkage of per-pitcher, per-cluster outcome rates.

cluster_meta.json's league_avgs are described as "for use as shrinkage
priors," but nothing shrinks anything without this module. A pitcher's
observed whiff/CSW/zone/GB rate and average EV-against, for a given cluster,
get pulled toward the cluster's league average by an amount that shrinks as
the pitcher's own sample size in that cluster grows:

    shrunk = (numerator + K*mu) / (n + K)

K (the prior's "effective sample size") is estimated from the data itself via
method-of-moments — Beta-Binomial for the four rate metrics, a Normal-Normal
("random effects" / DerSimonian-Laird-style) estimator for the continuous
avg_ev_against — not an arbitrary constant. Both reduce to the same update
rule above, so there's one shared `shrink_metric` for all five metrics.

Usage:
    python -m pitch_clusters.shrinkage [--years 2021 2022 2023 2024]
        [--grain career|season] [--min-n-for-prior 20] [--pitcher "name"]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pitch_clusters.assign import get_assigner
from pitch_clusters.pitcher_cluster_stats import DERIVED_DIR, load_or_build_counts

SHRUNK_PATH = DERIVED_DIR / "pitcher_cluster_shrunk.parquet"

RATE_METRICS: dict[str, tuple[str, str]] = {
    "whiff_rate": ("n_whiffs", "n_swings"),
    "csw_rate": ("n_csw", "n_pitches"),
    "zone_pct": ("n_in_zone", "n_pitches"),
    "gb_rate": ("n_gb", "n_bip"),
}
ALL_METRICS = [*RATE_METRICS, "avg_ev_against"]

_DEFAULT_K = {
    "whiff_rate": 80.0,
    "csw_rate": 120.0,
    "zone_pct": 150.0,
    "gb_rate": 60.0,
    "avg_ev_against": 50.0,
}
_MIN_N_FOR_PRIOR = 20
_MIN_PITCHERS_FOR_PRIOR = 8
_K_FLOOR = 1.0
_VAR_FLOOR = 1e-6

_COUNT_COLS = [
    "n_pitches", "n_swings", "n_whiffs", "n_called_strikes", "n_csw",
    "n_in_zone", "n_bip", "n_gb", "n_ev", "sum_ev", "sum_ev_sq",
]


def shrink_metric(numerator: np.ndarray, n: np.ndarray, K: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Empirical-Bayes posterior mean: (numerator + K*mu) / (n + K).

    Shared update rule for Beta-Binomial rate metrics (numerator=successes,
    n=trials) and the Normal-Normal avg_ev_against metric (numerator=sum of
    EV values, n=count of batted balls) — same shape either way.
    """
    return (numerator + K * mu) / (n + K)


def _rate_prior(cluster_counts: pd.DataFrame, num_col: str, denom_col: str, mu: float, metric: str, min_n: int) -> dict:
    denom = cluster_counts[denom_col].to_numpy(dtype=np.float64)
    qualified = denom >= min_n
    n_qualified = int(qualified.sum())
    if n_qualified < _MIN_PITCHERS_FOR_PRIOR:
        return {"K": _DEFAULT_K[metric], "mu": mu, "n_qualified": n_qualified, "fallback": True}

    n_q = denom[qualified]
    p = cluster_counts[num_col].to_numpy(dtype=np.float64)[qualified] / n_q
    sample_var = float(np.var(p, ddof=1))
    expected_noise = float(np.mean(p * (1 - p) / n_q))
    sigma2_talent = sample_var - expected_noise

    if sigma2_talent <= _VAR_FLOOR:
        return {"K": _DEFAULT_K[metric], "mu": mu, "n_qualified": n_qualified, "fallback": True}

    K = mu * (1 - mu) / sigma2_talent - 1
    return {"K": max(K, _K_FLOOR), "mu": mu, "n_qualified": n_qualified, "fallback": False}


def _ev_prior(cluster_counts: pd.DataFrame, mu: float, min_n: int) -> dict:
    n_ev = cluster_counts["n_ev"].to_numpy(dtype=np.float64)
    qualified = n_ev >= min_n
    n_qualified = int(qualified.sum())
    if n_qualified < _MIN_PITCHERS_FOR_PRIOR:
        return {"K": _DEFAULT_K["avg_ev_against"], "mu": mu, "n_qualified": n_qualified, "fallback": True}

    n_q = n_ev[qualified]
    sum_ev_q = cluster_counts["sum_ev"].to_numpy(dtype=np.float64)[qualified]
    sum_ev_sq_q = cluster_counts["sum_ev_sq"].to_numpy(dtype=np.float64)[qualified]
    mean_ev = sum_ev_q / n_q

    within_dof = float(n_q.sum() - n_qualified)
    if within_dof <= 0:
        return {"K": _DEFAULT_K["avg_ev_against"], "mu": mu, "n_qualified": n_qualified, "fallback": True}
    sigma2_w = float((sum_ev_sq_q - n_q * mean_ev ** 2).sum() / within_dof)

    sample_var = float(np.var(mean_ev, ddof=1))
    expected_noise = float(np.mean(sigma2_w / n_q))
    tau2 = sample_var - expected_noise

    if tau2 <= _VAR_FLOOR or sigma2_w <= 0:
        return {"K": _DEFAULT_K["avg_ev_against"], "mu": mu, "n_qualified": n_qualified, "fallback": True}

    K = sigma2_w / tau2
    return {"K": max(K, _K_FLOOR), "mu": mu, "n_qualified": n_qualified, "fallback": False}


def fit_shrinkage_priors(counts: pd.DataFrame, hand: str, min_n: int = _MIN_N_FOR_PRIOR) -> dict[int, dict[str, dict]]:
    """Estimate the empirical-Bayes prior (K, mu) per (cluster, metric) for one hand.

    `counts` should already be at the grain the priors will be applied at
    (one row per qualifying "pitcher-unit" — a pitcher-career or a
    pitcher-season row — per cluster); the cross-pitcher variance used to
    estimate K is taken over those rows.
    """
    assigner = get_assigner()
    hand_counts = counts[counts["hand"] == hand]

    priors: dict[int, dict[str, dict]] = {}
    for cluster in range(assigner.n_components()):
        cluster_counts = hand_counts[hand_counts["cluster"] == cluster]
        league_avgs = assigner.get_league_avgs(cluster, hand)
        metrics = {
            metric: _rate_prior(cluster_counts, num_col, denom_col, league_avgs[metric], metric, min_n)
            for metric, (num_col, denom_col) in RATE_METRICS.items()
        }
        metrics["avg_ev_against"] = _ev_prior(cluster_counts, league_avgs["avg_ev_against"], min_n)
        priors[cluster] = metrics
    return priors


def _collapse_to_career(counts: pd.DataFrame) -> pd.DataFrame:
    agg = {col: (col, "sum") for col in _COUNT_COLS}
    return counts.groupby(["pitcher", "hand", "cluster"], as_index=False).agg(
        player_name=("player_name", "first"), **agg,
    )


def compute_shrunk_stats(
    counts: pd.DataFrame, grain: str = "career", min_n: int = _MIN_N_FOR_PRIOR
) -> pd.DataFrame:
    """Add raw_<metric>, shrunk_<metric>, K_<metric> columns for all 5 metrics.

    grain="career" (default) sums each pitcher's counts across years per
    (pitcher, hand, cluster) before shrinking — maximizes sample size, the
    best default for "how good is this pitch, really." grain="season" shrinks
    each (pitcher, year, hand, cluster) row directly instead, for
    recency-sensitive uses (note: pitcher-seasons are then the statistical
    unit for estimating K, folding year-to-year variation into the prior).
    """
    if grain == "career":
        work = _collapse_to_career(counts)
    elif grain == "season":
        work = counts.copy()
    else:
        raise ValueError(f"grain must be 'career' or 'season', got {grain!r}")

    for hand in ("R", "L"):
        priors = fit_shrinkage_priors(work, hand, min_n=min_n)
        mask = work["hand"] == hand
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

        K_ev = clusters.map(lambda c: priors[c]["avg_ev_against"]["K"]).to_numpy(dtype=np.float64)
        mu_ev = clusters.map(lambda c: priors[c]["avg_ev_against"]["mu"]).to_numpy(dtype=np.float64)
        sum_ev = work.loc[mask, "sum_ev"].to_numpy(dtype=np.float64)
        n_ev = work.loc[mask, "n_ev"].to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            work.loc[mask, "raw_avg_ev_against"] = np.where(n_ev > 0, sum_ev / n_ev, np.nan)
        work.loc[mask, "shrunk_avg_ev_against"] = shrink_metric(sum_ev, n_ev, K_ev, mu_ev)
        work.loc[mask, "K_avg_ev_against"] = K_ev

    return work


def _print_priors_diagnostic(work: pd.DataFrame, min_n: int) -> None:
    print(f"\nFitted empirical-Bayes priors (min_n={min_n}):")
    header = f"  {'hand':>4} {'cluster':>7} {'metric':>16} {'K':>8} {'mu':>7} {'n_qual':>7}  fallback"
    print(header)
    for hand in ("R", "L"):
        priors = fit_shrinkage_priors(work, hand, min_n=min_n)
        for cluster, metrics in priors.items():
            for metric, p in metrics.items():
                print(f"  {hand:>4} {cluster:>7} {metric:>16} {p['K']:>8.1f} "
                      f"{p['mu']:>7.3f} {p['n_qualified']:>7}  {p['fallback']}")


def _print_pitcher_report(shrunk: pd.DataFrame, name_query: str) -> None:
    matches = shrunk[shrunk["player_name"].str.contains(name_query, case=False, na=False, regex=False)]
    if matches.empty:
        print(f"\nNo pitcher matching {name_query!r} found.")
        return

    unique_pitchers = matches[["pitcher", "player_name"]].drop_duplicates()
    if len(unique_pitchers) > 1:
        print(f"\nMultiple pitchers match {name_query!r}, be more specific:")
        for _, row in unique_pitchers.iterrows():
            print(f"  {row['player_name']} (id={row['pitcher']})")
        return

    name = unique_pitchers.iloc[0]["player_name"]
    sort_cols = ["cluster", "year"] if "year" in matches.columns else ["cluster"]
    print(f"\n{name} — raw vs. shrunk:")
    for _, row in matches.sort_values(sort_cols).iterrows():
        year_tag = f" ({int(row['year'])})" if "year" in matches.columns else ""
        print(f"  C{int(row['cluster'])}{year_tag}: n={int(row['n_pitches'])}")
        for metric in ALL_METRICS:
            raw, shrunk_val, K = row[f"raw_{metric}"], row[f"shrunk_{metric}"], row[f"K_{metric}"]
            raw_s = f"{raw:.3f}" if pd.notna(raw) else "n/a"
            print(f"      {metric:>16}: raw={raw_s}  shrunk={shrunk_val:.3f}  (K={K:.1f})")


def main():
    parser = argparse.ArgumentParser(description="Empirical-Bayes shrinkage of per-cluster outcome rates")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument("--grain", choices=["career", "season"], default="career")
    parser.add_argument("--min-n-for-prior", type=int, default=_MIN_N_FOR_PRIOR)
    parser.add_argument("--pitcher", type=str, default=None,
                         help="Print raw-vs-shrunk stats for a pitcher (name substring match)")
    parser.add_argument("--rebuild-counts", action="store_true")
    args = parser.parse_args()

    counts = load_or_build_counts(args.years, rebuild=args.rebuild_counts)
    print(f"Loaded {len(counts):,} count rows for {counts['pitcher'].nunique():,} pitchers")

    work = _collapse_to_career(counts) if args.grain == "career" else counts
    _print_priors_diagnostic(work, args.min_n_for_prior)

    shrunk = compute_shrunk_stats(counts, grain=args.grain, min_n=args.min_n_for_prior)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    shrunk.to_parquet(SHRUNK_PATH, index=False)
    print(f"\nsaved {SHRUNK_PATH} ({len(shrunk):,} rows, grain={args.grain})")

    if args.pitcher:
        _print_pitcher_report(shrunk, args.pitcher)


if __name__ == "__main__":
    main()
