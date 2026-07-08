"""Pitcher-vs-batter matchup engine: expected performance from combining a
pitcher's cluster usage mix with a batter's per-cluster shrunk profile.

Introduces no new statistical machinery — it's a combinator over the two
existing pipelines: arsenal_fingerprints' pitcher usage vectors (% of pitches
per cluster) and batter_shrinkage's per-cluster shrunk performance rates. For
each cluster the pitcher throws, the batter's shrunk rate in that cluster is
weighted by the pitcher's usage% there; summing across clusters gives an
overall "expected" line for this specific matchup, compared against what an
average batter would be expected to do against the same arsenal (the same
usage weights applied to each cluster's league-average-batter rate).

Usage:
    python -m pitch_clusters.matchup --pitcher "name" --batter "name"
        [--years 2021 2022 2023 2024] [--pitcher-year 2024]
"""

from __future__ import annotations

import argparse

import pandas as pd

from pitch_clusters.arsenal_fingerprints import (
    MIN_PITCHES_DEFAULT,
    _resolve_pitcher,
    build_usage_vectors,
)
from pitch_clusters.assign import get_assigner
from pitch_clusters.batter_cluster_stats import load_or_build_counts as load_batter_counts
from pitch_clusters.batter_shrinkage import (
    ALL_METRICS,
    CONTINUOUS_METRICS,
    RATE_METRICS,
    _pooled_mu,
)
from pitch_clusters.batter_shrinkage import compute_shrunk_stats as compute_batter_shrunk
from pitch_clusters.pitcher_cluster_stats import load_or_build_counts as load_pitcher_counts
from pitch_clusters.shrinkage import _MIN_N_FOR_PRIOR

_DISPLAY_METRICS = ["whiff_rate", "chase_rate", "hard_hit_rate", "barrel_rate", "woba", "xwoba"]
_USAGE_MIN_PCT = 0.01  # skip clusters a pitcher barely throws in the per-cluster breakdown


def _resolve_batter(counts: pd.DataFrame, name_query: str) -> tuple[int, str] | None:
    matches = counts[counts["player_name"].str.contains(name_query, case=False, na=False, regex=False)]
    if matches.empty:
        print(f"No batter matching {name_query!r} found.")
        return None

    unique = matches[["batter", "player_name"]].drop_duplicates()
    if len(unique) > 1:
        print(f"Multiple batters match {name_query!r}, be more specific:")
        for _, row in unique.iterrows():
            print(f"  {row['player_name']} (id={row['batter']})")
        return None

    row = unique.iloc[0]
    return int(row["batter"]), str(row["player_name"])


def _cluster_league_pooled_mu(batter_counts_hand: pd.DataFrame, cluster: int) -> dict[str, float]:
    """League-average-batter rate/mean per metric for one cluster, pooled
    across every batter row against this pitcher hand. Same pooling
    batter_shrinkage.fit_shrinkage_priors uses as its prior mean, so this is
    exactly what a zero-sample batter's shrunk stat would evaluate to.
    """
    cc = batter_counts_hand[batter_counts_hand["cluster"] == cluster]
    mu = {metric: _pooled_mu(cc, num_col, denom_col) for metric, (num_col, denom_col) in RATE_METRICS.items()}
    mu.update({
        metric: _pooled_mu(cc, sum_col, n_col)
        for metric, (n_col, sum_col, _sumsq_col) in CONTINUOUS_METRICS.items()
    })
    return mu


def _batter_hand_profile(batter_shrunk: pd.DataFrame, batter_id: int, hand: str) -> pd.DataFrame:
    """This batter's career shrunk stats per cluster against pitchers of `hand`.

    Switch hitters can have rows under both stand values for the same
    pitcher hand (mostly noise from rare same-side at-bats); keep whichever
    stand has more pitches per cluster since that's their standard side
    against this pitcher hand.
    """
    profile = batter_shrunk[(batter_shrunk["batter"] == batter_id) & (batter_shrunk["pitcher_hand"] == hand)]
    if profile["stand"].nunique() > 1:
        profile = profile.loc[profile.groupby("cluster")["n_pitches"].idxmax()]
    return profile.set_index("cluster")


def compute_matchup(usage_row: pd.Series, batter_profile: pd.DataFrame, batter_counts_hand: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster the pitcher throws >= _USAGE_MIN_PCT of the time,
    with this batter's shrunk metrics and the league-average-batter baseline
    for the same cluster. A batter with no career pitches in a cluster
    defaults to the baseline (shrink_metric with n=0 evaluates to mu anyway).
    """
    rows = []
    for cluster in range(get_assigner().n_components()):
        usage = usage_row.get(f"usage_{cluster}", 0.0)
        if usage < _USAGE_MIN_PCT:
            continue

        baseline = _cluster_league_pooled_mu(batter_counts_hand, cluster)
        row = {"cluster": cluster, "usage_pct": usage}
        has_data = cluster in batter_profile.index
        for metric in ALL_METRICS:
            row[f"batter_{metric}"] = batter_profile.loc[cluster, f"shrunk_{metric}"] if has_data else baseline[metric]
            row[f"league_{metric}"] = baseline[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_matchup(cluster_rows: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Usage-weighted expected value per metric: (this batter, league-average
    batter). Weights renormalize over the clusters present here so the long
    tail of barely-thrown pitch shapes doesn't quietly bias the total.
    """
    total_usage = cluster_rows["usage_pct"].sum()
    return {
        metric: (
            (cluster_rows["usage_pct"] * cluster_rows[f"batter_{metric}"]).sum() / total_usage,
            (cluster_rows["usage_pct"] * cluster_rows[f"league_{metric}"]).sum() / total_usage,
        )
        for metric in ALL_METRICS
    }


def _describe_cluster(hand: str, cluster: int) -> str:
    assigner = get_assigner()
    avgs = assigner.get_league_avgs(cluster, hand)
    if cluster in assigner.fastball_clusters(hand):
        kind = "fastball"
    elif cluster in assigner.breaking_clusters(hand):
        kind = "breaking"
    else:
        kind = "offspeed"
    return f"{kind} ({avgs.get('velo', 0):.1f} mph, {avgs.get('spin', 0):.0f} rpm)"


def print_matchup_report(
    pitcher_name: str, batter_name: str, hand: str,
    cluster_rows: pd.DataFrame, summary: dict[str, tuple[float, float]],
) -> None:
    print(f"\n{pitcher_name} ({hand}HP) vs {batter_name}")

    print("\nOverall expected performance (usage-weighted across pitcher's arsenal):")
    for metric in ALL_METRICS:
        batter_exp, league_exp = summary[metric]
        print(f"  {metric:>16}: batter={batter_exp:.3f}  league-avg-batter={league_exp:.3f}  "
              f"delta={batter_exp - league_exp:+.3f}")

    print("\nBy pitch shape (sorted by pitcher's usage):")
    for _, row in cluster_rows.sort_values("usage_pct", ascending=False).iterrows():
        cluster = int(row["cluster"])
        print(f"  C{cluster} {_describe_cluster(hand, cluster)} — usage={row['usage_pct']:.1%}")
        for metric in _DISPLAY_METRICS:
            b, league = row[f"batter_{metric}"], row[f"league_{metric}"]
            print(f"      {metric:>14}: batter={b:.3f}  league={league:.3f}  delta={b - league:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Pitcher-vs-batter matchup: usage-weighted expected performance")
    parser.add_argument("--pitcher", type=str, required=True, help="Pitcher (name substring match)")
    parser.add_argument("--batter", type=str, required=True, help="Batter (name substring match)")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument("--pitcher-year", type=int, default=None,
                         help="Pitcher season to use for usage mix (default: most recent qualifying year)")
    parser.add_argument("--min-pitches", type=int, default=MIN_PITCHES_DEFAULT,
                         help="Min pitches for the pitcher's season usage mix to qualify")
    parser.add_argument("--min-n-for-prior", type=int, default=_MIN_N_FOR_PRIOR)
    parser.add_argument("--rebuild-counts", action="store_true")
    args = parser.parse_args()

    pitcher_counts = load_pitcher_counts(args.years, rebuild=args.rebuild_counts)
    batter_counts = load_batter_counts(args.years, rebuild=args.rebuild_counts)

    all_vectors = pd.concat(
        [build_usage_vectors(pitcher_counts, hand, min_pitches=args.min_pitches) for hand in ("R", "L")],
        ignore_index=True,
    )
    resolved_pitcher = _resolve_pitcher(all_vectors, args.pitcher)
    if resolved_pitcher is None:
        return
    pitcher_id, hand = resolved_pitcher

    resolved_batter = _resolve_batter(batter_counts, args.batter)
    if resolved_batter is None:
        return
    batter_id, batter_name = resolved_batter

    hand_vectors = all_vectors[all_vectors["hand"] == hand]
    year = args.pitcher_year or int(hand_vectors.loc[hand_vectors["pitcher"] == pitcher_id, "year"].max())
    matches = hand_vectors[(hand_vectors["pitcher"] == pitcher_id) & (hand_vectors["year"] == year)]
    if matches.empty:
        print(f"No qualifying usage data for pitcher_id={pitcher_id} in {year} (>= {args.min_pitches} pitches)")
        return
    usage_row = matches.iloc[0]

    batter_shrunk = compute_batter_shrunk(batter_counts, grain="career", min_n=args.min_n_for_prior)
    batter_profile = _batter_hand_profile(batter_shrunk, batter_id, hand)
    batter_counts_hand = batter_counts[batter_counts["pitcher_hand"] == hand]

    cluster_rows = compute_matchup(usage_row, batter_profile, batter_counts_hand)
    summary = summarize_matchup(cluster_rows)

    print(f"\nUsing {usage_row['player_name']}'s {year} usage mix "
          f"({int(usage_row['n_pitches']):,} pitches, min={args.min_pitches})")
    print_matchup_report(usage_row["player_name"], batter_name, hand, cluster_rows, summary)


if __name__ == "__main__":
    main()
