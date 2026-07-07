"""Pitcher-level arsenal fingerprints: comps, archetypes, and drift.

Aggregates each pitcher-season into an 8-dim usage vector (% of pitches per
cluster) and uses it for three things:
    - comps: cosine-similarity nearest neighbors ("who throws like X")
    - archetypes: KMeans over usage vectors ("power fastball/slider" vs.
      "6-pitch mix-and-match" style groupings)
    - drift: the same pitcher's usage vector across multiple years

RHP and LHP are always kept separate — the two hands' GMMs are fit
independently, so even though cluster shape is aligned via the pfx_x
mirroring in pitch_clusters.fit, usage-% distributions between hands aren't
directly comparable.

Usage:
    python -m pitch_clusters.arsenal_fingerprints [--years 2021 2022 2023 2024]
        [--min-pitches 200] [--n-archetypes 6] [--comps "name"] [--drift "name"]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from pitch_clusters.assign import get_assigner
from pitch_clusters.pitcher_cluster_stats import DERIVED_DIR, load_or_build_counts

VECTORS_PATH = DERIVED_DIR / "pitcher_usage_vectors.parquet"
MIN_PITCHES_DEFAULT = 200
N_ARCHETYPES_DEFAULT = 6
HAND_LABEL = {"R": "RHP", "L": "LHP"}


def build_usage_vectors(counts: pd.DataFrame, hand: str, min_pitches: int = MIN_PITCHES_DEFAULT) -> pd.DataFrame:
    """One row per qualifying (pitcher, year) for this hand: usage_0..usage_{n-1}
    (% of that pitcher-year's pitches per cluster, zero-filled, sums to 1).
    Filters to n_pitches >= min_pitches (drops small-sample call-ups).
    """
    n_components = get_assigner().n_components()
    hand_counts = counts[counts["hand"] == hand]

    totals = hand_counts.groupby(["pitcher", "year"], as_index=False).agg(
        player_name=("player_name", "first"), n_pitches=("n_pitches", "sum"),
    )
    qualifying_keys = totals.loc[totals["n_pitches"] >= min_pitches, ["pitcher", "year"]]

    pivot = (
        hand_counts.pivot_table(
            index=["pitcher", "year"], columns="cluster", values="n_pitches",
            aggfunc="sum", fill_value=0,
        )
        .reindex(columns=range(n_components), fill_value=0)
        .reset_index()
    )
    pivot.columns = ["pitcher", "year", *(f"usage_{c}" for c in range(n_components))]

    vectors = qualifying_keys.merge(totals, on=["pitcher", "year"]).merge(pivot, on=["pitcher", "year"])
    usage_cols = [f"usage_{c}" for c in range(n_components)]
    vectors[usage_cols] = vectors[usage_cols].div(vectors["n_pitches"], axis=0)
    vectors.insert(2, "hand", hand)
    return vectors[["pitcher", "player_name", "hand", "year", "n_pitches", *usage_cols]].reset_index(drop=True)


def _usage_cols(vectors: pd.DataFrame) -> list[str]:
    return [c for c in vectors.columns if c.startswith("usage_")]


def find_comps(vectors: pd.DataFrame, pitcher: int, year: int, top_n: int = 10) -> pd.DataFrame:
    """Cosine-similarity nearest neighbors within the query pitcher's own hand."""
    query = vectors[(vectors["pitcher"] == pitcher) & (vectors["year"] == year)]
    if query.empty:
        raise ValueError(f"No usage vector for pitcher={pitcher}, year={year}")
    hand = query.iloc[0]["hand"]
    usage_cols = _usage_cols(vectors)

    hand_vectors = vectors[vectors["hand"] == hand].reset_index(drop=True)
    sims = cosine_similarity(query[usage_cols].to_numpy(), hand_vectors[usage_cols].to_numpy())[0]
    result = hand_vectors.assign(similarity=sims)
    result = result[~((result["pitcher"] == pitcher) & (result["year"] == year))]
    return result.sort_values("similarity", ascending=False).head(top_n)[
        ["pitcher", "player_name", "year", "similarity", *usage_cols]
    ]


def fit_archetypes(vectors: pd.DataFrame, n_archetypes: int = N_ARCHETYPES_DEFAULT, seed: int = 42) -> tuple[KMeans, np.ndarray]:
    """KMeans over raw usage vectors (no StandardScaler — dims already share
    a [0,1]/simplex scale; standardizing would over-weight rarely-used
    clusters, the opposite of what we want)."""
    X = vectors[_usage_cols(vectors)].to_numpy()
    km = KMeans(n_clusters=n_archetypes, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    return km, labels


def archetype_summary(vectors: pd.DataFrame, km: KMeans, labels: np.ndarray) -> pd.DataFrame:
    """One row per archetype: centroid usage, pitcher-season count, example pitchers."""
    usage_cols = _usage_cols(vectors)
    tagged = vectors.assign(archetype=labels)

    rows = []
    for a in range(km.n_clusters):
        sub = tagged[tagged["archetype"] == a]
        centroid = km.cluster_centers_[a]
        dist = np.linalg.norm(sub[usage_cols].to_numpy() - centroid, axis=1)
        # drop_duplicates keeps distinct pitchers even if several of their own
        # seasons land closest to this archetype's centroid
        examples = (
            sub.assign(_dist=dist).sort_values("_dist")
            .drop_duplicates(subset="player_name").head(3)["player_name"].tolist()
        )
        row = {"archetype": a, "n_pitcher_seasons": len(sub), "examples": ", ".join(examples)}
        row.update(dict(zip(usage_cols, centroid)))
        rows.append(row)
    return pd.DataFrame(rows)


def arsenal_drift(vectors: pd.DataFrame, pitcher: int) -> pd.DataFrame:
    """This pitcher's usage vectors across all qualifying years, sorted by year."""
    return vectors[vectors["pitcher"] == pitcher].sort_values("year").reset_index(drop=True)


def _resolve_pitcher(vectors: pd.DataFrame, name_query: str) -> tuple[int, str] | None:
    matches = vectors[vectors["player_name"].str.contains(name_query, case=False, na=False, regex=False)]
    if matches.empty:
        print(f"No pitcher matching {name_query!r} found.")
        return None

    unique = matches[["pitcher", "player_name", "hand"]].drop_duplicates(subset=["pitcher"])
    if len(unique) > 1:
        print(f"Multiple pitchers match {name_query!r}, be more specific:")
        for _, row in unique.iterrows():
            print(f"  {row['player_name']} (id={row['pitcher']}, {row['hand']}HP)")
        return None

    row = unique.iloc[0]
    return int(row["pitcher"]), str(row["hand"])


def main():
    parser = argparse.ArgumentParser(description="Pitcher arsenal fingerprints: comps, archetypes, drift")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument("--min-pitches", type=int, default=MIN_PITCHES_DEFAULT)
    parser.add_argument("--n-archetypes", type=int, default=N_ARCHETYPES_DEFAULT)
    parser.add_argument("--comps", type=str, default=None, help="Find comps for a pitcher (name substring match)")
    parser.add_argument("--comps-year", type=int, default=None, help="Season to use for --comps (default: pitcher's most recent qualifying year)")
    parser.add_argument("--drift", type=str, default=None, help="Show a pitcher's usage vector across years (name substring match)")
    parser.add_argument("--rebuild-counts", action="store_true")
    args = parser.parse_args()

    counts = load_or_build_counts(args.years, rebuild=args.rebuild_counts)
    print(f"Loaded {len(counts):,} count rows for {counts['pitcher'].nunique():,} pitchers")

    all_vectors = []
    for hand in ("R", "L"):
        vectors = build_usage_vectors(counts, hand, min_pitches=args.min_pitches)
        print(f"\n{HAND_LABEL[hand]}: {len(vectors):,} qualifying pitcher-seasons "
              f"(>= {args.min_pitches} pitches), {vectors['pitcher'].nunique():,} unique pitchers")

        X = vectors[_usage_cols(vectors)].to_numpy()
        print("  Silhouette score by K (diagnostic — not auto-selected):")
        for k in range(3, 11):
            scan_labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
            print(f"    K={k}: {silhouette_score(X, scan_labels):.3f}")

        km, labels = fit_archetypes(vectors, n_archetypes=args.n_archetypes)
        summary = archetype_summary(vectors, km, labels)
        print(f"\n  Archetypes (K={args.n_archetypes}):")
        for _, row in summary.iterrows():
            usage_str = ", ".join(f"C{i}={row[f'usage_{i}']:.2f}" for i in range(len(_usage_cols(vectors))))
            print(f"    A{int(row['archetype'])} (n={int(row['n_pitcher_seasons'])}): {usage_str}")
            print(f"        e.g. {row['examples']}")

        all_vectors.append(vectors)

    combined = pd.concat(all_vectors, ignore_index=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(VECTORS_PATH, index=False)
    print(f"\nsaved {VECTORS_PATH} ({len(combined):,} rows)")

    if args.comps:
        resolved = _resolve_pitcher(combined, args.comps)
        if resolved:
            pitcher_id, hand = resolved
            hand_vectors = combined[combined["hand"] == hand]
            year = args.comps_year or int(hand_vectors.loc[hand_vectors["pitcher"] == pitcher_id, "year"].max())
            comps = find_comps(combined, pitcher_id, year, top_n=10)
            print(f"\nComps for pitcher_id={pitcher_id}, year={year}:")
            for _, row in comps.iterrows():
                print(f"  {row['player_name']} ({int(row['year'])}): similarity={row['similarity']:.3f}")

    if args.drift:
        resolved = _resolve_pitcher(combined, args.drift)
        if resolved:
            pitcher_id, _ = resolved
            drift = arsenal_drift(combined, pitcher_id)
            usage_cols = _usage_cols(combined)
            print(f"\nArsenal drift for pitcher_id={pitcher_id}:")
            for _, row in drift.iterrows():
                usage_str = ", ".join(f"C{i}={row[c]:.2f}" for i, c in enumerate(usage_cols))
                print(f"  {int(row['year'])} (n={int(row['n_pitches'])}): {usage_str}")


if __name__ == "__main__":
    main()
