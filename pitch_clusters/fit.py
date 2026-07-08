"""Fit unsupervised pitch-type clusters from Statcast physical characteristics.

Fits separate BayesianGaussianMixture models for RHP and LHP, each with up to
8 components. pfx_x is mirrored (negated) for LHP before fitting/prediction so
cluster index N represents the same pitch shape for both hands (reported
league averages still use the true, unmirrored pfx_x). The Dirichlet process
prior automatically prunes unused components. Persists fitted models,
scalers, and cluster metadata to data/models/pitch_clusters/.

Each hand is fit with several random restarts (--n-init), run concurrently as
separate processes rather than sklearn's internal sequential n_init loop, and
the restart with the best variational lower bound is kept per hand.

Usage:
    python -m pitch_clusters.fit [--years 2021 2022 2023 2024]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Cap BLAS threads per process before numpy/sklearn are imported: with several
# GMM restarts running concurrently across processes, unrestricted BLAS
# threading would oversubscribe the machine's cores instead of speeding
# things up.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["pfx_x", "pfx_z", "release_speed", "release_spin_rate", "spin_axis"]
_PFX_X_IDX = CLUSTER_FEATURES.index("pfx_x")

_REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
OUTPUT_DIR = _REPO_ROOT / "data" / "models" / "pitch_clusters"

_SWING_DESCRIPTIONS = frozenset({
    "swinging_strike", "swinging_strike_blocked", "foul_tip", "foul",
    "hit_into_play", "foul_bunt", "missed_bunt", "hit_into_play_no_out",
    "hit_into_play_score",
})
_WHIFF_DESCRIPTIONS = frozenset({
    "swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt",
})
_ZONE_NUMS = list(range(1, 10))


def load_pitch_data(years: list[int]) -> pd.DataFrame:
    """Load raw pitch-level Statcast data for the given years.

    Prefers parquet (faster); falls back to CSV if parquet is absent. This is
    the from-scratch entrypoint (used only by this module, before anything
    else has run), so unlike load_processed_years below it also handles the
    CSV fallback.
    """
    cols_needed = [
        "pitcher", "p_throws", "pitch_type",
        *CLUSTER_FEATURES,
        "zone", "description", "launch_speed", "bb_type",
    ]
    dfs = []
    for y in years:
        parquet_path = PROCESSED_DIR / f"statcast_pitches_{y}.parquet"
        csv_path = PROCESSED_DIR / f"statcast_pitches_{y}.csv"

        if parquet_path.exists():
            print(f"  Loading {parquet_path.name}...", flush=True)
            # Read the schema only (no row data) to see which columns exist —
            # pd.read_parquet(path, columns=[]) looks like it'd do this but
            # actually returns a 0-row DataFrame, silently dropping all data.
            available = pq.ParquetFile(parquet_path).schema.names
            df = pd.read_parquet(parquet_path, columns=[c for c in cols_needed if c in available])
        elif csv_path.exists():
            print(f"  Loading {csv_path.name} (CSV)...", flush=True)
            available = pd.read_csv(csv_path, nrows=0).columns.tolist()
            df = pd.read_csv(csv_path, usecols=[c for c in cols_needed if c in available], low_memory=False)
        else:
            print(f"  [WARN] No data found for {y}, skipping")
            continue

        dfs.append(df)
        print(f"    -> {len(df):,} pitches", flush=True)

    if not dfs:
        raise RuntimeError(f"No pitch data found in {PROCESSED_DIR} for years {years}")
    return pd.concat(dfs, ignore_index=True)


def load_processed_years(years: list[int], columns: list[str]) -> pd.DataFrame:
    """Load cached processed-season parquet files, restricted to whichever of
    `columns` are present in each file's schema.

    Shared by every downstream module (pitcher_cluster_stats, batter_cluster_stats,
    ...) that needs a full, unsampled pass over the pitch data with its own
    column subset — the "read schema first, don't silently drop rows" trick
    used to live duplicated in each of those modules.
    """
    frames = []
    for y in years:
        path = PROCESSED_DIR / f"statcast_pitches_{y}.parquet"
        if not path.exists():
            print(f"  [WARN] no processed data for {y}, skipping")
            continue
        available = pq.ParquetFile(path).schema.names
        df = pd.read_parquet(path, columns=[c for c in columns if c in available])
        print(f"  Loaded {path.name}: {len(df):,} pitches", flush=True)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No processed data found for years {years}")
    return pd.concat(frames, ignore_index=True)


def _default_avgs() -> dict[str, float]:
    return {
        "velo": 93.0, "spin": 2200.0, "pfx_x": 0.0, "pfx_z": 5.0,
        "usage_pct": 0.0, "whiff_rate": 0.22, "csw_rate": 0.28,
        "zone_pct": 0.45, "avg_ev_against": 87.0, "gb_rate": 0.43,
    }


def compute_cluster_league_avgs(
    df: pd.DataFrame, cluster_labels: np.ndarray, n_clusters: int
) -> dict[int, dict[str, float]]:
    """Compute per-cluster league-average statistics for shrinkage priors.

    Uses numpy array masking instead of a DataFrame copy+groupby to avoid
    allocating a full copy of a multi-million-row DataFrame. `df`'s pfx_x is
    expected to be the true, unmirrored value (not the fitting-time mirrored
    version) so reported averages read correctly regardless of hand.
    """
    total = len(df)

    # Extract all needed arrays once — avoids repeated pandas overhead per cluster
    velo = df["release_speed"].to_numpy(dtype=np.float64) if "release_speed" in df.columns else None
    spin = df["release_spin_rate"].to_numpy(dtype=np.float64) if "release_spin_rate" in df.columns else None
    pfx_x = df["pfx_x"].to_numpy(dtype=np.float64) if "pfx_x" in df.columns else None
    pfx_z = df["pfx_z"].to_numpy(dtype=np.float64) if "pfx_z" in df.columns else None
    ev = df["launch_speed"].to_numpy(dtype=np.float64) if "launch_speed" in df.columns else None

    is_swing: np.ndarray | None
    is_whiff: np.ndarray | None
    is_cs: np.ndarray | None
    if "description" in df.columns:
        desc = df["description"].to_numpy()
        is_swing = np.isin(desc, list(_SWING_DESCRIPTIONS))
        is_whiff = np.isin(desc, list(_WHIFF_DESCRIPTIONS))
        is_cs = desc == "called_strike"
    else:
        is_swing = is_whiff = is_cs = None

    in_zone: np.ndarray | None
    if "zone" in df.columns:
        in_zone = np.isin(df["zone"].to_numpy(), _ZONE_NUMS)
    else:
        in_zone = None

    is_bip: np.ndarray | None
    is_gb: np.ndarray | None
    if "bb_type" in df.columns:
        bb_type = df["bb_type"].to_numpy()
        is_bip = pd.notna(bb_type)
        is_gb = bb_type == "ground_ball"
    else:
        is_bip = is_gb = None

    avgs: dict[int, dict[str, float]] = {}
    defaults = _default_avgs()

    for c in range(n_clusters):
        mask = cluster_labels == c
        n = int(mask.sum())
        if n == 0:
            avgs[c] = defaults.copy()
            continue

        stats: dict[str, float] = {}
        stats["velo"] = float(np.nanmean(velo[mask])) if velo is not None else 93.0
        stats["spin"] = float(np.nanmean(spin[mask])) if spin is not None else 2200.0
        stats["pfx_x"] = float(np.nanmean(pfx_x[mask])) if pfx_x is not None else 0.0
        stats["pfx_z"] = float(np.nanmean(pfx_z[mask])) if pfx_z is not None else 0.0
        stats["usage_pct"] = n / total if total > 0 else 1.0 / n_clusters

        n_swings = int(is_swing[mask].sum()) if is_swing is not None else 0
        n_whiffs = int(is_whiff[mask].sum()) if is_whiff is not None else 0
        stats["whiff_rate"] = n_whiffs / n_swings if n_swings > 0 else 0.22

        n_cs = int(is_cs[mask].sum()) if is_cs is not None else 0
        stats["csw_rate"] = (n_cs + n_whiffs) / n

        n_zone = int(in_zone[mask].sum()) if in_zone is not None else 0
        stats["zone_pct"] = n_zone / n

        if ev is not None and is_bip is not None:
            bip_ev = ev[mask & is_bip]
            valid_ev = bip_ev[~np.isnan(bip_ev)]
            stats["avg_ev_against"] = float(valid_ev.mean()) if len(valid_ev) > 0 else 87.0
        else:
            stats["avg_ev_against"] = 87.0

        if is_bip is not None and is_gb is not None:
            n_bip = int(is_bip[mask].sum())
            stats["gb_rate"] = int(is_gb[mask].sum()) / n_bip if n_bip > 0 else 0.43
        else:
            stats["gb_rate"] = 0.43

        for k, v in stats.items():
            if isinstance(v, float) and np.isnan(v):
                stats[k] = defaults[k]
        avgs[c] = stats

    return avgs


def _prep_hand(df: pd.DataFrame, hand: str) -> tuple[pd.DataFrame, StandardScaler, np.ndarray]:
    """Drop NaN feature rows and build the scaled fitting matrix for one hand.

    Mirrors (negates) pfx_x for LHP so both hands' feature space shares the
    same geometry; the returned `clean` DataFrame keeps the true, unmirrored
    pfx_x for later league-average reporting.
    """
    clean = df.dropna(subset=CLUSTER_FEATURES)
    n_dropped = len(df) - len(clean)
    print(f"\n{'=' * 60}")
    print(f"Preparing {hand}HP: {len(df):,} pitches")
    if n_dropped:
        print(f"  Dropped {n_dropped:,} pitches with NaN features ({100 * n_dropped / len(df):.1f}%)")
    print(f"  Training on {len(clean):,} pitches")

    X = clean[CLUSTER_FEATURES].to_numpy(dtype=np.float64)
    if hand == "L":
        X[:, _PFX_X_IDX] *= -1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return clean, scaler, X_scaled


def _fit_restart(
    hand: str, X_scaled: np.ndarray, n_components: int, seed: int
) -> tuple[str, BayesianGaussianMixture, float]:
    """Fit one BayesianGMM restart (n_init=1) with a given seed.

    Runs as its own process so multiple restarts execute concurrently instead
    of sklearn's sequential internal n_init loop. The caller picks the
    restart with the highest `lower_bound_` per hand — the same criterion
    sklearn uses internally to select among its own n_init restarts.
    """
    gmm = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1.0 / n_components,
        covariance_type="full",
        n_init=1,
        max_iter=500,
        random_state=seed,
        verbose=0,
    )
    gmm.fit(X_scaled)
    return hand, gmm, float(gmm.lower_bound_)


def _report_hand(
    hand: str,
    clean: pd.DataFrame,
    X_scaled: np.ndarray,
    gmm: BayesianGaussianMixture,
    n_components: int,
) -> dict:
    """Print a cluster profile report and build metadata for a fitted hand."""
    labels = gmm.predict(X_scaled)
    n_effective = int((gmm.weights_ > 0.01).sum())

    print(f"\n{'=' * 60}")
    print(f"{hand}HP results: {len(clean):,} pitches")
    print(f"  Effective components: {n_effective}/{n_components}")
    print(f"  Weights: {np.sort(gmm.weights_)[::-1].round(3).tolist()}")

    for u, c in sorted(zip(*np.unique(labels, return_counts=True)), key=lambda x: -x[1]):
        print(f"    Cluster {u}: {c:>8,} pitches ({100 * c / len(labels):.1f}%)")

    league_avgs = compute_cluster_league_avgs(clean, labels, n_components)

    cluster_velos = [(c, league_avgs[c]["velo"]) for c in range(n_components) if gmm.weights_[c] > 0.01]
    fastball_clusters = [c for c, _ in sorted(cluster_velos, key=lambda x: -x[1])[:2]]
    print(f"  Fastball clusters (by velo): {fastball_clusters}")

    print(f"\n  Cluster profiles ({hand}HP):")
    for c in range(n_components):
        if gmm.weights_[c] < 0.01:
            continue
        a = league_avgs[c]
        tag = " [FB]" if c in fastball_clusters else ""
        print(f"    C{c}{tag}: velo={a['velo']:.1f} spin={a['spin']:.0f} "
              f"pfx_x={a['pfx_x']:.1f} pfx_z={a['pfx_z']:.1f} "
              f"usage={a['usage_pct']:.3f} whiff={a['whiff_rate']:.3f}")

    return {
        "hand": hand,
        "n_components": n_components,
        "n_effective": n_effective,
        "n_pitches": len(clean),
        "fastball_clusters": fastball_clusters,
        "league_avgs": {str(k): v for k, v in league_avgs.items()},
        "weights": gmm.weights_.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit physics-based pitch cluster models")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-init", type=int, default=3,
                        help="GMM random restarts per hand, fit concurrently in separate "
                             "processes (use 1 for a single fast fit per hand during dev)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Fitting pitch clusters for years {args.years}")
    df = load_pitch_data(args.years)
    print(f"\nTotal: {len(df):,} pitches  "
          f"(RHP: {(df['p_throws']=='R').sum():,}  LHP: {(df['p_throws']=='L').sum():,})")

    df_r = df[df["p_throws"] == "R"].reset_index(drop=True)
    df_l = df[df["p_throws"] == "L"].reset_index(drop=True)
    del df  # free memory before spawning workers

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean + scale both hands up front so restarts for R and L can share one
    # flat worker pool below. Nesting a pool inside a per-hand worker would
    # serialize the two hands behind each other instead of running all
    # restarts (both hands) concurrently.
    prepped = {hand: _prep_hand(hdf, hand) for hand, hdf in [("R", df_r), ("L", df_l)]}

    n_restarts = 2 * args.n_init
    n_workers = min(n_restarts, os.cpu_count() or 2)
    print(f"\nFitting {n_restarts} restarts ({args.n_init} per hand) "
          f"across {n_workers} worker processes...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_fit_restart, hand, prepped[hand][2], args.n_components, args.seed + i)
            for hand in ("R", "L")
            for i in range(args.n_init)
        ]
        results = [f.result() for f in futures]
    print(f"All restarts done in {time.time() - t0:.1f}s", flush=True)

    best: dict[str, tuple[BayesianGaussianMixture, float]] = {}
    for hand, gmm, lower_bound in results:
        if hand not in best or lower_bound > best[hand][1]:
            best[hand] = (gmm, lower_bound)

    models: dict[str, tuple[BayesianGaussianMixture, StandardScaler]] = {}
    metas: dict[str, dict] = {}
    for hand in ("R", "L"):
        clean, scaler, X_scaled = prepped[hand]
        gmm, lower_bound = best[hand]
        print(f"  {hand}HP best restart: lower_bound_={lower_bound:.2f}")
        metas[hand] = _report_hand(hand, clean, X_scaled, gmm, args.n_components)
        models[hand] = (gmm, scaler)

    joblib.dump({"gmm": models["R"][0], "scaler": models["R"][1]}, OUTPUT_DIR / "rhp_gmm.joblib")
    joblib.dump({"gmm": models["L"][0], "scaler": models["L"][1]}, OUTPUT_DIR / "lhp_gmm.joblib")

    meta = {
        "cluster_features": CLUSTER_FEATURES,
        "n_components": args.n_components,
        "years": args.years,
        "seed": args.seed,
        "rhp": metas["R"],
        "lhp": metas["L"],
    }
    with open(OUTPUT_DIR / "cluster_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nArtifacts saved to {OUTPUT_DIR}/")
    print("  rhp_gmm.joblib, lhp_gmm.joblib, cluster_meta.json")


if __name__ == "__main__":
    main()
