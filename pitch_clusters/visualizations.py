"""Visualize how the fitted BayesianGMM pitch clusters carve up feature space.

Loads the persisted RHP/LHP models (data/models/pitch_clusters/*.joblib,
cluster_meta.json) plus a light sample of real pitches, and renders five PNGs
to plots/:

    01_movement.png         pfx_x vs pfx_z scatter + GMM covariance ellipses
    02_velocity_spin.png    velocity vs spin rate scatter + GMM ellipses
    03_usage.png            cluster usage (%) bar chart
    04_profile_heatmap.png  standardized cluster feature profiles (diverging)
    05_whiff_csw.png        whiff rate vs CSW rate bubble chart, sized by usage

Every chart uses the same cluster-ID -> color mapping, so e.g. "cluster 5" is
the same hue in every panel and across both hands. That's a meaningful signal,
not just decoration: pitch_clusters.fit mirrors pfx_x for LHP specifically so
that cluster N means the same pitch shape for both hands.

Usage:
    python -m pitch_clusters.visualizations [--years 2021 2022 2023 2024]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from pitch_clusters.assign import get_assigner
from pitch_clusters.fit import CLUSTER_FEATURES, OUTPUT_DIR as MODEL_DIR, PROCESSED_DIR

_REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = _REPO_ROOT / "plots"

_PFX_X_IDX = CLUSTER_FEATURES.index("pfx_x")

# Fixed categorical palette (dataviz skill reference default, light mode) —
# one hue per cluster slot 0-7, held constant across every chart.
CLUSTER_COLORS = [
    "#2a78d6",  # 0 blue
    "#1baf7a",  # 1 aqua
    "#eda100",  # 2 yellow
    "#008300",  # 3 green
    "#4a3aa7",  # 4 violet
    "#e34948",  # 5 red
    "#e87ba4",  # 6 magenta
    "#eb6834",  # 7 orange
]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"
DIVERGING_NEG = "#2a78d6"  # blue: below the hand's population mean
DIVERGING_POS = "#e34948"  # red: above the hand's population mean

HANDS = ("R", "L")
HAND_LABELS = {"R": "RHP", "L": "LHP"}
HAND_META_KEY = {"R": "rhp", "L": "lhp"}


def _style_axes(ax, grid: bool = True) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def load_models() -> tuple[dict[str, tuple], dict]:
    """Load the fitted (gmm, scaler) pair per hand, plus cluster metadata."""
    models = {}
    for hand, fname in [("R", "rhp_gmm.joblib"), ("L", "lhp_gmm.joblib")]:
        data = joblib.load(MODEL_DIR / fname)
        models[hand] = (data["gmm"], data["scaler"])
    with open(MODEL_DIR / "cluster_meta.json") as f:
        meta = json.load(f)
    return models, meta


def load_pitch_sample(years: list[int], per_year: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Load a light, representative sample of real pitches for the scatter plots.

    Reads only the columns needed (fast columnar parquet read) and samples per
    year before concatenating, instead of loading the full multi-million-row
    dataset just to subsample it afterward.
    """
    cols = ["p_throws", *CLUSTER_FEATURES]
    frames = []
    for y in years:
        path = PROCESSED_DIR / f"statcast_pitches_{y}.parquet"
        if not path.exists():
            print(f"  [WARN] no processed data for {y}, skipping")
            continue
        available = pq.ParquetFile(path).schema.names
        df = pd.read_parquet(path, columns=[c for c in cols if c in available])
        df = df.dropna(subset=CLUSTER_FEATURES)
        if len(df) > per_year:
            df = df.sample(per_year, random_state=seed)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No processed data found for years {years}")
    return pd.concat(frames, ignore_index=True)


def _unmirror_component(
    mean_scaled: np.ndarray, cov_scaled: np.ndarray, scaler, hand: str
) -> tuple[np.ndarray, np.ndarray]:
    """Map a fitted GMM component from scaled (and, for LHP, pfx_x-mirrored)
    space back to real physical units, so it overlays correctly on a scatter
    of true (unmirrored) pitch data.
    """
    mean = mean_scaled * scaler.scale_ + scaler.mean_
    cov = cov_scaled * np.outer(scaler.scale_, scaler.scale_)
    if hand == "L":
        mean = mean.copy()
        cov = cov.copy()
        mean[_PFX_X_IDX] *= -1
        cov[_PFX_X_IDX, :] = -cov[_PFX_X_IDX, :]
        cov[:, _PFX_X_IDX] = -cov[:, _PFX_X_IDX]
    return mean, cov


def _ellipse_2d(mean: np.ndarray, cov: np.ndarray, i: int, j: int, confidence: float = 0.95):
    """2D confidence-ellipse params (center, width, height, angle) for a pair
    of feature indices, marginalized out of a full-dimensional Gaussian.
    """
    sub_mean = mean[[i, j]]
    sub_cov = cov[np.ix_([i, j], [i, j])]
    eigvals, eigvecs = np.linalg.eigh(sub_cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    scale = np.sqrt(chi2.ppf(confidence, df=2))
    width, height = 2 * scale * np.sqrt(np.clip(eigvals, 0, None))
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return sub_mean, width, height, angle


def _active_clusters(meta_hand: dict) -> list[int]:
    """Cluster indices for this hand, most-used first, dropping components
    the fit pruned to ~zero weight (matches fit.py's 1% effective threshold).
    """
    weights = meta_hand["weights"]
    active = [c for c, w in enumerate(weights) if w > 0.01]
    return sorted(active, key=lambda c: -weights[c])


def _cluster_tag(c: int, meta_hand: dict) -> str:
    tag = " (FB)" if c in meta_hand["fastball_clusters"] else ""
    return f"C{c}{tag}"


def _scatter_with_ellipses(
    fig_title: str,
    feat_i: str,
    feat_j: str,
    xlabel: str,
    ylabel: str,
    models: dict,
    meta: dict,
    labeled: pd.DataFrame,
    out_path: Path,
) -> None:
    i, j = CLUSTER_FEATURES.index(feat_i), CLUSTER_FEATURES.index(feat_j)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), facecolor=SURFACE)

    for ax, hand in zip(axes, HANDS):
        gmm, scaler = models[hand]
        meta_hand = meta[HAND_META_KEY[hand]]
        hand_df = labeled[labeled["p_throws"] == hand]
        active = _active_clusters(meta_hand)

        for c in active:
            color = CLUSTER_COLORS[c]
            pts = hand_df[hand_df["cluster"] == c]
            ax.scatter(
                pts[feat_i], pts[feat_j], s=10, alpha=0.35, color=color,
                linewidths=0, zorder=2, label=_cluster_tag(c, meta_hand),
            )
            mean, cov = _unmirror_component(gmm.means_[c], gmm.covariances_[c], scaler, hand)
            center, width, height, angle = _ellipse_2d(mean, cov, i, j)
            ax.add_patch(Ellipse(
                center, width, height, angle=angle,
                facecolor=color, alpha=0.10, edgecolor=color, linewidth=2, zorder=3,
            ))
            ax.annotate(
                str(c), center, ha="center", va="center", fontsize=9,
                fontweight="bold", color=TEXT_PRIMARY, zorder=4,
            )

        _style_axes(ax)
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
        ax.set_title(HAND_LABELS[hand], color=TEXT_PRIMARY, fontsize=12, loc="left")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2)

    fig.suptitle(fig_title, color=TEXT_PRIMARY, fontsize=14, x=0.02, ha="left", y=1.01)
    fig.text(
        0.02, -0.02,
        "Ellipses: 95% confidence region of each fitted Gaussian component (numbered at its center).",
        color=TEXT_MUTED, fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_movement(models, meta, labeled: pd.DataFrame) -> None:
    _scatter_with_ellipses(
        "Pitch movement clusters", "pfx_x", "pfx_z",
        "Horizontal break (in, catcher's view)", "Vertical break (in)",
        models, meta, labeled, PLOTS_DIR / "01_movement.png",
    )


def plot_velocity_spin(models, meta, labeled: pd.DataFrame) -> None:
    _scatter_with_ellipses(
        "Velocity / spin clusters", "release_speed", "release_spin_rate",
        "Release velocity (mph)", "Spin rate (rpm)",
        models, meta, labeled, PLOTS_DIR / "02_velocity_spin.png",
    )


def plot_usage(meta: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE)
    for ax, hand in zip(axes, HANDS):
        meta_hand = meta[HAND_META_KEY[hand]]
        order = _active_clusters(meta_hand)
        weights = [meta_hand["weights"][c] * 100 for c in order]
        colors = [CLUSTER_COLORS[c] for c in order]
        labels = [_cluster_tag(c, meta_hand) for c in order]

        bars = ax.bar(range(len(order)), weights, color=colors, width=0.62, zorder=2)
        for bar, w in zip(bars, weights):
            ax.annotate(
                f"{w:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY,
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Usage (%)", color=TEXT_SECONDARY, fontsize=10)
        ax.set_title(HAND_LABELS[hand], color=TEXT_PRIMARY, fontsize=12, loc="left")
        ax.set_ylim(0, max(weights) * 1.18)
        _style_axes(ax)
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
        ax.grid(False, axis="x")

    fig.suptitle("Cluster usage share", color=TEXT_PRIMARY, fontsize=14, x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    out_path = PLOTS_DIR / "03_usage.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_profile_heatmap(models: dict, meta: dict) -> None:
    """Standardized cluster feature profiles (z-scores in the model's own
    fitting space — pfx_x is mirrored for LHP here, matching how the model
    was fit, since the point of this chart is showing what the model "sees").
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 6.5), facecolor=SURFACE)
    vmax = 3.0  # clip z-scores at +/-3 for a stable, comparable color scale

    for ax, hand in zip(axes, HANDS):
        gmm, _ = models[hand]
        meta_hand = meta[HAND_META_KEY[hand]]
        order = _active_clusters(meta_hand)

        # gmm.means_ is already in standardized (z-score) space by construction
        # (StandardScaler was fit on this hand's own population).
        grid = np.array([gmm.means_[c] for c in order])
        grid = np.clip(grid, -vmax, vmax)

        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for r, c in enumerate(order):
            for col in range(len(CLUSTER_FEATURES)):
                ax.text(
                    col, r, f"{grid[r, col]:.1f}", ha="center", va="center",
                    fontsize=8, color=TEXT_PRIMARY,
                )
        ax.set_xticks(range(len(CLUSTER_FEATURES)))
        ax.set_xticklabels(CLUSTER_FEATURES, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([_cluster_tag(c, meta_hand) for c in order], fontsize=9)
        ax.set_title(HAND_LABELS[hand], color=TEXT_PRIMARY, fontsize=12, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Cluster feature profiles (z-score vs. hand's population)",
        color=TEXT_PRIMARY, fontsize=13, x=0.02, ha="left", y=1.03,
    )
    fig.text(
        0.02, -0.03,
        "pfx_x is mirrored for LHP (model-fitting convention) so it lines up with the RHP row above.",
        color=TEXT_MUTED, fontsize=8.5,
    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cbar.set_label("Standard deviations from mean", color=TEXT_SECONDARY, fontsize=9)

    out_path = PLOTS_DIR / "04_profile_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_whiff_csw(meta: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), facecolor=SURFACE)
    for ax, hand in zip(axes, HANDS):
        meta_hand = meta[HAND_META_KEY[hand]]
        order = _active_clusters(meta_hand)

        xs = [meta_hand["league_avgs"][str(c)]["whiff_rate"] * 100 for c in order]
        ys = [meta_hand["league_avgs"][str(c)]["csw_rate"] * 100 for c in order]

        for c in order:
            a = meta_hand["league_avgs"][str(c)]
            color = CLUSTER_COLORS[c]
            size = 300 + 4000 * a["usage_pct"]
            x, y = a["whiff_rate"] * 100, a["csw_rate"] * 100
            ax.scatter(
                x, y, s=size, color=color,
                alpha=0.55, edgecolors=SURFACE, linewidths=2, zorder=2,
            )
            label = f"{c}*" if c in meta_hand["fastball_clusters"] else str(c)
            ax.annotate(
                label, (x, y), ha="center", va="center", fontsize=9,
                fontweight="bold", color=TEXT_PRIMARY, zorder=3,
            )

        # Every bubble is already directly labeled with its cluster ID, so a
        # color legend here would be pure redundancy (and, with bubbles this
        # close to the data's own corners, would overlap them) — colors still
        # match the same cluster IDs used in the other charts.
        x_pad = (max(xs) - min(xs)) * 0.18 or 1.0
        y_pad = (max(ys) - min(ys)) * 0.18 or 1.0
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

        _style_axes(ax)
        ax.set_xlabel("Whiff rate (%)", color=TEXT_SECONDARY, fontsize=10)
        ax.set_ylabel("CSW rate (%)", color=TEXT_SECONDARY, fontsize=10)
        ax.set_title(HAND_LABELS[hand], color=TEXT_PRIMARY, fontsize=12, loc="left")

    fig.suptitle(
        "Whiff vs. CSW rate by cluster  (bubble size = usage share)",
        color=TEXT_PRIMARY, fontsize=13, x=0.02, ha="left", y=1.02,
    )
    fig.text(0.02, -0.02, "* = fastball cluster (by mean velocity)", color=TEXT_MUTED, fontsize=8.5)
    fig.tight_layout()
    out_path = PLOTS_DIR / "05_whiff_csw.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize fitted pitch clusters")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument("--per-year-sample", type=int, default=4000,
                         help="Rows sampled per year for the scatter plots")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models and metadata...")
    models, meta = load_models()

    print(f"Loading a {args.per_year_sample}/year pitch sample for {args.years}...")
    sample = load_pitch_sample(args.years, per_year=args.per_year_sample)
    print(f"  {len(sample):,} pitches sampled")

    print("Assigning clusters to the sample...")
    assigner = get_assigner()
    sample = sample.assign(cluster=assigner.assign(sample))

    print("Rendering charts...")
    plot_movement(models, meta, sample)
    plot_velocity_spin(models, meta, sample)
    plot_usage(meta)
    plot_profile_heatmap(models, meta)
    plot_whiff_csw(meta)

    print(f"\nAll charts saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
