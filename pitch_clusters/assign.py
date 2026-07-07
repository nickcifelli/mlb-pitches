"""Unsupervised pitch-type cluster assignment from physical characteristics.

Loads pre-fit BayesianGaussianMixture models (one per handedness) and assigns
each pitch to one of 8 archetype clusters based on movement, velocity, spin
rate, and spin axis. Replaces brittle categorical pitch_type labels that break
when StatCast reclassifies pitch types (e.g. adding "Sweeper").

pfx_x is mirrored (negated) for LHP before scaling/prediction, matching how
the models were fit in pitch_clusters.fit — this keeps cluster index N
meaning the same pitch shape for both hands.

Usage:
    assigner = get_assigner()
    clusters = assigner.assign(pitch_df)  # pd.Series of int cluster labels
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CLUSTER_FEATURES = ["pfx_x", "pfx_z", "release_speed", "release_spin_rate", "spin_axis"]
_PFX_X_IDX = CLUSTER_FEATURES.index("pfx_x")

_MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "pitch_clusters"

_DEFAULT_CLUSTER_AVGS: dict[str, float] = {
    "velo": 93.0,
    "spin": 2200.0,
    "pfx_x": 0.0,
    "pfx_z": 5.0,
    "usage_pct": 0.0,
    "whiff_rate": 0.22,
    "csw_rate": 0.28,
    "zone_pct": 0.45,
    "avg_ev_against": 87.0,
    "gb_rate": 0.43,
}


class PitchClusterAssigner:
    """Assign pitches to unsupervised archetype clusters.

    Loads pre-fit BayesianGMM models and StandardScalers for RHP and LHP.
    Provides vectorized cluster assignment from pitch DataFrames.
    """

    def __init__(self, model_dir: str | Path | None = None):
        model_dir = Path(model_dir) if model_dir else _MODEL_DIR

        rhp_path = model_dir / "rhp_gmm.joblib"
        if rhp_path.exists():
            data = joblib.load(rhp_path)
            self._rhp_gmm = data["gmm"]
            self._rhp_scaler = data["scaler"]
        else:
            self._rhp_gmm = None
            self._rhp_scaler = None

        lhp_path = model_dir / "lhp_gmm.joblib"
        if lhp_path.exists():
            data = joblib.load(lhp_path)
            self._lhp_gmm = data["gmm"]
            self._lhp_scaler = data["scaler"]
        else:
            self._lhp_gmm = None
            self._lhp_scaler = None

        meta_path = model_dir / "cluster_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._meta = {}

        self._league_avgs: dict[str, dict[int, dict[str, float]]] = {}
        for hand_key in ("rhp", "lhp"):
            hand_meta = self._meta.get(hand_key, {})
            raw_avgs = hand_meta.get("league_avgs", {})
            self._league_avgs[hand_key] = {int(k): v for k, v in raw_avgs.items()}

    @property
    def available(self) -> bool:
        return self._rhp_gmm is not None and self._lhp_gmm is not None

    def assign(self, pitch_df: pd.DataFrame) -> pd.Series:
        """Assign cluster labels (0..n_components-1) to each pitch.

        Missing/invalid features -> label -1. -1 is used specifically because
        it can never collide with a real predicted cluster id (0-indexed, so
        the largest valid label is n_components-1) — a positional sentinel
        like the old n_components-1 would silently merge "missing data" into
        whatever the last real archetype happens to be.
        """
        if pitch_df.empty:
            return pd.Series(dtype=int)

        result = pd.Series(-1, index=pitch_df.index, dtype=int)

        missing = [c for c in CLUSTER_FEATURES if c not in pitch_df.columns]
        if missing:
            raise ValueError(f"Missing clustering columns: {missing}")
        if "p_throws" not in pitch_df.columns:
            raise ValueError("Column 'p_throws' required for handedness routing")

        for hand, gmm, scaler in [
            ("R", self._rhp_gmm, self._rhp_scaler),
            ("L", self._lhp_gmm, self._lhp_scaler),
        ]:
            if gmm is None or scaler is None:
                continue
            mask = pitch_df["p_throws"] == hand
            if not mask.any():
                continue
            subset = pitch_df.loc[mask, CLUSTER_FEATURES]
            valid = subset.notna().all(axis=1)
            if not valid.any():
                continue
            X = subset.loc[valid].to_numpy(dtype=np.float64)
            if hand == "L":
                X[:, _PFX_X_IDX] *= -1
            labels = gmm.predict(scaler.transform(X))
            result.loc[valid.index[valid]] = labels

        return result

    def assign_single_hand(self, pitch_df: pd.DataFrame, hand: str) -> pd.Series:
        """Assign cluster labels when all pitches are from one handedness."""
        if pitch_df.empty:
            return pd.Series(dtype=int)

        gmm = self._rhp_gmm if hand == "R" else self._lhp_gmm
        scaler = self._rhp_scaler if hand == "R" else self._lhp_scaler

        result = pd.Series(-1, index=pitch_df.index, dtype=int)
        if gmm is None or scaler is None:
            return result

        subset = pitch_df[CLUSTER_FEATURES]
        valid = subset.notna().all(axis=1)
        if not valid.any():
            return result

        X = subset.loc[valid].to_numpy(dtype=np.float64)
        if hand == "L":
            X[:, _PFX_X_IDX] *= -1
        labels = gmm.predict(scaler.transform(X))
        result.loc[valid.index[valid]] = labels
        return result

    def fastball_clusters(self, hand: str) -> set[int]:
        """Cluster indices identified as fastball-type (top 2 by mean velocity)."""
        hand_key = "rhp" if hand == "R" else "lhp"
        return set(self._meta.get(hand_key, {}).get("fastball_clusters", [0, 1]))

    def breaking_clusters(self, hand: str) -> set[int]:
        """Cluster indices identified as breaking-ball type (spin rate > 2100 RPM).

        The 450 RPM gap between breaking balls (~2400-2600) and offspeed (~1700-1950)
        is the most stable physical boundary in pitch classification.
        """
        fb = self.fastball_clusters(hand)
        return {
            c for c, avgs in self.get_all_league_avgs(hand).items()
            if c not in fb and avgs.get("spin", 0) > 2100
        }

    def classify_pitches(self, pitch_df: pd.DataFrame) -> pd.DataFrame:
        """Return boolean columns is_fastball, is_breaking, is_offspeed per pitch."""
        clusters = self.assign(pitch_df)

        if "p_throws" not in pitch_df.columns:
            return pd.DataFrame(
                {"is_fastball": False, "is_breaking": False, "is_offspeed": True},
                index=pitch_df.index,
            )

        r_mask = pitch_df["p_throws"] == "R"
        l_mask = ~r_mask

        is_fb = (r_mask & clusters.isin(self.fastball_clusters("R"))) | \
                (l_mask & clusters.isin(self.fastball_clusters("L")))
        is_brk = (r_mask & clusters.isin(self.breaking_clusters("R"))) | \
                 (l_mask & clusters.isin(self.breaking_clusters("L")))

        return pd.DataFrame(
            {"is_fastball": is_fb, "is_breaking": is_brk, "is_offspeed": ~is_fb & ~is_brk},
            index=pitch_df.index,
        )

    def get_league_avgs(self, cluster_idx: int, hand: str) -> dict[str, float]:
        """Per-cluster league-average statistics for use as shrinkage priors."""
        hand_key = "rhp" if hand == "R" else "lhp"
        return self._league_avgs.get(hand_key, {}).get(cluster_idx, _DEFAULT_CLUSTER_AVGS.copy())

    def get_all_league_avgs(self, hand: str) -> dict[int, dict[str, float]]:
        hand_key = "rhp" if hand == "R" else "lhp"
        return self._league_avgs.get(hand_key, {})

    def n_components(self) -> int:
        return self._meta.get("n_components", 8)


_default_assigner: PitchClusterAssigner | None = None


def get_assigner(model_dir: str | Path | None = None) -> PitchClusterAssigner:
    """Get or create the module-level singleton PitchClusterAssigner."""
    global _default_assigner
    if _default_assigner is None:
        _default_assigner = PitchClusterAssigner(model_dir)
    return _default_assigner


def reset_assigner() -> None:
    """Reset the singleton (for testing)."""
    global _default_assigner
    _default_assigner = None
