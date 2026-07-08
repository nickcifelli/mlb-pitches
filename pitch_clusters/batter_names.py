"""Batter id -> name lookup via pybaseball's Chadwick register.

Statcast pitch-level exports carry the pitcher's own player_name column but
never the batter's — only the batter's MLBAM id. This resolves ids to
"First Last" names via pybaseball.playerid_reverse_lookup, caching results to
data/derived/batter_names.parquet so repeat runs only fetch ids not already
resolved.

Usage:
    names = get_batter_names([605141, 660271])
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
NAMES_PATH = _REPO_ROOT / "data" / "derived" / "batter_names.parquet"


def get_batter_names(batter_ids: list[int]) -> pd.DataFrame:
    """Return a (batter, player_name) frame for the given MLBAM ids.

    Ids already cached in NAMES_PATH are served without a network call; any
    new ids are looked up in one batch and merged into the cache.
    """
    ids = sorted({int(i) for i in batter_ids})
    cached = (
        pd.read_parquet(NAMES_PATH) if NAMES_PATH.exists()
        else pd.DataFrame(columns=["batter", "player_name"])
    )

    missing = sorted(set(ids) - set(cached["batter"]))
    if missing:
        from pybaseball import playerid_reverse_lookup

        print(f"  Looking up names for {len(missing):,} new batter ids...", flush=True)
        fetched = playerid_reverse_lookup(missing, key_type="mlbam")
        resolved = pd.DataFrame({
            "batter": fetched["key_mlbam"].astype(int),
            "player_name": fetched["name_first"].str.title() + " " + fetched["name_last"].str.title(),
        })
        cached = pd.concat([cached, resolved], ignore_index=True).drop_duplicates(subset="batter")
        NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(NAMES_PATH, index=False)

    return cached[cached["batter"].isin(ids)].reset_index(drop=True)
