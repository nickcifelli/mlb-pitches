# mlb-pitches

Unsupervised pitch archetype clustering and pitcher analytics from MLB
Statcast data. Instead of relying on Statcast's own `pitch_type` labels
(they get reclassified retroactively and are heuristic), this
fits Bayesian Gaussian Mixture Models directly on the physical
characteristics of each pitch (movement, velocity, spin rate, spin axis) to
discover pitch archetypes from the data itself, then builds pitcher level
analytics on top: empirical Bayes shrunk outcome rates, pitcher comps via
usage-vector similarity, and archetype clustering of pitchers.

## How it works

- **Clustering**: a separate `BayesianGaussianMixture` (up to 8 components,
  full covariance) is fit per handedness. LHP's horizontal break is mirrored
  before fitting so cluster index `N` means the same pitch shape for both
  hands — a RHP's slider and a LHP's slider land in the same cluster despite
  the sign of horizontal break flipping between hands. The Dirichlet process
  prior lets the model prune unused components rather than forcing exactly 8.
- **Pitcher analytics**: per pitcher, per cluster outcome rates (whiff, CSW,
  zone%, ground-ball%, exit velocity) are shrunk toward the cluster's league
  average via empirical Bayes, with the shrinkage strength estimated from
  the data itself (not a hardcoded constant) so a pitcher's grade on a
  pitch they've thrown 20 times regresses hard toward the league mean, while
  a pitch they've thrown 2,000 times barely moves. Pitcher-seasons are also
  reduced to an 8d cluster usage vector, enabling similarity-based
  "comps" and KMeans-based archetype clustering of pitchers.


## Setup

```
pip install -e .
```

Requires Python >= 3.11. See `pyproject.toml` for dependencies.

## Pipeline

Each stage reads what the previous stage wrote to `data/`. All of `data/`
except `data/models/pitch_clusters/cluster_meta.json` is gitignored — run the
pipeline yourself to regenerate it.

| Step | Command | What it does |
|---|---|---|
| 1 | `python -m pitch_clusters.statcast --years 2021 2022 2023 2024` | Fetches Statcast pitch data via `pybaseball`, caches to `data/raw/` and `data/processed/` |
| 2 | `python -m pitch_clusters.fit --years 2021 2022 2023 2024` | Fits the RHP/LHP cluster models, saves to `data/models/pitch_clusters/` |
| 3 | `python -m pitch_clusters.visualizations` | Renders cluster diagnostics (movement plots, usage, feature profiles) to `plots/` |
| 4 | `python -m pitch_clusters.pitcher_cluster_stats` | Aggregates per-pitcher, per-cluster pitch/outcome counts to `data/derived/` |
| 5 | `python -m pitch_clusters.shrinkage --pitcher "name"` | Empirical-Bayes shrinkage of outcome rates; optional single-pitcher report |
| 6 | `python -m pitch_clusters.arsenal_fingerprints --comps "name" --drift "name"` | Pitcher archetypes, comps, and arsenal-mix drift over time |

Step 1 is network-bound (rate-limited fetches from Baseball Savant) and is
the slow step; everything after it runs against cached local data and is
fast, including on a full 4-season, ~3M-pitch dataset.

`pitch_clusters.assign` isn't run directly it's the runtime API
(`get_assigner()`) other modules and any downstream consumer use to load the
fitted models and assign clusters to new pitch data.

## License

MIT — see [LICENSE](LICENSE).
