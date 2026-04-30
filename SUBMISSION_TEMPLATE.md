# Submission Writeup

See **[README.md](README.md)** for the full writeup. The README covers all sections of this template (final score, approach, what didn't work, AI tooling, next experiments, reproduction, external data) in plain English.

**Final Dev composite score:** 0.7156 (starter baseline: 0.83 — lower is better)
**Image size:** 842 MB tarball (limit: 2 GB)
**Approach in one line:** Two separate models — LightGBM for intent + small GRU predicting residuals over a constant-velocity baseline for trajectory — bundled into a single `model.pkl`.
**Total time:** ~6 hours (including ~30 min Colab GPU training).
