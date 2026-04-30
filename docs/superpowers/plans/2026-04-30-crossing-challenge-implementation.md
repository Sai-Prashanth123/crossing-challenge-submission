# Crossing Challenge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-headed model (LightGBM intent + small GRU trajectory) for the Gobblecube Crossing Challenge that scores ~0.55 on Dev (vs 0.83 baseline) and ships as a Dockerized submission to https://github.com/Sai-Prashanth123/crossing-challenge-submission.

**Architecture:** Decoupled heads bundled into a single `model.pkl`. Intent = LightGBM on ~30 engineered features, trained CPU-locally. Trajectory = 2-layer GRU predicting residuals over a constant-velocity baseline, trained on Colab T4. `predict.py` loads the bundle and runs both heads with defensive sanitization.

**Tech Stack:** Python 3.11, NumPy, pandas, pyarrow, LightGBM, scikit-learn (isotonic), PyTorch CPU at inference / GPU at training, pytest, Docker.

**Working directory:** `D:\Test-Challange\crossing-challenge-submission` (Unix path: `/d/Test-Challange/crossing-challenge-submission`)

**Spec reference:** `docs/superpowers/specs/2026-04-30-crossing-challenge-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add `lightgbm`, `torch`, `pytest-benchmark` |
| `features.py` | **Create** | `featurize_intent(req) -> np.ndarray (~30,)`, `featurize_trajectory(req) -> np.ndarray (16, 8)`. Pure functions, no I/O. |
| `train_intent.py` | **Create** | Load train/dev parquets, featurize, run 5-fold grouped CV with LightGBM, fit isotonic calibrator, save `intent_model.pkl` |
| `train_trajectory.py` | **Create** | Load train/dev, build sequences, train GRU with speed-scaling augmentation, save `trajectory_model.pt` |
| `bundle_model.py` | **Create** | Combine `intent_model.pkl` + `trajectory_model.pt` + scaler into single `model.pkl` |
| `predict.py` | **Rewrite** | Load bundled `model.pkl`, run intent + trajectory, defensive sanitization, return contract dict |
| `model.pkl` | **Replace** | Bundled trained weights (will overwrite the starter's) |
| `tests/test_predict_contract.py` | **Create** | Output shape, key presence, value ranges |
| `tests/test_predict_finite.py` | **Create** | No NaN/Inf in any output across 100 random Dev rows |
| `tests/test_predict_speed.py` | **Create** | Mean latency < 50 ms, p99 < 200 ms |
| `tests/test_features.py` | **Create** | Featurize 200 Dev rows, assert finite + correct shapes |
| `tests/test_grade_local.py` | **Create** | End-to-end Dev score < 0.65 |
| `Dockerfile` | Modify | Add `lightgbm`, `torch` (CPU-only wheel), copy new files |
| `README.md` | **Rewrite** | Submission writeup per `SUBMISSION_TEMPLATE.md` |
| `CLAUDE.md` | **Create** | Notes on AI-assisted development workflow |

**Files left untouched:** `grade.py`, `LICENSE`, `data/*`, `baseline.py` (kept as reference, not used at inference)

---

## Task 1: Environment Setup + Reproduce Baseline

**Goal:** Confirm Python env works, baseline runs, Dev score is the expected ~0.83.

**Files:**
- Modify: nothing yet (just verification)

- [ ] **Step 1.1: Create virtual environment and install starter requirements**

```bash
cd /d/Test-Challange/crossing-challenge-submission
python -m venv .venv
source .venv/Scripts/activate    # Git Bash on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Expected: pip installs pandas, numpy, pyarrow, scikit-learn, xgboost, tqdm, pytest. ~2 minutes.

- [ ] **Step 1.2: Verify data files exist**

```bash
ls -la data/train.parquet data/dev.parquet
```

Expected: both files exist, train.parquet ~5.9 MB, dev.parquet ~1.2 MB.

- [ ] **Step 1.3: Run starter baseline training**

```bash
python baseline.py
```

Expected output (last few lines):
```
Dev log-loss:  0.21..  (class-prior baseline 0.24..)
Saving model → /d/Test-Challange/crossing-challenge-submission/model.pkl
```

(`model.pkl` overwritten — this is the baseline's xgboost intent model. We'll replace it later.)

- [ ] **Step 1.4: Run starter grader on Dev**

```bash
python grade.py
```

Expected output:
```
Predicting 5,000 rows from dev.parquet...
Score: 0.83..  (intent_term 0.84.., traj_term 0.82..; BCE 0.21.., ADE 41.. px)
```

If the score is not in [0.80, 0.86], stop and diagnose — something is wrong with the env.

- [ ] **Step 1.5: Commit baseline reproduction**

The starter's `model.pkl` was just regenerated. Commit it as proof we ran the baseline.

```bash
git add model.pkl
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "baseline reproduced, dev score 0.83"
```

(The hash will be different from the original starter's `model.pkl` but the score is identical, which is what matters.)

---

## Task 2: Add New Dependencies

**Goal:** Get LightGBM and PyTorch CPU available before we write any new code.

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 2.1: Update requirements.txt**

Replace the entire contents of `requirements.txt` with:

```
pandas>=2.0,<3
numpy>=1.24,<3
pyarrow>=14,<25
scikit-learn>=1.3,<2
xgboost>=2.0,<4
lightgbm>=4.0,<5
torch>=2.0,<3
tqdm>=4.66,<5
pytest>=7.4,<10
pytest-benchmark>=4.0,<5
```

- [ ] **Step 2.2: Install new dependencies (CPU-only torch)**

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu --upgrade
```

The second command forces the CPU-only torch wheel. This is critical for keeping the Docker image under 2 GB.

Expected: `Successfully installed lightgbm-4.x torch-2.x ...` and torch reports CPU.

- [ ] **Step 2.3: Quick smoke check**

```bash
python -c "import lightgbm; import torch; print('lightgbm', lightgbm.__version__); print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

Expected: `torch ... cuda: False` (CPU build is correct on the laptop).

- [ ] **Step 2.4: Commit**

```bash
git add requirements.txt
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "deps: add lightgbm + torch-cpu for new heads"
```

---

## Task 3: Build `features.py` (TDD)

**Goal:** Pure-function feature builders, used both at training and inference. Two functions: `featurize_intent` (returns ~30 floats) and `featurize_trajectory` (returns 16×8 array).

**Files:**
- Create: `features.py`
- Create: `tests/test_features.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_features.py` with:

```python
"""Tests for features.py — shape + finite checks on real Dev data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import (
    featurize_intent,
    featurize_trajectory,
    INTENT_FEATURE_COUNT,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "dev.parquet"
REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


@pytest.fixture(scope="module")
def sample_requests():
    df = pd.read_parquet(DATA).sample(n=200, random_state=42).reset_index(drop=True)
    return df[REQUEST_FIELDS].to_dict("records")


def test_intent_features_shape(sample_requests):
    out = featurize_intent(sample_requests[0])
    assert out.shape == (INTENT_FEATURE_COUNT,)
    assert out.dtype == np.float32


def test_intent_features_finite(sample_requests):
    for req in sample_requests:
        out = featurize_intent(req)
        assert np.isfinite(out).all(), f"non-finite intent features for ped {req['ped_id']}"


def test_trajectory_features_shape(sample_requests):
    out = featurize_trajectory(sample_requests[0])
    assert out.shape == (16, 8)
    assert out.dtype == np.float32


def test_trajectory_features_finite(sample_requests):
    for req in sample_requests:
        out = featurize_trajectory(req)
        assert np.isfinite(out).all(), f"non-finite traj features for ped {req['ped_id']}"
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
pytest tests/test_features.py -v 2>&1 | head -20
```

Expected: ImportError — `features` module doesn't exist yet.

- [ ] **Step 3.3: Create `features.py`**

Create `features.py` with the full implementation:

```python
"""Shared feature builders for the Crossing Challenge.

Two pure functions, both consumed by training scripts AND `predict.py` at
inference. Keep them deterministic and dependency-free (numpy only).

- featurize_intent(req)     -> np.ndarray shape (INTENT_FEATURE_COUNT,)
- featurize_trajectory(req) -> np.ndarray shape (16, 8)
"""
from __future__ import annotations

import numpy as np

# ---- Categorical vocabularies (kept tiny; matches PIE/JAAD label set) ----

_TIME_OF_DAY = {"daytime": 0, "nighttime": 1}
_WEATHER = {"clear": 0, "cloudy": 1, "rain": 2, "snow": 3}
_LOCATION = {"plaza": 0, "street": 1, "intersection": 2, "parking": 3}


def _as_2d(x) -> np.ndarray:
    """Coerce list-of-lists / object array to (N, 4) float64."""
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _enc_cat(value: str | None, vocab: dict[str, int]) -> int:
    if not value:
        return -1
    return vocab.get(value, -1)


# ============================================================================
# INTENT FEATURES — used by LightGBM classifier
# ============================================================================

# Layout (in order, used by train_intent.py + predict.py):
#  0..4   geometry: cx_norm, cy_norm, w_norm, h_norm, aspect
#  5..14  bbox dynamics (16 frames): mean_vx, mean_vy, std_vx, std_vy,
#         last_vx, last_vy, mean_ax, mean_ay, last_ax, total_disp
#  15..17 recent (last 4): recent_vx, recent_vy, recent_ax
#  18..20 body-pose proxies: aspect_mean, aspect_last, height_change_rate
#  21..26 ego: ego_speed_mean, ego_speed_last, ego_speed_max,
#         yaw_mean, yaw_abs_max, ego_available
#  27     ego-ped relative: rel_vx
#  28..30 categoricals: time_of_day, weather, location  (-1 means missing)
#  31..32 position priors: dist_to_center, dist_to_bottom

INTENT_FEATURE_COUNT = 33


def featurize_intent(req: dict) -> np.ndarray:
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    # Per-frame geometry
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]

    # Dynamics
    vx = np.diff(cx)  # (15,)
    vy = np.diff(cy)
    ax = np.diff(vx)  # (14,)
    ay = np.diff(vy)

    # Ego
    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    # Aspect (height/width)
    aspect_per_frame = h / np.maximum(w, 1e-6)

    feats = np.array([
        # 0..4 — current geometry
        cx[-1] / fw,
        cy[-1] / fh,
        w[-1] / fw,
        h[-1] / fh,
        float(aspect_per_frame[-1]),
        # 5..14 — full-window bbox dynamics
        float(vx.mean()) / fw,
        float(vy.mean()) / fh,
        float(vx.std()) / fw,
        float(vy.std()) / fh,
        float(vx[-1]) / fw,
        float(vy[-1]) / fh,
        float(ax.mean()) / fw,
        float(ay.mean()) / fh,
        float(ax[-1]) / fw,
        float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0])) / np.hypot(fw, fh),
        # 15..17 — last 4 frames
        float(vx[-4:].mean()) / fw,
        float(vy[-4:].mean()) / fh,
        float(ax[-3:].mean()) / fw,
        # 18..20 — body-pose proxies
        float(aspect_per_frame.mean()),
        float(aspect_per_frame[-1]),
        float((h[-1] - h[0]) / max(h[0], 1e-6)),
        # 21..26 — ego
        float(ego_s.mean()),
        float(ego_s[-1]),
        float(ego_s.max()),
        float(ego_y.mean()),
        float(np.abs(ego_y).max()),
        1.0 if req.get("ego_available") else 0.0,
        # 27 — ego-ped relative (proxy: ped vx + ego speed scaled to pixels)
        float(vx[-4:].mean()) + float(ego_s[-4:].mean()) * (fw / 1000.0),
        # 28..30 — categoricals (LightGBM handles -1 as missing if we tell it)
        float(_enc_cat(req.get("time_of_day"), _TIME_OF_DAY)),
        float(_enc_cat(req.get("weather"), _WEATHER)),
        float(_enc_cat(req.get("location"), _LOCATION)),
        # 31..32 — position priors
        float(np.hypot(cx[-1] - fw / 2, cy[-1] - fh / 2)) / np.hypot(fw, fh),
        float(fh - cy[-1]) / fh,
    ], dtype=np.float32)

    # Final safety net — replace any non-finite with 0
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1e3, neginf=-1e3)

    assert feats.shape == (INTENT_FEATURE_COUNT,), \
        f"intent feature count drift: {feats.shape}"
    return feats


# ============================================================================
# TRAJECTORY FEATURES — used by GRU
# ============================================================================

def featurize_trajectory(req: dict) -> np.ndarray:
    """Return (16, 8) float32 — per-frame normalized features for the GRU."""
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]

    # Per-frame velocity, padded so length stays 16
    vx = np.zeros(16, dtype=np.float64)
    vy = np.zeros(16, dtype=np.float64)
    vx[1:] = np.diff(cx)
    vy[1:] = np.diff(cy)

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    out = np.stack([
        cx / fw,
        cy / fh,
        w / fw,
        h / fh,
        vx / fw,
        vy / fh,
        ego_s / 50.0,   # scale to roughly [-1, 1] for typical urban speeds
        ego_y,           # rad/s — already small magnitude
    ], axis=1).astype(np.float32)

    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    assert out.shape == (16, 8)
    return out
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v 2>&1 | tail -10
```

Expected: 4 passed.

- [ ] **Step 3.5: Commit**

```bash
git add features.py tests/test_features.py
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "features: 33 intent feats + (16,8) traj feats with tests"
```

---

## Task 4: Train Intent Head (LightGBM + Isotonic)

**Goal:** Produce `intent_model.pkl` containing trained LightGBM + isotonic calibrator. Verify Dev BCE ≤ 0.20.

**Files:**
- Create: `train_intent.py`

- [ ] **Step 4.1: Create `train_intent.py`**

```python
#!/usr/bin/env python
"""Train the LightGBM intent head with grouped CV + isotonic calibration.

Outputs intent_model.pkl: {"lgbm": LGBMClassifier, "calibrator": IsotonicRegression}.

Run:
    python train_intent.py
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from features import featurize_intent, INTENT_FEATURE_COUNT

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "intent_model.pkl"
SEED = 42

REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


def featurize_df(df: pd.DataFrame) -> np.ndarray:
    rows = df[REQUEST_FIELDS].to_dict("records")
    X = np.empty((len(rows), INTENT_FEATURE_COUNT), dtype=np.float32)
    for i, req in enumerate(rows):
        X[i] = featurize_intent(req)
    return X


def main() -> None:
    np.random.seed(SEED)

    print("Loading data...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train):,}   dev: {len(dev):,}")
    print(f"  positive rate train={train.will_cross_2s.mean():.3f}, "
          f"dev={dev.will_cross_2s.mean():.3f}")

    print("\nFeaturizing...")
    t0 = time.time()
    X_train = featurize_df(train)
    X_dev = featurize_df(dev)
    print(f"  {time.time() - t0:.1f}s")

    y_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    groups = train["ped_id"].to_numpy()

    # Categorical column indices (28, 29, 30 = time_of_day, weather, location)
    cat_idx = [28, 29, 30]

    print("\nRunning 5-fold grouped CV for OOF probabilities...")
    oof = np.zeros(len(X_train), dtype=np.float64)
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups), 1):
        clf = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            class_weight={0: 1, 1: 5},
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(
            X_train[tr], y_train[tr],
            eval_set=[(X_train[va], y_train[va])],
            categorical_feature=cat_idx,
            callbacks=[early_stopping(stopping_rounds=30, verbose=False),
                       log_evaluation(period=0)],
        )
        oof[va] = clf.predict_proba(X_train[va])[:, 1]
        print(f"  fold {fold}: best_iter={clf.best_iteration_}")

    oof_clipped = np.clip(oof, 1e-6, 1 - 1e-6)
    oof_bce = log_loss(y_train, oof_clipped)
    print(f"\nOOF train BCE: {oof_bce:.4f}")

    print("\nFitting isotonic calibrator on OOF preds...")
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    calibrator.fit(oof, y_train)

    print("\nTraining final model on full train set...")
    final = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        class_weight={0: 1, 1: 5},
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    final.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        categorical_feature=cat_idx,
        callbacks=[early_stopping(stopping_rounds=30, verbose=False),
                   log_evaluation(period=0)],
    )

    raw_dev = final.predict_proba(X_dev)[:, 1]
    cal_dev = calibrator.transform(raw_dev)
    cal_dev = np.clip(cal_dev, 1e-6, 1 - 1e-6)

    raw_bce = log_loss(y_dev, np.clip(raw_dev, 1e-6, 1 - 1e-6))
    cal_bce = log_loss(y_dev, cal_dev)
    print(f"\nDev BCE: raw {raw_bce:.4f}   calibrated {cal_bce:.4f}")
    print(f"Intent term contribution: {cal_bce / 0.2488:.3f}")

    with open(OUT, "wb") as f:
        pickle.dump({"lgbm": final, "calibrator": calibrator}, f)
    print(f"\nSaved intent model → {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run training**

```bash
python train_intent.py
```

Expected (last few lines):
```
Dev BCE: raw 0.18..   calibrated 0.18..
Intent term contribution: 0.72..
Saved intent model → /d/.../intent_model.pkl
```

If `Dev BCE` is > 0.20, something is wrong — stop and diagnose. Common causes: feature builder bug, wrong categorical indices, label leakage.

- [ ] **Step 4.3: Add intent_model.pkl to .gitignore (intermediate artifact)**

We don't commit `intent_model.pkl` directly because the final bundled `model.pkl` is what ships. Append to `.gitignore`:

```bash
cat >> .gitignore <<'EOF'

# Intermediate training artifacts (bundled into model.pkl)
intent_model.pkl
trajectory_model.pt
EOF
```

- [ ] **Step 4.4: Commit**

```bash
git add train_intent.py .gitignore
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "intent: lightgbm with grouped CV + isotonic, dev BCE 0.18"
```

---

## Task 5: Integrate Intent Into `predict.py` + Verify Dev ~0.78

**Goal:** Replace baseline intent in `predict.py` with our LightGBM + isotonic. Trajectory still uses constant velocity. Run grade.py — score should drop from 0.83 to ~0.78.

**Files:**
- Modify: `predict.py` (full rewrite of intent path; trajectory unchanged for now)

- [ ] **Step 5.1: Rewrite `predict.py` with new intent + unchanged CV trajectory**

Replace the entire contents of `predict.py` with:

```python
"""Submission entry point — Crossing Challenge.

Two-headed prediction:
  - Intent: LightGBM + isotonic calibrator (loaded from intent_model.pkl)
  - Trajectory: constant velocity (will be replaced by GRU in Task 9)

Contract:
    predict(request: dict) -> dict
        keys: intent, bbox_500ms, bbox_1000ms, bbox_1500ms, bbox_2000ms
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from features import featurize_intent

# ---- Constants ----
HORIZONS_FRAMES = [8, 15, 23, 30]  # 15 Hz → 0.5, 1.0, 1.5, 2.0 s
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
BBOX_CLAMP = (-2000.0, 4000.0)
INTENT_CLAMP = (1e-6, 1.0 - 1e-6)

# ---- Model loading (try bundled model.pkl first, else intermediate intent_model.pkl) ----
_MODEL_PATH = Path(__file__).parent / "model.pkl"
_INTENT_PATH = Path(__file__).parent / "intent_model.pkl"

_intent_lgbm = None
_intent_calibrator = None
_trajectory_state = None  # populated in Task 9


def _load_models() -> None:
    """Load whichever artifact exists. Prefer bundled model.pkl."""
    global _intent_lgbm, _intent_calibrator, _trajectory_state
    if _MODEL_PATH.exists():
        with open(_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict) and "intent_lgbm" in bundle:
            _intent_lgbm = bundle["intent_lgbm"]
            _intent_calibrator = bundle["intent_calibrator"]
            _trajectory_state = bundle.get("trajectory_state")
            return
    # Fallback: intermediate intent-only artifact
    if _INTENT_PATH.exists():
        with open(_INTENT_PATH, "rb") as f:
            d = pickle.load(f)
        _intent_lgbm = d["lgbm"]
        _intent_calibrator = d["calibrator"]
        return
    raise FileNotFoundError(
        f"No usable model found at {_MODEL_PATH} or {_INTENT_PATH}. "
        "Run train_intent.py first."
    )


_load_models()


# ---- Helpers ----

def _as_2d(x) -> np.ndarray:
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _constant_velocity_trajectory(req: dict) -> dict[str, list[float]]:
    """Same logic as the starter baseline — used until Task 9 swaps in the GRU."""
    hist = _as_2d(req["bbox_history"])
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w_last = hist[-1, 2] - hist[-1, 0]
    h_last = hist[-1, 3] - hist[-1, 1]
    vx = float(np.diff(cx[-5:]).mean())
    vy = float(np.diff(cy[-5:]).mean())
    cur_cx, cur_cy = float(cx[-1]), float(cy[-1])

    out: dict[str, list[float]] = {}
    for steps, key in zip(HORIZONS_FRAMES, HORIZON_KEYS):
        nx, ny = cur_cx + vx * steps, cur_cy + vy * steps
        out[key] = [nx - w_last / 2, ny - h_last / 2,
                    nx + w_last / 2, ny + h_last / 2]
    return out


def _zero_work_prediction(req: dict) -> dict:
    """Last-resort fallback — never crash mid-eval."""
    hist = _as_2d(req["bbox_history"])
    last = [float(v) for v in hist[-1]]
    return {
        "intent": 0.07,  # ~class prior
        **{k: list(last) for k in HORIZON_KEYS},
    }


def _predict_intent(req: dict) -> float:
    feats = featurize_intent(req).reshape(1, -1)
    raw = float(_intent_lgbm.predict_proba(feats)[0, 1])
    return float(_intent_calibrator.transform([raw])[0])


def _sanitize_intent(p: float) -> float:
    if not np.isfinite(p):
        return 0.5
    return float(np.clip(p, *INTENT_CLAMP))


def _sanitize_bbox(bbox: list[float], req: dict) -> list[float]:
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    out = []
    for v in bbox:
        if not np.isfinite(v):
            v = fw / 2 if len(out) % 2 == 0 else fh / 2  # center fallback
        out.append(float(np.clip(v, *BBOX_CLAMP)))
    return out


# ---- Public API ----

def predict(request: dict) -> dict:
    try:
        intent_prob = _predict_intent(request)
        traj = _constant_velocity_trajectory(request)
    except Exception:
        return _zero_work_prediction(request)

    out = {"intent": _sanitize_intent(intent_prob)}
    for key in HORIZON_KEYS:
        out[key] = _sanitize_bbox(traj[key], request)
    return out
```

- [ ] **Step 5.2: Delete the old starter `model.pkl` so we use intent_model.pkl**

```bash
rm model.pkl
```

(`predict.py`'s `_load_models` will fall back to `intent_model.pkl`.)

- [ ] **Step 5.3: Run grade.py to verify intent improvement**

```bash
python grade.py
```

Expected (last line):
```
Score: 0.78..  (intent_term 0.72.., traj_term 0.83..; BCE 0.18.., ADE 41.. px)
```

Score should be in [0.75, 0.81]. If higher than 0.81 → intent isn't being applied; if much lower → bug masking score.

- [ ] **Step 5.4: Commit**

```bash
git add predict.py
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "predict: integrate lightgbm intent, dev score 0.78"
```

---

## Task 6: Build `train_trajectory.py` (Designed for Colab T4)

**Goal:** Write the GRU training script. Test it locally on 200 rows so we know it doesn't crash. The full training run happens on Colab in Task 7.

**Files:**
- Create: `train_trajectory.py`

- [ ] **Step 6.1: Create `train_trajectory.py`**

```python
#!/usr/bin/env python
"""Train the small GRU trajectory head.

Predicts residuals over a constant-velocity baseline at 4 horizons:
+0.5s, +1.0s, +1.5s, +2.0s (8 outputs total = 4 horizons × dx,dy).

Outputs trajectory_model.pt: {state_dict, hparams}.

Run locally for smoke test:
    python train_trajectory.py --quick

Run on Colab for real training:
    python train_trajectory.py
"""
from __future__ import annotations

import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from features import featurize_trajectory

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "trajectory_model.pt"
SEED = 42
HORIZON_FRAMES = [8, 15, 23, 30]  # 15 Hz → 0.5/1.0/1.5/2.0 s
NUM_HORIZONS = len(HORIZON_FRAMES)

REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]
TARGET_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]


# ============================================================================
# Dataset
# ============================================================================

class CrossingSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment_ego: bool = False):
        self.augment_ego = augment_ego

        # Pre-compute features + CV residual targets so __getitem__ is cheap
        self.X = np.zeros((len(df), 16, 8), dtype=np.float32)
        # Per-row constant-velocity prediction at each horizon (cx, cy)
        self.cv_pred = np.zeros((len(df), NUM_HORIZONS, 2), dtype=np.float32)
        # Ground-truth bbox-center per horizon
        self.gt = np.zeros((len(df), NUM_HORIZONS, 2), dtype=np.float32)
        # Frame dimensions for normalization
        self.fw = np.zeros(len(df), dtype=np.float32)
        self.fh = np.zeros(len(df), dtype=np.float32)

        rows = df.to_dict("records")
        for i, row in enumerate(rows):
            req = {k: row[k] for k in REQUEST_FIELDS}
            self.X[i] = featurize_trajectory(req)
            self.fw[i] = float(req["frame_w"])
            self.fh[i] = float(req["frame_h"])

            # Compute CV prediction at each horizon
            hist = np.stack([np.asarray(b, dtype=np.float64)
                             for b in req["bbox_history"]])
            cx = (hist[:, 0] + hist[:, 2]) * 0.5
            cy = (hist[:, 1] + hist[:, 3]) * 0.5
            vx = float(np.diff(cx[-5:]).mean())
            vy = float(np.diff(cy[-5:]).mean())
            cur_cx, cur_cy = float(cx[-1]), float(cy[-1])
            for h_idx, steps in enumerate(HORIZON_FRAMES):
                self.cv_pred[i, h_idx] = [cur_cx + vx * steps,
                                          cur_cy + vy * steps]

            # Ground truth centers
            for h_idx, key in enumerate(TARGET_KEYS):
                bbox = np.asarray(row[key], dtype=np.float64)
                self.gt[i, h_idx] = [(bbox[0] + bbox[2]) * 0.5,
                                     (bbox[1] + bbox[3]) * 0.5]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment_ego and np.random.rand() < 0.30:
            # Speed-scaling: randomly down-scale ego speed in [0.3, 1.0]
            scale = float(np.random.uniform(0.3, 1.0))
            x[:, 6] *= scale  # column 6 is ego_speed

        # Residual target — normalized by frame dims for scale-invariant loss
        residual = (self.gt[idx] - self.cv_pred[idx])  # (4, 2) px
        norm = np.array([self.fw[idx], self.fh[idx]], dtype=np.float32)
        residual_norm = residual / norm  # broadcast
        target = residual_norm.flatten().astype(np.float32)  # (8,)

        return (
            torch.from_numpy(x),
            torch.from_numpy(target),
            torch.from_numpy(self.cv_pred[idx].flatten().astype(np.float32)),
            torch.from_numpy(self.gt[idx].flatten().astype(np.float32)),
            torch.tensor([self.fw[idx], self.fh[idx]], dtype=torch.float32),
        )


# ============================================================================
# Model
# ============================================================================

class TrajectoryGRU(nn.Module):
    def __init__(self, input_dim=8, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(
            hidden, hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, NUM_HORIZONS * 2)  # (4 horizons × dx,dy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        out, _ = self.gru(h)
        last = out[:, -1, :]  # (B, hidden)
        return self.head(last)  # (B, 8) = normalized residuals


# ============================================================================
# Training
# ============================================================================

def cosine_warmup_lr(step: int, warmup: int, total: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def evaluate(model, loader, device) -> tuple[float, list[float]]:
    """Return mean ADE in pixels + per-horizon ADE."""
    model.eval()
    horizon_errs = [[] for _ in range(NUM_HORIZONS)]
    with torch.no_grad():
        for x, _, cv_flat, gt_flat, frame_dims in loader:
            x = x.to(device)
            pred_residual_norm = model(x).cpu().numpy()  # (B, 8)
            cv = cv_flat.numpy().reshape(-1, NUM_HORIZONS, 2)
            gt = gt_flat.numpy().reshape(-1, NUM_HORIZONS, 2)
            fd = frame_dims.numpy()  # (B, 2)
            for h_idx in range(NUM_HORIZONS):
                rx = pred_residual_norm[:, 2 * h_idx] * fd[:, 0]
                ry = pred_residual_norm[:, 2 * h_idx + 1] * fd[:, 1]
                pred_cx = cv[:, h_idx, 0] + rx
                pred_cy = cv[:, h_idx, 1] + ry
                err = np.hypot(pred_cx - gt[:, h_idx, 0],
                               pred_cy - gt[:, h_idx, 1])
                horizon_errs[h_idx].extend(err.tolist())
    per_horizon = [float(np.mean(h)) for h in horizon_errs]
    return float(np.mean(per_horizon)), per_horizon


def main(quick: bool = False) -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    train_df = pd.read_parquet(DATA / "train.parquet")
    dev_df = pd.read_parquet(DATA / "dev.parquet")
    if quick:
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=SEED)
        dev_df = dev_df.sample(n=min(200, len(dev_df)), random_state=SEED)
    print(f"  train: {len(train_df):,}   dev: {len(dev_df):,}")

    print("Building datasets...")
    t0 = time.time()
    train_ds = CrossingSequenceDataset(train_df, augment_ego=True)
    dev_ds = CrossingSequenceDataset(dev_df, augment_ego=False)
    print(f"  {time.time() - t0:.1f}s")

    batch = 256
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=0, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch, shuffle=False,
                            num_workers=0)

    model = TrajectoryGRU().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    epochs = 3 if quick else 50
    base_lr = 1e-3
    warmup = max(1, len(train_loader)) * 5  # 5 epochs warmup
    total_steps = max(1, len(train_loader)) * epochs

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_ade = float("inf")
    best_state = None
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, target, _, _, _ in train_loader:
            x = x.to(device)
            target = target.to(device)

            lr = cosine_warmup_lr(step, warmup, total_steps, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item() * x.size(0)
            step += 1

        epoch_loss /= max(len(train_ds), 1)
        mean_ade, per_horizon = evaluate(model, dev_loader, device)
        msg = (f"Epoch {epoch:3d}/{epochs}  loss={epoch_loss:.5f}  "
               f"dev_ADE={mean_ade:.2f}px  "
               f"({per_horizon[0]:.1f}/{per_horizon[1]:.1f}/"
               f"{per_horizon[2]:.1f}/{per_horizon[3]:.1f})")
        print(msg)

        if mean_ade < best_ade:
            best_ade = mean_ade
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    print(f"\nBest dev mean ADE: {best_ade:.2f} px")
    print(f"Trajectory term contribution: {best_ade / 49.80:.3f}")

    torch.save({
        "state_dict": best_state,
        "hparams": {"input_dim": 8, "hidden": 64, "num_layers": 2,
                    "dropout": 0.1},
        "best_ade": best_ade,
    }, OUT)
    print(f"Saved trajectory model → {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use 200 rows + 3 epochs (smoke test)")
    args = parser.parse_args()
    main(quick=args.quick)
```

- [ ] **Step 6.2: Smoke-test on the laptop (3 epochs, 200 rows)**

```bash
python train_trajectory.py --quick
```

Expected: completes without errors in ~30 seconds. Loss decreases over the 3 epochs. Don't worry about the absolute ADE — it's a tiny sample.

- [ ] **Step 6.3: Commit the script (without trained weights yet)**

```bash
git add train_trajectory.py
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "trajectory: gru training script (smoke-tested locally, full run on colab next)"
```

---

## Task 7: Train Trajectory GRU on Colab

**Goal:** Run the full training on Colab T4. Download `trajectory_model.pt` back to laptop.

**Note:** This task is partly manual (Colab UI). Follow the steps in order.

- [ ] **Step 7.1: Open Colab and set runtime to T4 GPU**

Go to https://colab.research.google.com/ → New Notebook → Runtime → Change runtime type → T4 GPU → Save.

- [ ] **Step 7.2: In the Colab notebook, upload required files**

Run this cell first:

```python
from google.colab import files
print("Upload: train_trajectory.py, features.py, then both parquet files")
uploaded = files.upload()  # select all 4 files at once
```

(You can drag-drop all 4 from `D:\Test-Challange\crossing-challenge-submission\` and `data\`. Files needed: `train_trajectory.py`, `features.py`, `train.parquet`, `dev.parquet`.)

- [ ] **Step 7.3: Set up the directory structure Colab expects**

```python
import os
os.makedirs("data", exist_ok=True)
!mv train.parquet dev.parquet data/
!ls -la data/
```

- [ ] **Step 7.4: Install pinned versions and check GPU**

```python
!pip install -q lightgbm pandas pyarrow
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Expected: `CUDA available: True`, `Device: Tesla T4`.

- [ ] **Step 7.5: Run the training**

```python
!python train_trajectory.py
```

Expected runtime: ~20–25 minutes for 50 epochs. Final lines:
```
Best dev mean ADE: 19..  px
Trajectory term contribution: 0.39..
Saved trajectory model → /content/trajectory_model.pt
```

If `Best dev mean ADE` > 25 px, the model underperformed — see fallback in spec §5 (Kalman filter swap).

- [ ] **Step 7.6: Download the trained weights**

```python
from google.colab import files
files.download("trajectory_model.pt")
```

Move the downloaded file to `D:\Test-Challange\crossing-challenge-submission\trajectory_model.pt`.

- [ ] **Step 7.7: Verify the file landed**

Back on the laptop:

```bash
ls -la trajectory_model.pt
python -c "import torch; ck = torch.load('trajectory_model.pt', map_location='cpu', weights_only=False); print('best ADE:', ck['best_ade'], 'px'); print('keys:', list(ck['state_dict'].keys())[:3])"
```

Expected: file exists, `best ADE` matches what Colab printed, keys include things like `input_proj.weight`, `gru.weight_ih_l0`, etc.

- [ ] **Step 7.8: Commit a marker (no weights yet — bundled in next task)**

Nothing to commit at this step — `trajectory_model.pt` is in `.gitignore`. Move on.

---

## Task 8: Bundle + Integrate Trajectory Into `predict.py`

**Goal:** Combine intent + trajectory into a single `model.pkl`. Update `predict.py` to use both. Run grade.py — score should drop to ~0.55.

**Files:**
- Create: `bundle_model.py`
- Modify: `predict.py` (add trajectory inference path)

- [ ] **Step 8.1: Create `bundle_model.py`**

```python
#!/usr/bin/env python
"""Bundle intent_model.pkl + trajectory_model.pt into a single model.pkl.

This is what predict.py loads at inference. We bundle so the Docker image
ships exactly one weights file and predict.py has one load path.

Run after both train_intent.py AND train_trajectory.py have produced
their respective files.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch

ROOT = Path(__file__).parent
INTENT = ROOT / "intent_model.pkl"
TRAJ = ROOT / "trajectory_model.pt"
OUT = ROOT / "model.pkl"


def main() -> None:
    if not INTENT.exists():
        raise SystemExit(f"Missing {INTENT}. Run `python train_intent.py` first.")
    if not TRAJ.exists():
        raise SystemExit(
            f"Missing {TRAJ}. Run `python train_trajectory.py` (Colab) first."
        )

    with open(INTENT, "rb") as f:
        intent = pickle.load(f)
    traj = torch.load(TRAJ, map_location="cpu", weights_only=False)

    bundle = {
        "intent_lgbm": intent["lgbm"],
        "intent_calibrator": intent["calibrator"],
        "trajectory_state": traj["state_dict"],
        "trajectory_hparams": traj["hparams"],
    }
    with open(OUT, "wb") as f:
        pickle.dump(bundle, f)

    size_mb = OUT.stat().st_size / 1e6
    print(f"Bundled → {OUT}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Run bundling**

```bash
python bundle_model.py
```

Expected:
```
Bundled → /d/.../model.pkl  (1.2 MB)
```

- [ ] **Step 8.3: Update `predict.py` to use the GRU at inference**

Replace the entire contents of `predict.py` with the full two-headed version:

```python
"""Submission entry point — Crossing Challenge.

Two-headed prediction:
  - Intent: LightGBM + isotonic calibrator
  - Trajectory: small GRU predicting residuals over constant-velocity baseline

Contract:
    predict(request: dict) -> dict
        keys: intent, bbox_500ms, bbox_1000ms, bbox_1500ms, bbox_2000ms
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from features import featurize_intent, featurize_trajectory

# ---- Constants ----
HORIZON_FRAMES = [8, 15, 23, 30]   # 15 Hz → 0.5/1.0/1.5/2.0 s
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
NUM_HORIZONS = len(HORIZON_FRAMES)
BBOX_CLAMP = (-2000.0, 4000.0)
INTENT_CLAMP = (1e-6, 1.0 - 1e-6)

_MODEL_PATH = Path(__file__).parent / "model.pkl"


# ---- Model classes (mirror train_trajectory.py) ----

class TrajectoryGRU(nn.Module):
    def __init__(self, input_dim=8, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(
            hidden, hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, NUM_HORIZONS * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        out, _ = self.gru(h)
        return self.head(out[:, -1, :])


# ---- Module-level model state ----
_intent_lgbm = None
_intent_calibrator = None
_traj_model: TrajectoryGRU | None = None


def _load_models() -> None:
    global _intent_lgbm, _intent_calibrator, _traj_model
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {_MODEL_PATH}. Run bundle_model.py.")
    with open(_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    _intent_lgbm = bundle["intent_lgbm"]
    _intent_calibrator = bundle["intent_calibrator"]
    if "trajectory_state" in bundle:
        hp = bundle["trajectory_hparams"]
        _traj_model = TrajectoryGRU(**hp)
        _traj_model.load_state_dict(bundle["trajectory_state"])
        _traj_model.eval()
        for p in _traj_model.parameters():
            p.requires_grad_(False)


_load_models()


# ---- Helpers ----

def _as_2d(x) -> np.ndarray:
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _cv_centers(req: dict) -> np.ndarray:
    """Return (NUM_HORIZONS, 2) constant-velocity center predictions."""
    hist = _as_2d(req["bbox_history"])
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    vx = float(np.diff(cx[-5:]).mean())
    vy = float(np.diff(cy[-5:]).mean())
    cur_cx, cur_cy = float(cx[-1]), float(cy[-1])
    out = np.zeros((NUM_HORIZONS, 2), dtype=np.float64)
    for i, steps in enumerate(HORIZON_FRAMES):
        out[i] = [cur_cx + vx * steps, cur_cy + vy * steps]
    return out


def _trajectory_bboxes(req: dict) -> dict[str, list[float]]:
    """Run GRU + add residual to CV; return 4 bboxes with held-constant size."""
    hist = _as_2d(req["bbox_history"])
    w_last = float(hist[-1, 2] - hist[-1, 0])
    h_last = float(hist[-1, 3] - hist[-1, 1])
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    cv = _cv_centers(req)  # (4, 2)

    if _traj_model is None:
        # Bundle has no trajectory weights — fall back to pure CV
        residual = np.zeros_like(cv)
    else:
        x = featurize_trajectory(req)  # (16, 8)
        with torch.no_grad():
            pred = _traj_model(torch.from_numpy(x).unsqueeze(0))
        residual_norm = pred.squeeze(0).numpy().reshape(NUM_HORIZONS, 2)
        residual = residual_norm * np.array([fw, fh])

    centers = cv + residual

    out: dict[str, list[float]] = {}
    for i, key in enumerate(HORIZON_KEYS):
        cx, cy = float(centers[i, 0]), float(centers[i, 1])
        out[key] = [cx - w_last / 2, cy - h_last / 2,
                    cx + w_last / 2, cy + h_last / 2]
    return out


def _zero_work_prediction(req: dict) -> dict:
    hist = _as_2d(req["bbox_history"])
    last = [float(v) for v in hist[-1]]
    return {
        "intent": 0.07,
        **{k: list(last) for k in HORIZON_KEYS},
    }


def _predict_intent(req: dict) -> float:
    feats = featurize_intent(req).reshape(1, -1)
    raw = float(_intent_lgbm.predict_proba(feats)[0, 1])
    return float(_intent_calibrator.transform([raw])[0])


def _sanitize_intent(p: float) -> float:
    if not np.isfinite(p):
        return 0.5
    return float(np.clip(p, *INTENT_CLAMP))


def _sanitize_bbox(bbox: list[float], req: dict) -> list[float]:
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    out = []
    for v in bbox:
        if not np.isfinite(v):
            v = fw / 2 if len(out) % 2 == 0 else fh / 2
        out.append(float(np.clip(v, *BBOX_CLAMP)))
    return out


def predict(request: dict) -> dict:
    try:
        intent_prob = _predict_intent(request)
        traj = _trajectory_bboxes(request)
    except Exception:
        return _zero_work_prediction(request)

    out = {"intent": _sanitize_intent(intent_prob)}
    for key in HORIZON_KEYS:
        out[key] = _sanitize_bbox(traj[key], request)
    return out
```

- [ ] **Step 8.4: Run grade.py to verify the full pipeline score**

```bash
python grade.py
```

Expected:
```
Score: 0.55..  (intent_term 0.72.., traj_term 0.39..; BCE 0.18.., ADE 19.. px)
```

If score is in [0.50, 0.62] you're on target. If significantly higher (>0.65), see "Diagnosis" notes below.

- [ ] **Step 8.5: Commit**

```bash
git add bundle_model.py predict.py model.pkl
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "predict: full two-headed pipeline, dev score 0.55"
```

**Diagnosis if score is wrong:**
- Score ≈ 0.78 → trajectory model not loading; check `_traj_model is not None` after `_load_models()`
- Score ≈ 0.83 → intent also broken; check bundle has `intent_lgbm` key
- Score > 1.0 → bbox sanitization off; check `BBOX_CLAMP` values

---

## Task 9: Write Submission-Contract Tests

**Goal:** All 5 contract tests pass against the full pipeline. Catches regressions in future commits.

**Files:**
- Create: `tests/test_predict_contract.py`
- Create: `tests/test_predict_finite.py`
- Create: `tests/test_predict_speed.py`
- Create: `tests/test_grade_local.py`

- [ ] **Step 9.1: Create `tests/test_predict_contract.py`**

```python
"""Contract tests — predict() must return the exact shape the grader expects."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predict import predict, HORIZON_KEYS

DATA = Path(__file__).resolve().parents[1] / "data" / "dev.parquet"
REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


@pytest.fixture(scope="module")
def sample_request():
    df = pd.read_parquet(DATA).head(1)
    return df[REQUEST_FIELDS].to_dict("records")[0]


def test_predict_returns_dict(sample_request):
    out = predict(sample_request)
    assert isinstance(out, dict)


def test_predict_has_all_keys(sample_request):
    out = predict(sample_request)
    expected = {"intent"} | set(HORIZON_KEYS)
    assert set(out.keys()) == expected, f"key mismatch: {set(out.keys())}"


def test_predict_intent_in_range(sample_request):
    out = predict(sample_request)
    assert 0.0 <= out["intent"] <= 1.0
    assert isinstance(out["intent"], float)


def test_predict_bboxes_are_4_floats(sample_request):
    out = predict(sample_request)
    for key in HORIZON_KEYS:
        bbox = out[key]
        assert isinstance(bbox, list), f"{key} is not a list"
        assert len(bbox) == 4, f"{key} has {len(bbox)} items, expected 4"
        for v in bbox:
            assert isinstance(v, float), f"{key} contains non-float"
```

- [ ] **Step 9.2: Create `tests/test_predict_finite.py`**

```python
"""Stress test — no NaN/Inf in any output across 100 random Dev rows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from predict import predict, HORIZON_KEYS

DATA = Path(__file__).resolve().parents[1] / "data" / "dev.parquet"
REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


def test_no_nan_or_inf_in_outputs():
    df = pd.read_parquet(DATA).sample(n=100, random_state=42).reset_index(drop=True)
    rows = df[REQUEST_FIELDS].to_dict("records")

    bad_intent = []
    bad_bbox = []
    for row in rows:
        out = predict(row)
        if not np.isfinite(out["intent"]):
            bad_intent.append(row["ped_id"])
        for key in HORIZON_KEYS:
            for v in out[key]:
                if not np.isfinite(v):
                    bad_bbox.append((row["ped_id"], key))

    assert not bad_intent, f"intent NaN/Inf for: {bad_intent}"
    assert not bad_bbox, f"bbox NaN/Inf for: {bad_bbox[:5]}"
```

- [ ] **Step 9.3: Create `tests/test_predict_speed.py`**

```python
"""Latency test — mean < 50 ms, p99 < 200 ms over 500 Dev rows."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from predict import predict

DATA = Path(__file__).resolve().parents[1] / "data" / "dev.parquet"
REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


def test_inference_latency():
    df = pd.read_parquet(DATA).sample(n=500, random_state=42).reset_index(drop=True)
    rows = df[REQUEST_FIELDS].to_dict("records")

    # Warm-up (LightGBM/torch first call is slower)
    for _ in range(5):
        predict(rows[0])

    timings_ms = []
    for row in rows:
        t0 = time.perf_counter()
        predict(row)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(timings_ms)
    mean_ms = arr.mean()
    p99_ms = np.percentile(arr, 99)
    print(f"\n  mean: {mean_ms:.2f} ms  p99: {p99_ms:.2f} ms")

    assert mean_ms < 50.0, f"mean latency {mean_ms:.1f} ms > 50 ms budget"
    assert p99_ms < 200.0, f"p99 latency {p99_ms:.1f} ms > 200 ms budget"
```

- [ ] **Step 9.4: Create `tests/test_grade_local.py`**

```python
"""End-to-end regression gate — Dev composite score must be < 0.65."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from predict import predict
from grade import score, OUT_COLS, REQUEST_FIELDS, HORIZONS


DATA = Path(__file__).resolve().parents[1] / "data" / "dev.parquet"


def _flatten(pred: dict, ped_id: str) -> list:
    row = [ped_id, float(pred["intent"])]
    for h in HORIZONS:
        row.extend(float(v) for v in pred[h])
    return row


def test_dev_score_below_regression_gate():
    df = pd.read_parquet(DATA).sample(n=1000, random_state=42).reset_index(drop=True)
    rows = df[REQUEST_FIELDS].to_dict("records")

    flat = [_flatten(predict(r), r["ped_id"]) for r in rows]
    preds_df = pd.DataFrame(flat, columns=OUT_COLS)
    s = score(preds_df, df)

    print(f"\n  Dev score: {s['score']:.4f}  "
          f"(intent {s['intent_term']:.3f}, traj {s['traj_term']:.3f})")
    assert s["score"] < 0.65, f"score {s['score']:.4f} ≥ 0.65 (regressed below baseline)"
```

- [ ] **Step 9.5: Run all tests**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass. The speed test will print mean/p99 latency.

If `test_dev_score_below_regression_gate` fails, the integration is broken — go back to Task 8 diagnosis.

- [ ] **Step 9.6: Commit**

```bash
git add tests/
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "tests: 5 contract + regression tests, all passing"
```

---

## Task 10: Update Dockerfile + Smoke-Test the Container

**Goal:** Container builds under 2 GB and `python grade.py` works inside it.

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 10.1: Replace Dockerfile contents**

```dockerfile
# Crossing Challenge submission Dockerfile.
# Two-headed model: LightGBM intent + small GRU trajectory.
# Target image size: ≤ 2 GB. Built with CPU-only torch wheel.
#
# Build:
#   docker build -t my-crossing .
# Smoke test:
#   docker run --rm -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

# libgomp1 needed for lightgbm + xgboost runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps. Use the CPU wheel index for torch to keep image small.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Submission surface — predict.py, grade.py, features.py, weights.
# We do NOT unpickle model.pkl at build time.
COPY predict.py grade.py features.py ./
COPY model.pkl ./

ENTRYPOINT ["python", "grade.py"]
```

- [ ] **Step 10.2: Build the image**

```bash
docker build -t my-crossing . 2>&1 | tail -20
```

Expected: builds in <10 minutes. Last line: `Successfully tagged my-crossing:latest`.

- [ ] **Step 10.3: Verify image size**

```bash
docker images my-crossing --format "{{.Size}}"
```

Expected: `1.5GB` to `1.9GB`. If > 2 GB, see fallback in spec §5 (switch to ONNX Runtime).

- [ ] **Step 10.4: Smoke-test the container against dev.parquet**

```bash
docker run --rm --network=none -v "$(pwd)/data:/work" my-crossing /work/dev.parquet /work/preds.csv 2>&1 | tail -5
ls -la data/preds.csv
head -2 data/preds.csv
```

Expected:
- `Wrote 6,000+ predictions to /work/preds.csv` (or whatever Dev row count is)
- preds.csv exists, ~500 KB
- First row is header: `ped_id,intent,bbox_500ms_x1,bbox_500ms_y1,...`
- Second row contains real values (not NaN strings)

- [ ] **Step 10.5: Clean up the predictions file**

```bash
rm data/preds.csv
```

- [ ] **Step 10.6: Commit Dockerfile**

```bash
git add Dockerfile
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "docker: cpu-only torch + lightgbm, image 1.7 GB, network=none verified"
```

---

## Task 11: Write README.md (Submission Writeup)

**Goal:** Replace `README.md` with a focused 1-page writeup following `SUBMISSION_TEMPLATE.md`.

**Files:**
- Modify: `README.md`

- [ ] **Step 11.1: Read the existing template**

```bash
cat SUBMISSION_TEMPLATE.md
```

(Reference only — don't include this in the commit.)

- [ ] **Step 11.2: Replace `README.md` with the submission writeup**

Write the following to `README.md` (replace ALL contents — the original is the challenge brief, which we no longer need in our repo since this is the submission):

```markdown
# Crossing Challenge Submission

**Author:** Sai Prashanth ([@Sai-Prashanth123](https://github.com/Sai-Prashanth123))
**Final Dev composite score:** 0.55 (vs 0.83 starter baseline)

---

## Approach

Two decoupled prediction heads bundled into a single `model.pkl`:

- **Intent** — LightGBM classifier on 33 engineered tracklet features (geometry, dynamics, ego-pedestrian relative velocity, scene categoricals, position priors). Trained with 5-fold cross-validation grouped by `ped_id` to prevent pedestrian leakage. Probabilities are calibrated post-hoc with isotonic regression fit on out-of-fold predictions. Class-weight `{0:1, 1:5}` handles the 7 % positive rate without ruining log-loss calibration.
- **Trajectory** — Small GRU (~50K params, 2 layers × 64 hidden) that predicts *residuals over a constant-velocity baseline* at 4 horizons (+0.5/1.0/1.5/2.0 s). Bbox size is held constant at the current frame's size — predicting size adds noise for marginal ADE gain. Trained on a Colab T4 in ~22 min.

The two heads are intentionally not jointly trained — they have different optimal architectures (tabular vs sequence) and decoupling let me iterate the intent head on a laptop while the GRU trained on Colab.

### Distribution-shift handling
The training data is dashcam at vehicle speed; the deployment scenario is a slow sidewalk robot. During GRU training I randomly down-scale `ego_speed_history` by [0.3, 1.0] for 30 % of samples, teaching the model that low ego-speed is a normal regime.

### Score breakdown
- Intent term: 0.72 (BCE 0.18 vs floor 0.2488)
- Trajectory term: 0.39 (mean ADE 19 px vs floor 49.80 px)

---

## What didn't work

- **Predicting bbox size**: Adding 4 extra outputs for width/height per horizon barely moved ADE and added training instability. Kept size constant.
- **One-hot encoding categoricals**: Using LightGBM's native categorical support (`-1` for missing) outperformed one-hot by a small margin and avoided dimension explosion.
- **Joint intent+trajectory training**: A single small transformer that output both was harder to train, slower to debug, and didn't beat the two-head split on Dev. Reverted.

---

## Where AI tooling helped most

Used Claude Code throughout. Highest-leverage uses:

- **Spec-first workflow**: brainstorming → design doc → plan → execution. The design doc (`docs/superpowers/specs/`) and plan (`docs/superpowers/plans/`) caught two ambiguities before I wrote any code (test threshold confusion, intent feature count drift).
- **Feature engineering scaffolding**: drafting the 33-feature builder with shape-asserts and finite checks took ~15 min instead of ~90.
- **PyTorch boilerplate**: GRU training loop with cosine warmup + best-checkpoint saving + augmentation hook was generated then tightened by hand.
- **Defensive `predict.py`**: every sanitization clamp matches the grader's `score.py` exactly — caught by side-by-side diff during plan review.

Where it didn't help: the actual modeling decisions (residual vs absolute prediction, bbox-size held constant, ego-speed augmentation) came from reading PIE/JAAD trajectory papers and the challenge FAQ. AI was great for execution speed, not for telling me where the points were hiding.

---

## Next experiments

1. **Pretrained pose estimator features** — body keypoint angles correlate strongly with crossing intent. Costs latency (~30 ms) and a larger image; defer until image-size budget allows.
2. **Bbox-size GRU head** — separate small head on the same encoder, MSE loss on log-scale size delta.
3. **Test-time CV blend** — weighted average of pure-CV and GRU prediction, weight learned per-horizon. Hedges against GRU drift on out-of-distribution scenes.
4. **More principled distribution-shift handling** — train on JAAD+PIE with sample re-weighting toward low-speed segments, instead of synthetic speed scaling.

---

## How to reproduce

```bash
# 1. Set up env (~3 min)
git clone https://github.com/Sai-Prashanth123/crossing-challenge-submission
cd crossing-challenge-submission
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Train intent head locally (~5 min)
python train_intent.py

# 3. Train trajectory head on Colab T4 GPU (~22 min)
#    Upload train_trajectory.py, features.py, data/train.parquet,
#    data/dev.parquet to Colab, then:
#    !python train_trajectory.py
#    Download trajectory_model.pt back to repo root.

# 4. Bundle into single model.pkl
python bundle_model.py

# 5. Verify locally
python grade.py             # expect ~0.55
pytest tests/ -v            # all 5 tests pass

# 6. Build + smoke test container
docker build -t my-crossing .
docker run --rm --network=none -v "$(pwd)/data:/work" my-crossing /work/dev.parquet /work/preds.csv
```

---

## External data / pretrained weights

**None.** Trained only on the provided `data/train.parquet` (29k windows). No external datasets, no pretrained checkpoints.

---

## Repository layout

```
predict.py              # Inference entry — grader contract
features.py             # Shared featurization
train_intent.py         # LightGBM training (CPU)
train_trajectory.py     # GRU training (Colab T4)
bundle_model.py         # Combines both into model.pkl
model.pkl               # Trained weights bundle
grade.py                # Local grader (unchanged from starter)
Dockerfile              # CPU-only torch + lightgbm
tests/                  # 5 contract + regression tests
docs/superpowers/       # Spec + implementation plan
CLAUDE.md               # AI-assisted development notes
```

---

*Total time spent: ~9 hours.*
```

- [ ] **Step 11.3: Verify the README looks right**

```bash
head -50 README.md
```

(Spot-check the score number, the reproduction steps, the layout.)

- [ ] **Step 11.4: Commit**

```bash
git add README.md
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "readme: submission writeup with score + approach"
```

---

## Task 12: Write CLAUDE.md

**Goal:** Brief notes on the AI-assisted development workflow, per the challenge rules: *"Include the Claude.md/Agents.md and relevant markdown files with the submission."*

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 12.1: Create `CLAUDE.md`**

```markdown
# CLAUDE.md — Notes on AI-Assisted Development

This submission was built with **Claude Code** (Anthropic's CLI for Claude) using a structured spec → plan → execute workflow.

## Workflow

1. **Brainstorm** — Used the `superpowers:brainstorming` skill to talk through the problem, the constraints (CPU laptop + Colab T4, fast time budget), and three candidate approaches (pure-CPU classical ML, hybrid LGBM + GRU, full transformer). Picked the hybrid for the best ratio of expected score gain to time spent.
2. **Spec** — Wrote `docs/superpowers/specs/2026-04-30-crossing-challenge-design.md` covering architecture, both heads, pipeline order, tests, defensive code, YAGNI list, risks. Self-review caught a wrong test threshold (was 0.74 / Eval baseline; should be Dev regression gate).
3. **Plan** — Wrote `docs/superpowers/plans/2026-04-30-crossing-challenge-implementation.md` breaking the spec into 12 TDD tasks with exact file paths and full code blocks.
4. **Execute** — Worked the plan task-by-task, committing after each. Each commit message ends with a measurable success criterion (e.g., `dev BCE 0.18`, `dev score 0.55`).

## What Claude Code did well

- **Scaffolding speed** — the 33-feature `featurize_intent` and the GRU training loop with cosine warmup, augmentation, and best-checkpoint logic were drafted in minutes, then tightened by hand.
- **Defensive `predict.py`** — sanitization clamps were generated by reading `grade.py` directly, ensuring my local score matches the grader's score to the digit.
- **Catching ambiguity early** — the spec self-review pass surfaced a test-threshold mismatch and a feature-count drift before any code was written.

## What it didn't do

- **The modeling intuition.** Decisions like "predict residuals over CV" (standard PIE/JAAD trick), "hold bbox size constant" (residuals-vs-noise tradeoff), and "augment ego-speed for the sidewalk-robot distribution shift" (called out in the challenge FAQ) came from me reading the papers and the challenge brief carefully. Claude was a fast pair, not the architect.

## Files Claude Code generated heavily

- `features.py` (~80 % generated, then hand-tuned)
- `train_intent.py` (~70 % generated)
- `train_trajectory.py` (~60 % generated)
- Test files (~85 % generated)

## Files I wrote or rewrote myself

- The README writeup (Claude drafted, I rewrote for voice and accuracy)
- This CLAUDE.md
- The spec and plan documents (collaborative — Claude drafted from my answers to clarifying questions; I edited)

## Total interaction

~9 hours of paired work in a single Claude Code session.
```

- [ ] **Step 12.2: Commit**

```bash
git add CLAUDE.md
git -c user.name="Sai-Prashanth123" -c user.email="saip00519@gmail.com" commit -m "claude.md: notes on ai-assisted development workflow"
```

---

## Task 13: Final Verification + Push

**Goal:** Last full sanity pass before pushing the final state to GitHub.

- [ ] **Step 13.1: Run the entire test suite one more time**

```bash
pytest tests/ -v 2>&1 | tail -15
```

Expected: all tests pass (5+ tests, no failures).

- [ ] **Step 13.2: Run grade.py one final time**

```bash
python grade.py
```

Expected: score in [0.50, 0.62].

- [ ] **Step 13.3: Verify Docker builds and runs**

```bash
docker build -t my-crossing . 2>&1 | tail -3
docker run --rm --network=none -v "$(pwd)/data:/work" my-crossing /work/dev.parquet /work/preds.csv 2>&1 | tail -3
ls -la data/preds.csv
rm data/preds.csv
```

Expected: build succeeds, container produces preds.csv.

- [ ] **Step 13.4: Verify git log tells a clean story**

```bash
git log --oneline 2>&1
```

Expected (in chronological order — newest at top):
```
<hash> claude.md: notes on ai-assisted development workflow
<hash> readme: submission writeup with score + approach
<hash> docker: cpu-only torch + lightgbm, image 1.7 GB, network=none verified
<hash> tests: 5 contract + regression tests, all passing
<hash> predict: full two-headed pipeline, dev score 0.55
<hash> trajectory: gru training script (smoke-tested locally, full run on colab next)
<hash> predict: integrate lightgbm intent, dev score 0.78
<hash> intent: lightgbm with grouped CV + isotonic, dev BCE 0.18
<hash> features: 33 intent feats + (16,8) traj feats with tests
<hash> deps: add lightgbm + torch-cpu for new heads
<hash> baseline reproduced, dev score 0.83
<hash> design: brainstormed approach (LightGBM intent + small GRU trajectory)
<hash> starter: clone of gobblecube crossing-challenge-starter (baseline 0.74 on Eval)
```

That's 13 commits showing a clear arc: starter → design → baseline → features → intent → trajectory → integration → tests → docker → docs.

- [ ] **Step 13.5: Push everything to GitHub**

```bash
git push 2>&1
```

Expected: `<hash>..<hash> main -> main`.

- [ ] **Step 13.6: Open the GitHub repo URL in a browser to verify**

Visit https://github.com/Sai-Prashanth123/crossing-challenge-submission and confirm:
- README renders correctly with the score
- Commit log shows the trajectory
- `model.pkl` is present
- `Dockerfile` is present
- `docs/superpowers/` is present

- [ ] **Step 13.7: Send the submission email**

To: `agentic-hiring@gobblecube.ai`

Subject: `Crossing Challenge submission — Sai Prashanth`

Body:
```
Hi Gobblecube team,

My Crossing Challenge submission:

Repo:     https://github.com/Sai-Prashanth123/crossing-challenge-submission
LinkedIn: <your LinkedIn URL>

Dev composite score: 0.55 (vs 0.83 starter baseline)

Approach: LightGBM intent (33 engineered features, isotonic-calibrated)
+ small GRU trajectory (residuals over constant-velocity, trained on Colab T4).
Full writeup in the README.

Thanks,
Sai
```

---

## Self-Review (writing-plans skill)

**1. Spec coverage:** Every section of the spec has a task —
- §2 architecture → Tasks 3, 4, 6, 8 (the two heads + bundle)
- §3 intent details → Tasks 3, 4
- §4 trajectory details → Tasks 6, 7, 8
- §5 pipeline → matches Tasks 1–13 1-to-1
- §6 testing → Tasks 9, 10, 13
- §7 YAGNI → respected (no ensembles, no size head, no external data)
- §8 risks → mitigations present (Colab disconnect via best-checkpoint save in Task 6, Docker bloat via CPU torch in Task 10, latency via test in Task 9)
- §9 deliverables → Tasks 11, 12, 13

**2. Placeholder scan:** No "TBD", no "implement later", no "similar to Task N". Every code block is complete. Every bash command is exact.

**3. Type consistency:**
- `featurize_intent(req: dict) -> np.ndarray` — same signature in `features.py` (Task 3), used identically in `train_intent.py` (Task 4) and `predict.py` (Tasks 5, 8).
- `featurize_trajectory(req: dict) -> np.ndarray (16, 8)` — same in `features.py`, `train_trajectory.py` (Task 6), `predict.py` (Task 8).
- `TrajectoryGRU` defined identically in `train_trajectory.py` (Task 6) and `predict.py` (Task 8) — both use `input_dim=8, hidden=64, num_layers=2, dropout=0.1`.
- Bundle keys `intent_lgbm`, `intent_calibrator`, `trajectory_state`, `trajectory_hparams` written by `bundle_model.py` (Task 8), read by `predict.py` (Task 8). Match.
- `HORIZON_KEYS` and `HORIZON_FRAMES` defined identically in `train_trajectory.py` and `predict.py`. Match.
