"""Submission entry point - Crossing Challenge.

Two-headed prediction:
  - Intent: LightGBM + isotonic calibrator (loaded from intent_model.pkl)
  - Trajectory: constant velocity (will be replaced by GRU in Task 8)

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
HORIZONS_FRAMES = [8, 15, 23, 30]  # 15 Hz -> 0.5, 1.0, 1.5, 2.0 s
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
BBOX_CLAMP = (-2000.0, 4000.0)
INTENT_CLAMP = (1e-6, 1.0 - 1e-6)

# ---- Model loading (try bundled model.pkl first, else intermediate intent_model.pkl) ----
_MODEL_PATH = Path(__file__).parent / "model.pkl"
_INTENT_PATH = Path(__file__).parent / "intent_model.pkl"

_intent_lgbm = None
_intent_calibrator = None
_trajectory_state = None  # populated in Task 8


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


def _constant_velocity_trajectory(req: dict) -> dict:
    """Same logic as the starter baseline - used until Task 8 swaps in the GRU."""
    hist = _as_2d(req["bbox_history"])
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w_last = hist[-1, 2] - hist[-1, 0]
    h_last = hist[-1, 3] - hist[-1, 1]
    vx = float(np.diff(cx[-5:]).mean())
    vy = float(np.diff(cy[-5:]).mean())
    cur_cx, cur_cy = float(cx[-1]), float(cy[-1])

    out = {}
    for steps, key in zip(HORIZONS_FRAMES, HORIZON_KEYS):
        nx, ny = cur_cx + vx * steps, cur_cy + vy * steps
        out[key] = [nx - w_last / 2, ny - h_last / 2,
                    nx + w_last / 2, ny + h_last / 2]
    return out


def _zero_work_prediction(req: dict) -> dict:
    """Last-resort fallback - never crash mid-eval."""
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


def _sanitize_bbox(bbox, req: dict):
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
