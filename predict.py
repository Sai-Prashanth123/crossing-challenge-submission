"""Submission entry point - Crossing Challenge.

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
HORIZON_FRAMES = [8, 15, 23, 30]   # 15 Hz -> 0.5/1.0/1.5/2.0 s
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
NUM_HORIZONS = len(HORIZON_FRAMES)
BBOX_CLAMP = (-2000.0, 4000.0)
INTENT_CLAMP = (1e-6, 1.0 - 1e-6)

_MODEL_PATH = Path(__file__).parent / "model.pkl"


# ---- Model classes (mirror train_trajectory.py) ----

class TrajectoryGRU(nn.Module):
    def __init__(self, input_dim=8, hidden=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(
            hidden, hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, NUM_HORIZONS * 2)

    def forward(self, x):
        h = self.input_proj(x)
        out, _ = self.gru(h)
        return self.head(out[:, -1, :])


# ---- Module-level model state ----
_intent_lgbm = None
_intent_calibrator = None
_traj_model = None


def _load_models():
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

def _as_2d(x):
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _cv_centers(req):
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


def _trajectory_bboxes(req):
    """Run GRU + add residual to CV; return 4 bboxes with held-constant size."""
    hist = _as_2d(req["bbox_history"])
    w_last = float(hist[-1, 2] - hist[-1, 0])
    h_last = float(hist[-1, 3] - hist[-1, 1])
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    cv = _cv_centers(req)  # (4, 2)

    if _traj_model is None:
        residual = np.zeros_like(cv)
    else:
        x = featurize_trajectory(req)  # (16, 8)
        with torch.no_grad():
            pred = _traj_model(torch.from_numpy(x).unsqueeze(0))
        residual_norm = pred.squeeze(0).numpy().reshape(NUM_HORIZONS, 2)
        residual = residual_norm * np.array([fw, fh])

    centers = cv + residual

    out = {}
    for i, key in enumerate(HORIZON_KEYS):
        cx, cy = float(centers[i, 0]), float(centers[i, 1])
        out[key] = [cx - w_last / 2, cy - h_last / 2,
                    cx + w_last / 2, cy + h_last / 2]
    return out


def _zero_work_prediction(req):
    hist = _as_2d(req["bbox_history"])
    last = [float(v) for v in hist[-1]]
    return {
        "intent": 0.07,
        **{k: list(last) for k in HORIZON_KEYS},
    }


def _predict_intent(req):
    feats = featurize_intent(req).reshape(1, -1)
    raw = float(_intent_lgbm.predict_proba(feats)[0, 1])
    return float(_intent_calibrator.transform([raw])[0])


def _sanitize_intent(p):
    if not np.isfinite(p):
        return 0.5
    return float(np.clip(p, *INTENT_CLAMP))


def _sanitize_bbox(bbox, req):
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
