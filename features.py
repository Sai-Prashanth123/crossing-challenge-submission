"""Shared feature builders for the Crossing Challenge.

Two pure functions, both consumed by training scripts AND predict.py at
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


def _enc_cat(value, vocab):
    if not value:
        return -1
    return vocab.get(value, -1)


# ============================================================================
# INTENT FEATURES - used by LightGBM classifier
# ============================================================================
# Layout (positional, used by train_intent.py + predict.py):
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


def featurize_intent(req):
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]

    vx = np.diff(cx)  # (15,)
    vy = np.diff(cy)
    ax = np.diff(vx)  # (14,)
    ay = np.diff(vy)

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    aspect_per_frame = h / np.maximum(w, 1e-6)

    feats = np.array([
        # 0..4 - current geometry
        cx[-1] / fw,
        cy[-1] / fh,
        w[-1] / fw,
        h[-1] / fh,
        float(aspect_per_frame[-1]),
        # 5..14 - full-window bbox dynamics
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
        # 15..17 - last 4 frames
        float(vx[-4:].mean()) / fw,
        float(vy[-4:].mean()) / fh,
        float(ax[-3:].mean()) / fw,
        # 18..20 - body-pose proxies
        float(aspect_per_frame.mean()),
        float(aspect_per_frame[-1]),
        float((h[-1] - h[0]) / max(h[0], 1e-6)),
        # 21..26 - ego
        float(ego_s.mean()),
        float(ego_s[-1]),
        float(ego_s.max()),
        float(ego_y.mean()),
        float(np.abs(ego_y).max()),
        1.0 if req.get("ego_available") else 0.0,
        # 27 - ego-ped relative
        float(vx[-4:].mean()) + float(ego_s[-4:].mean()) * (fw / 1000.0),
        # 28..30 - categoricals
        float(_enc_cat(req.get("time_of_day"), _TIME_OF_DAY)),
        float(_enc_cat(req.get("weather"), _WEATHER)),
        float(_enc_cat(req.get("location"), _LOCATION)),
        # 31..32 - position priors
        float(np.hypot(cx[-1] - fw / 2, cy[-1] - fh / 2)) / np.hypot(fw, fh),
        float(fh - cy[-1]) / fh,
    ], dtype=np.float32)

    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1e3, neginf=-1e3)

    assert feats.shape == (INTENT_FEATURE_COUNT,), \
        f"intent feature count drift: {feats.shape}"
    return feats


# ============================================================================
# TRAJECTORY FEATURES - used by GRU
# ============================================================================

def featurize_trajectory(req):
    """Return (16, 8) float32 - per-frame normalized features for the GRU."""
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]

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
        ego_s / 50.0,
        ego_y,
    ], axis=1).astype(np.float32)

    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    assert out.shape == (16, 8)
    return out
