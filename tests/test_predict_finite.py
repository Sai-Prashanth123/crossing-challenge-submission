"""Stress test - no NaN/Inf in any output across 100 random Dev rows."""
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
