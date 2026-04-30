"""Latency test - mean < 50 ms, p99 < 200 ms over 500 Dev rows."""
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
