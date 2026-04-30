"""Contract tests - predict() must return the exact shape the grader expects."""
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
