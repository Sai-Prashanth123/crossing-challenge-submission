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
