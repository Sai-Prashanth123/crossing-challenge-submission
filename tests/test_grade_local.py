"""End-to-end regression gate - Dev composite score must be < 0.80."""
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
    assert s["score"] < 0.80, f"score {s['score']:.4f} >= 0.80 (regressed below baseline)"
