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


def featurize_df(df):
    rows = df[REQUEST_FIELDS].to_dict("records")
    X = np.empty((len(rows), INTENT_FEATURE_COUNT), dtype=np.float32)
    for i, req in enumerate(rows):
        X[i] = featurize_intent(req)
    return X


def main():
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
    print(f"\nSaved intent model -> {OUT}")


if __name__ == "__main__":
    main()
