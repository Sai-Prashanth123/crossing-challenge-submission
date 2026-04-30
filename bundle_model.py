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


def main():
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
    print(f"Bundled -> {OUT}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
