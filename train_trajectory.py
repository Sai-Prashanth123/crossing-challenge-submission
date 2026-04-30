#!/usr/bin/env python
"""Train the small GRU trajectory head.

Predicts residuals over a constant-velocity baseline at 4 horizons:
+0.5s, +1.0s, +1.5s, +2.0s (8 outputs total = 4 horizons x dx,dy).

Outputs trajectory_model.pt: {state_dict, hparams}.

Run locally for smoke test:
    python train_trajectory.py --quick

Run on Colab for real training:
    python train_trajectory.py
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from features import featurize_trajectory

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "trajectory_model.pt"
SEED = 42
HORIZON_FRAMES = [8, 15, 23, 30]  # 15 Hz -> 0.5/1.0/1.5/2.0 s
NUM_HORIZONS = len(HORIZON_FRAMES)

REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]
TARGET_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]


# ============================================================================
# Dataset
# ============================================================================

class CrossingSequenceDataset(Dataset):
    def __init__(self, df, augment_ego=False):
        self.augment_ego = augment_ego

        self.X = np.zeros((len(df), 16, 8), dtype=np.float32)
        self.cv_pred = np.zeros((len(df), NUM_HORIZONS, 2), dtype=np.float32)
        self.gt = np.zeros((len(df), NUM_HORIZONS, 2), dtype=np.float32)
        self.fw = np.zeros(len(df), dtype=np.float32)
        self.fh = np.zeros(len(df), dtype=np.float32)

        rows = df.to_dict("records")
        for i, row in enumerate(rows):
            req = {k: row[k] for k in REQUEST_FIELDS}
            self.X[i] = featurize_trajectory(req)
            self.fw[i] = float(req["frame_w"])
            self.fh[i] = float(req["frame_h"])

            hist = np.stack([np.asarray(b, dtype=np.float64)
                             for b in req["bbox_history"]])
            cx = (hist[:, 0] + hist[:, 2]) * 0.5
            cy = (hist[:, 1] + hist[:, 3]) * 0.5
            vx = float(np.diff(cx[-5:]).mean())
            vy = float(np.diff(cy[-5:]).mean())
            cur_cx, cur_cy = float(cx[-1]), float(cy[-1])
            for h_idx, steps in enumerate(HORIZON_FRAMES):
                self.cv_pred[i, h_idx] = [cur_cx + vx * steps,
                                          cur_cy + vy * steps]

            for h_idx, key in enumerate(TARGET_KEYS):
                bbox = np.asarray(row[key], dtype=np.float64)
                self.gt[i, h_idx] = [(bbox[0] + bbox[2]) * 0.5,
                                     (bbox[1] + bbox[3]) * 0.5]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment_ego and np.random.rand() < 0.30:
            scale = float(np.random.uniform(0.3, 1.0))
            x[:, 6] *= scale  # column 6 is ego_speed

        residual = (self.gt[idx] - self.cv_pred[idx])  # (4, 2) px
        norm = np.array([self.fw[idx], self.fh[idx]], dtype=np.float32)
        residual_norm = residual / norm
        target = residual_norm.flatten().astype(np.float32)  # (8,)

        return (
            torch.from_numpy(x),
            torch.from_numpy(target),
            torch.from_numpy(self.cv_pred[idx].flatten().astype(np.float32)),
            torch.from_numpy(self.gt[idx].flatten().astype(np.float32)),
            torch.tensor([self.fw[idx], self.fh[idx]], dtype=torch.float32),
        )


# ============================================================================
# Model
# ============================================================================

class TrajectoryGRU(nn.Module):
    def __init__(self, input_dim=8, hidden=64, num_layers=2, dropout=0.1):
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


# ============================================================================
# Training
# ============================================================================

def cosine_warmup_lr(step, warmup, total, base):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def evaluate(model, loader, device):
    model.eval()
    horizon_errs = [[] for _ in range(NUM_HORIZONS)]
    with torch.no_grad():
        for x, _, cv_flat, gt_flat, frame_dims in loader:
            x = x.to(device)
            pred_residual_norm = model(x).cpu().numpy()  # (B, 8)
            cv = cv_flat.numpy().reshape(-1, NUM_HORIZONS, 2)
            gt = gt_flat.numpy().reshape(-1, NUM_HORIZONS, 2)
            fd = frame_dims.numpy()
            for h_idx in range(NUM_HORIZONS):
                rx = pred_residual_norm[:, 2 * h_idx] * fd[:, 0]
                ry = pred_residual_norm[:, 2 * h_idx + 1] * fd[:, 1]
                pred_cx = cv[:, h_idx, 0] + rx
                pred_cy = cv[:, h_idx, 1] + ry
                err = np.hypot(pred_cx - gt[:, h_idx, 0],
                               pred_cy - gt[:, h_idx, 1])
                horizon_errs[h_idx].extend(err.tolist())
    per_horizon = [float(np.mean(h)) for h in horizon_errs]
    return float(np.mean(per_horizon)), per_horizon


def main(quick=False):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    train_df = pd.read_parquet(DATA / "train.parquet")
    dev_df = pd.read_parquet(DATA / "dev.parquet")
    if quick:
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=SEED)
        dev_df = dev_df.sample(n=min(200, len(dev_df)), random_state=SEED)
    print(f"  train: {len(train_df):,}   dev: {len(dev_df):,}")

    print("Building datasets...")
    t0 = time.time()
    train_ds = CrossingSequenceDataset(train_df, augment_ego=True)
    dev_ds = CrossingSequenceDataset(dev_df, augment_ego=False)
    print(f"  {time.time() - t0:.1f}s")

    batch = 256
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=0, drop_last=not quick)
    dev_loader = DataLoader(dev_ds, batch_size=batch, shuffle=False,
                            num_workers=0)

    model = TrajectoryGRU().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    epochs = 3 if quick else 50
    base_lr = 1e-3
    steps_per_epoch = max(1, len(train_loader))
    warmup = steps_per_epoch * 5
    total_steps = steps_per_epoch * epochs

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_ade = float("inf")
    best_state = None
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for x, target, _, _, _ in train_loader:
            x = x.to(device)
            target = target.to(device)

            lr = cosine_warmup_lr(step, warmup, total_steps, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item() * x.size(0)
            seen += x.size(0)
            step += 1

        epoch_loss /= max(seen, 1)
        mean_ade, per_horizon = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch:3d}/{epochs}  loss={epoch_loss:.5f}  "
              f"dev_ADE={mean_ade:.2f}px  "
              f"({per_horizon[0]:.1f}/{per_horizon[1]:.1f}/"
              f"{per_horizon[2]:.1f}/{per_horizon[3]:.1f})")

        if mean_ade < best_ade:
            best_ade = mean_ade
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    print(f"\nBest dev mean ADE: {best_ade:.2f} px")
    print(f"Trajectory term contribution: {best_ade / 49.80:.3f}")

    torch.save({
        "state_dict": best_state,
        "hparams": {"input_dim": 8, "hidden": 64, "num_layers": 2,
                    "dropout": 0.1},
        "best_ade": best_ade,
    }, OUT)
    print(f"Saved trajectory model -> {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use 200 rows + 3 epochs (smoke test)")
    args = parser.parse_args()
    main(quick=args.quick)
