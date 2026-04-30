# Crossing Challenge — Design Spec

**Date:** 2026-04-30
**Author:** Sai Prashanth (with Claude Code)
**Goal:** Beat the 0.74 Eval baseline on the Gobblecube Crossing Challenge — pedestrian crossing intent + 2-second trajectory prediction — and ship a hireable submission.

---

## 1. Context & Constraints

**Challenge:** Predict, from 16 frames (≈1.07 s) of past pedestrian bbox history + ego motion, two things:
1. `intent` — P(crosses within next 2 s), float in [0, 1]
2. Four future bounding boxes at +0.5 / 1.0 / 1.5 / 2.0 s

**Scoring:** `score = 0.5 * (BCE/BCE_FLOOR) + 0.5 * (mean_pixel_ADE/ADE_FLOOR)`
- `BCE_FLOOR = 0.2488`, `ADE_FLOOR = 49.80 px`
- 1.0 = "did literally nothing"; 0.0 = perfect; lower is better
- Baseline: 0.83 Dev / 0.74 Eval

**Hard constraints:**
- Docker image ≤ 2 GB
- 4 GB RAM / 4 CPUs / 30-min wall-clock at scoring
- `--network=none` at scoring (no external API calls)
- `predict()` called row-by-row; output order must match input

**Personal constraints:**
- Compute: laptop CPU + Google Colab T4 (free tier)
- Time budget: fast — weekend-style focused work
- Goal: realistic Eval target ~0.50–0.55

---

## 2. Architecture

Two **decoupled** prediction heads bundled into a single `model.pkl`:

```
predict(request) →
  ├─ Intent head:     LightGBM classifier on engineered tabular features
  └─ Trajectory head: 2-layer GRU predicting residuals over constant-velocity baseline
```

**Why decoupled rather than joint:**
- Intent is a tabular classification problem; trajectory is a sequence regression problem. Different optimal architectures.
- Decoupled training lets us iterate the intent head on CPU (~5 min) while the GRU trains separately on Colab (~25 min).
- Cleaner story for the README — two independent improvements, each measurable.
- Fewer cross-component bugs; easier to swap one head if it underperforms.

**File layout in the submission repo:**

```
crossing-challenge-submission/
├── predict.py               # Inference entry — grader's contract
├── train_intent.py          # LightGBM training (CPU, ~5 min)
├── train_trajectory.py      # GRU training (Colab GPU, ~25 min)
├── features.py              # Shared featurization (used by intent + at inference)
├── model.pkl                # Bundled: {lgbm, isotonic_calibrator, gru_state, scaler}
├── grade.py                 # Unchanged from starter
├── Dockerfile               # +lightgbm, +torch-cpu
├── requirements.txt
├── tests/
│   ├── test_predict_contract.py
│   ├── test_predict_finite.py
│   ├── test_predict_speed.py
│   ├── test_features_no_nan.py
│   └── test_grade_local.py
├── data/                    # train.parquet, dev.parquet (from starter)
├── docs/superpowers/specs/  # This file
├── README.md                # Submission writeup (rewritten)
└── CLAUDE.md                # Notes on AI-assisted development
```

---

## 3. Intent Head Design (LightGBM)

### Why LightGBM over XGBoost (the baseline)
- Native categorical handling (no one-hot for `time_of_day`, `weather`, `location`)
- Slightly faster training, similar accuracy
- Cleaner calibration out-of-the-box

### Feature set (~30 features)

| Group | Features | Rationale |
|---|---|---|
| Bbox geometry (current) | normalized cx, cy, w, h, aspect | Where is the pedestrian in the frame |
| Bbox dynamics (full 16 frames) | mean/std/last vx, vy, ax, ay, total displacement | Movement pattern over the past second |
| Bbox dynamics (last 4 frames) | recent vx, vy, ax | Most predictive — last ~270 ms |
| Body-pose proxies | aspect ratio mean/last, height-change rate | Bending or stepping forward changes aspect |
| Ego motion | mean/last/max ego_speed, mean/abs-max yaw_rate, ego_available flag | Vehicle slowing → ped more likely to cross |
| Ego-ped relative | relative velocity = ped_vx + ego_speed * pixel_scale | Hidden signal — partially addresses ego-motion distribution shift |
| Scene context (categorical) | time_of_day, weather, location | Cross-rate varies by scene |
| Position priors | distance to image center, distance to bottom edge | Bottom-of-frame peds are on sidewalk → about to step |

### Training setup
- **Data**: `data/train.parquet` (~29k windows)
- **Validation**: 5-fold cross-validation, **grouped by `ped_id`** (no pedestrian leakage across folds)
- **Class imbalance handling**: `class_weight={0: 1, 1: 5}` — boosts ~7% positive rate without ruining log-loss calibration
- **Hyperparameters**: 500 max trees, depth=6, lr=0.05, early stopping on holdout fold
- **Calibration**: isotonic regression fit on out-of-fold predictions, applied at inference
- **Reproducibility**: `random_state=42`, `np.random.seed(42)`

### Expected impact
- Baseline intent BCE: ~0.21 → intent_term ≈ 0.84
- Target intent BCE: ~0.18 → intent_term ≈ 0.72
- Drops the score by ~0.06

### Inference cost
- < 1 ms per request on CPU

---

## 4. Trajectory Head Design (Small GRU)

### Architecture

```
Input: (batch, 16 timesteps, 8 features)
  Per-frame features: cx_norm, cy_norm, w_norm, h_norm,
                      vx_norm, vy_norm, ego_speed, ego_yaw

Linear projection: 8 → 64
GRU: 2 layers, hidden=64, dropout=0.1, unidirectional
Take last hidden state: (batch, 64)
Linear output head: 64 → 8  (4 horizons × dx, dy residuals)

Final prediction:
  bbox_center(t) = constant_velocity_pred(t) + GRU_residual(t)
  bbox_size(t)   = current_bbox_size  (held constant)
```

**Total parameters:** ~50 K. Trains in ~20 min on Colab T4.

### Why predict residuals over constant-velocity instead of raw centers
- Constant velocity already captures the easy linear motion — model only learns the correction
- Residuals are small (~10–30 px), gradients well-conditioned
- Clean fallback: if GRU outputs NaN, we get the constant-velocity baseline back automatically
- Standard trick from PIE/JAAD trajectory literature

### Why hold bbox size constant
- For non-crossing pedestrians (the dominant class), bbox width/height changes over 2 s are small
- Predicting size adds 8 more outputs, more noise, marginal ADE gain
- Defer to "next experiments" in the README

### Training setup
- **Loss**: MSE on (dx, dy) residuals, equally weighted across 4 horizons
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Schedule**: Cosine LR with 5-epoch linear warmup, 50 epochs total
- **Batch size**: 256
- **Train/val split**: video-disjoint (matches the dataset's split)
- **Checkpoint**: save best model by mean val ADE
- **Reproducibility**: `torch.manual_seed(42)`, deterministic CuDNN

### Distribution-shift handling (per the FAQ hint about ego motion)
- The training data is mostly dashcam at vehicle speed; deployment scenario is a slow sidewalk robot
- **Augmentation**: for 30% of training samples, randomly scale `ego_speed_history` by a factor in [0.3, 1.0]
- Teaches the model that low ego-speed is a normal regime
- 5-line code change; called out explicitly in the README

### Expected impact
- Baseline mean ADE on Eval: ~32 px (7.9 / 18.7 / 37.4 / 61.1 across horizons)
- Target mean ADE: ~18–22 px → traj_term drops from 0.64 to ~0.40
- Biggest wins at +1.5 s and +2.0 s, where constant velocity is most wrong

### Inference cost
- ~3 ms per request on CPU (PyTorch)

---

## 5. Pipeline (Step-by-Step Execution Order)

Each step ends with a measurable success criterion AND a git commit. If a step misses target by >20%, stop and diagnose before proceeding.

| # | Step | Where | Time | Success Criterion | Commit Message |
|---|---|---|---|---|---|
| 1 | Setup environment, verify baseline | Laptop | 5 min | `python grade.py` prints ~0.83 | `baseline reproduced, dev score 0.83` |
| 2 | Build `features.py` | Laptop | 30 min | Featurize 1 row → no NaN, expected shape | `feature builder, 30 features` |
| 3 | Train intent (`train_intent.py`) | Laptop | 5 min | Dev BCE ≤ 0.19 | `lightgbm intent, dev BCE 0.18` |
| 4 | Wire intent into `predict.py` | Laptop | 15 min | `python grade.py` prints ~0.78 | `intent integrated, dev score 0.78` |
| 5 | Train trajectory GRU | Colab T4 | 25 min | Val mean ADE ≤ 22 px | `gru trajectory, dev mean ADE 20 px` |
| 6 | Bundle + integrate full pipeline | Laptop | 30 min | `python grade.py` prints ~0.55 | `full pipeline, dev score 0.55` |
| 7 | Docker build + smoke test | Laptop | 30 min | Image ≤ 2 GB; container produces valid CSV | `dockerized, image 1.6 GB` |
| 8 | README + CLAUDE.md | Laptop | 1 hr | `SUBMISSION_TEMPLATE.md` rubric satisfied | `submission writeup` |
| 9 | Push final to GitHub, send link | — | — | — | — |

**Total wall-clock: ~6–10 hours of focused work + ~25 min Colab.**

### Designed-in fallbacks
1. **If Colab fails or GRU underperforms** → swap to Kalman filter trajectory (constant-acceleration model). Still ships, lands ~0.62.
2. **If Docker image exceeds 2 GB** → switch from PyTorch to ONNX Runtime (saves ~600 MB).

---

## 6. Testing & Grader-Contract Safety

The grader is strict about output shape, row order, and finite values. We test the contract aggressively.

### Tests in `tests/`

| Test | What it verifies |
|---|---|
| `test_predict_contract.py` | `predict()` returns dict with all 5 required keys; intent ∈ [0,1]; each bbox is list of 4 floats |
| `test_predict_finite.py` | For 100 random rows from `dev.parquet`, output is fully finite (no NaN/Inf) |
| `test_predict_speed.py` | Mean predict() latency < 50 ms, p99 < 200 ms over 500 rows |
| `test_features_no_nan.py` | Featurize 200 random rows; assert all finite |
| `test_grade_local.py` | End-to-end: load 1k dev rows, predict all, score; assert score < 0.65 (regression gate; Dev baseline is 0.83, our target is ~0.55–0.62) |
| `test_docker_smoke.py` (manual) | `docker run` produces CSV with exact row count + correct columns |

**Test discipline:** run before every commit. Run before pushing. Run inside Docker.

### Defensive code in `predict.py`

```python
def predict(request: dict) -> dict:
    try:
        intent_prob = _predict_intent(request)
        traj = _predict_trajectory(request)
    except Exception:
        return _zero_work_prediction(request)  # never crash mid-eval

    if not np.isfinite(intent_prob):
        intent_prob = 0.5
    intent_prob = float(np.clip(intent_prob, 1e-6, 1.0 - 1e-6))

    out = {"intent": intent_prob}
    for key in HORIZON_KEYS:
        bbox = traj[key]
        bbox = [float(v) if np.isfinite(v) else _center_fallback(request) for v in bbox]
        bbox = [float(np.clip(v, -2000.0, 4000.0)) for v in bbox]  # match grader clamp
        out[key] = bbox
    return out
```

**Three principles:**
1. Never crash mid-row — a single bad request can't blow up the whole eval
2. Match the grader's clamps exactly (`[-2000, 4000]`, `[1e-6, 1-1e-6]`) so dev grading and real grading agree to the digit
3. Zero-work fallback always exists (predict current bbox + class-prior intent)

### Grader-contract checklist (verified before submission)

- [ ] Output dict has exactly 5 keys: `intent`, `bbox_500ms`, `bbox_1000ms`, `bbox_1500ms`, `bbox_2000ms`
- [ ] Each bbox is `[x1, y1, x2, y2]` in pixel coordinates, in that order
- [ ] Row order matches input order (natural — call `predict()` in order)
- [ ] CSV output includes `ped_id` column (grader uses it for row alignment)
- [ ] Docker image ≤ 2 GB (`docker images` after build)
- [ ] No external network calls (verify locally with `docker run --network=none`)
- [ ] `predict()` works without GPU, < 200 ms per request
- [ ] `model.pkl` is NOT unpickled at Docker build time (matches starter's security model)

---

## 7. Explicit YAGNI List

Things we are deliberately **not** building, because they add complexity without proportional score gain or hireability signal:

- ❌ Ensembling multiple GRUs — one model, clean story
- ❌ Bbox size prediction (height/width) — diminishing returns
- ❌ External datasets (raw PIE/JAAD) — disqualifier per challenge rules
- ❌ Test-time augmentation — latency cost too high
- ❌ Hyperparameter sweep — single Colab run, sensible defaults, ship
- ❌ Joint intent+trajectory training — couples gradients, harder to debug
- ❌ Model monitoring / A/B testing infra — wrong scope for a take-home
- ❌ Distributed training — single T4 is plenty for a 50K-param model

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Colab session disconnects mid-training | Medium | Save checkpoints every 5 epochs; resume from latest |
| GRU overfits 29k-row train set | Medium | Dropout 0.1, early stopping on val ADE, weight decay |
| Docker image exceeds 2 GB with PyTorch | Low | Use `torch==2.x+cpu` wheels (no CUDA); fallback to ONNX Runtime |
| `predict()` latency > 200 ms on grader machine | Low | Latency test in CI; current estimate ~5 ms total |
| Intent calibration drift on Eval | Medium | Use isotonic on out-of-fold preds (more robust than Platt scaling) |
| Ego-motion distribution shift hurts trajectory | High | Speed-scaling augmentation in training (covered in §4) |

---

## 9. Submission Deliverables

Per the challenge rules:

- `predict.py` — fixed signature, all defensive code in place
- `Dockerfile` — builds in <10 min, image ≤ 2 GB
- `model.pkl` — trained weights bundle (LGBM + GRU state + calibrator + scaler)
- `README.md` — rewritten following `SUBMISSION_TEMPLATE.md`:
  - Final Dev composite score
  - Approach summary
  - Two or three things that didn't work
  - Where AI tooling helped most (specific examples)
  - Next experiments
  - Reproduction commands
  - External data / pretrained weights (none used)
- `CLAUDE.md` — brief notes on the AI-assisted development workflow (per challenge rules)
- This design doc (`docs/superpowers/specs/2026-04-30-crossing-challenge-design.md`)
- Clean git log with ~9 commits, one per pipeline step

**Final repo:** https://github.com/Sai-Prashanth123/crossing-challenge-submission

**Submission email:** agentic-hiring@gobblecube.ai (with LinkedIn profile)
