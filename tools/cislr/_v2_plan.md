# V2 Training Plan: Root Cause Analysis & Implementation

## 1. Root Causes of 10.5% Accuracy

### 1a. Feature Collapse: 84.4% of Samples Have Zero Hands

**The dominant root cause** is that MediaPipe failed to detect hands on 84.4% of the V1
training set. The GRU receives mostly-zero (128-dim) input frames, forcing it to
memorize noise or bias.

| Hand Presence | V1 Samples | Percentage |
|---------------|-----------|-----------|
| Both hands (>50% frames) | 14 | 4.4% |
| One hand (>50% frames) | 36 | 11.2% |
| Neither hand (sporadic/none) | 271 | 84.4% |

**Why MediaPipe failed on CISLR videos:**
- CISLR frames are cropped to the signer's upper body at ~320×240 resolution.
- MediaPipe Hands expects at least 200×200 px ROI per hand; the downscaled hands
  in a 320×240 crop often fall below the detection confidence threshold.
- The extraction pipeline used default MediaPipe `min_detection_confidence=0.5`.
  Many frames have hands too small or blurred for the default threshold.

### 1b. Extreme Data Scarcity

Even among frames *with* detections, the per-class sample count is critically low:

| Metric | Value |
|--------|-------|
| Classes with >=8 samples | 8 / 50 |
| Mean samples per class | 6.4 |
| Typical train/val/test split | 4 / 1 / 1 |

With 4 training samples, the model has no hope of learning a generalizable
decision boundary, regardless of architecture.

### 1c. Temporal Truncation Wastes Signal

- Median video length: **105 frames**
- Current `SEQ_LEN=24` retains only 23% of temporal content on average.
- No videos are under 48 frames; a fixed window of 64–96 would preserve more
  signal without exceeeding GPU memory budgets.

---

## 2. Dataset Issues

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| 84% dead frames (no hand detections) | Model learns noise, not signs | Re-extract with tuned MediaPipe params |
| 5–6 samples/class average | No generalization possible | Reduce V2 vocab size + collect more |
| All frames truncated (min=50) | Lost temporal structure | Increase SEQ_LEN to 64 or 96 |
| 50 classes is too many for data | 2% random baseline, data too spread | Reduce to 10–15 high-confidence classes |

---

## 3. Model Issues

| Issue | Detail |
|-------|--------|
| Vanilla GRU | Simple, but no attention over time. The model sees 24 frames of (mostly zero) features and must compress everything into one hidden state. |
| No regularization beyond dropout=0.3 | Missing weight decay, label smoothing, gradient clipping, mixup |
| Fixed 24-frame window | Suboptimal for median 105-frame videos |
| Single linear classification head | No intermediate bottleneck or projection |

---

## 4. Recommended V2 Architecture

### 4a. Primary Recommendation: Transformer Encoder

```
Input:  [B, T, 128]  (T = 64 or 96)
  └─→ Linear Projection: 128 → d_model (128)
  └─→ Positional Encoding (learnable)
  └─→ TransformerEncoder × 4 layers
       └─ Multi-Head Self-Attention (4 heads)
       └─ Feed-Forward (d_model × 2)
       └─ LayerNorm + Dropout(0.2)
  └─→ Mean Pool over time
  └─→ Dropout(0.3)
  └─→ Linear(d_model → num_classes)
```

**Why Transformer > GRU for this data:**
- Self-attention can focus on the few frames where hands *are* detected and
  ignore the dead frames — GRU must compress everything into a single hidden
  vector.
- Mean pooling over time is more robust to variable-length "bursts" of
  detections than the final hidden state.
- Transformers handle the sparse hand-presence pattern naturally.

### 4b. Conservative Alternative: Bidirectional GRU + Attention

```
Input:  [B, T, 128]
  └─→ BiGRU (2 layers, hidden=128)
  └─→ Attention-weighted sum over time
  └─→ BatchNorm + Dropout(0.3)
  └─→ Linear(hidden*2 → num_classes)
```

### 4c. Ablation Baseline: Strided GRU at SEQ_LEN=64

Keep the current 2-layer GRU but:
- Increase SEQ_LEN to 64
- Add gradient clipping (max_norm=1.0)
- Add weight decay (1e-4)
- Use OneCycleLR scheduler

---

## 5. V2 Dataset Pipeline

### Step 1: Filter to high-presence glosses

Scan all NPZ files and keep only glosses where:
- At least 60% of frames have `left_present | right_present > 0.5`
- At least 1 hand is present in >50% of frames overall

This will filter out the ~84% dead samples, leaving a smaller but higher-quality
corpus.

### Step 2: Re-extract with lowered MediaPipe threshold

Run `extract_landmarks.py` with:
- `--min-detection-confidence 0.3`
- `--min-tracking-confidence 0.3`
- Target the ~30% of frames that had sporadic detections

### Step 3: Stratified vocabulary reduction

Select **10–15 glosses** that meet ALL criteria:
- >= 20 total samples after Step 1 filter
- At least 2 unique signers if signer metadata available
- Practical for real-world usage (THANK YOU, YES, NO, HELP, WATER, MONDAY,
  MOTHER, GOOD, BAD, NAME, INDIA, ANGRY, WASH, etc.)

### Step 4: Augmentation techniques for landmark sequences

| Technique | Description | Implementation |
|-----------|-------------|---------------|
| Temporal masking | Randomly zero-out 10–20% of frames to simulate detection dropout | `random.sample(range(T), k=int(T*0.15))` |
| Time warping | Slight random scaling of frame indices (±10%) | `torch.nn.functional.interpolate(scale_factor=1+noise)` |
| Gaussian noise | Add N(0, 0.01) to landmark coordinates | `features += torch.randn_like(features) * 0.01` |
| Hand dropout | Randomly drop left/right hand presence for a clip to simulate one-hand | Bernoulli(0.1) per sample |
| Mixup | Linear interpolation between two samples of same class | `lambda*x_i + (1-lambda)*x_j` |

### Step 5: Dynamic sequence length

Instead of a fixed truncation, use:
- SEQ_LEN = min(max_frames, 96)
- Pad shorter sequences, truncate longer ones
- Store `attention_mask` for Transformer (1 for real frames, 0 for padding)

---

## 6. V2 Training Pipeline

```
1. Data Loading
   ├─ CSV manifest (filtered high-presence NPZ files)
   ├─ On-the-fly: normalize → pad/truncate → augment
   └─ BatchSampler: stratified by class (ensure every batch has ≥1 of each class)

2. Model
   ├─ TransformerEncoder (4 layers, 4 heads, d_model=128)
   └─ Classification head

3. Training
   ├─ Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
   ├─ Scheduler: CosineAnnealingLR with warmup (5 epochs)
   ├─ Loss: LabelSmoothingCrossEntropy (ε=0.1)
   ├─ Gradient clipping: max_norm=1.0
   ├─ Early stopping: patience=20, monitor val_loss
   └─ Max epochs: 200

4. Evaluation
   ├─ Test set per-class accuracy
   ├─ Confusion matrix
   ├─ Precision / Recall / F1 per class
   └─ ROC-AUC (one-vs-rest)
```

---

## 7. Sample Count vs. Accuracy Estimates

Estimates assume 70/15/15 split with Transformer encoder + augmentations,
and that we solve the hand-presence problem (>=60% of frames have hand data).

| Target Accuracy | Samples/Class Required | Total Dataset (10 classes) | Confidence |
|----------------|----------------------|---------------------------|-----------|
| 50% | ~15–20 | 150–200 | High — achievable with filtered data + augmentation |
| 70% | ~30–50 | 300–500 | Medium — depends on class separability |
| 85% | ~80+ | 800+ | Low — requires near-perfect hand detection + more data |

**Current status:** With the existing dataset, after filtering to high-presence
samples, we may have ~5–10 classes with 10–20 usable samples each. That puts us
in the **50% accuracy range** if we solve the hand-presence problem.

---

## 8. Concrete Implementation Steps (Priority Order)

### Must-do (V2.0):
1. **Audit hand presence per gloss** — identify which glosses have usable data
2. **Filter vocabulary** to 10–15 glosses with ≥10 high-presence samples each
3. **Increase SEQ_LEN** from 24 → 64 (matches p90 frame count, no truncation)
4. **Add weight decay + gradient clipping** to current GRU
5. **Add temporal masking augmentation**
6. **Re-train and evaluate**

### Should-do (V2.1):
7. **Implement Transformer encoder** as the primary architecture
8. **Add mixup augmentation**
9. **Add label smoothing**
10. **Export ONNX with dynamic sequence length**

### Nice-to-have (V2.2):
11. **Re-extract landmarks** with lowered MediaPipe thresholds
12. **Collect more data** (target 30+ samples per class)
13. **Multi-signer validation split** if signer metadata available

---

## 9. Data Quality Gate

Before any V2 training, run this gate check:

```python
# Must pass ALL checks to proceed:
checks = {
    "High-presence glosses >= 10": count_glosses(min_presence=0.6) >= 10,
    "Samples per class >= 10": all(counts >= 10),
    "Test set accuracy baseline > 0.25": baseline_accuracy > 0.25,
}
```

If gate fails, do NOT train — fix data quality first.
