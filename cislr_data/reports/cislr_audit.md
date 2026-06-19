# CISLR Dataset Audit

Generated: 2026-06-18T18:37:06.033975+00:00
Dataset root: `cislr_data`

## Totals
- Rows: **7,050**
- Unique glosses: **4,764**
- Unique signers: **0**

## Class Distribution
- Min / max samples per gloss: 1 / 13
- Mean / median samples per gloss: 1.48 / 1
- Singleton glosses: 3,400
- Glosses with >=2 samples: 1,364
- Glosses with >=5 samples: 82
- Glosses with >=10 samples: 3

## Top 20 Signs
- `MONDAY`: 13
- `THURSDAY`: 11
- `INTEREST`: 10
- `PENCIL`: 9
- `FRIDAY`: 9
- `MOTHER`: 8
- `YELLOW`: 8
- `TUESDAY`: 8
- `CHRISTMAS`: 8
- `INCREASE`: 7
- `IRON`: 7
- `LIGHT`: 7
- `ANGRY`: 7
- `BILL`: 7
- `NAME`: 7
- `SUNDAY`: 7
- `INDIA`: 7
- `RUBBER`: 7
- `DATE`: 7
- `WEDNESDAY`: 7

## SAKSHAM Target Overlap
- HELLO: 0
- HELP: 2
- STOP: 3
- YES: 2
- NO: 2
- WATER: 4
- THANK YOU: 6
- PLEASE: 3
- SORRY: 4
- NO_SIGN: 0

## Storage Layout
- Video search root: `cislr_data\CISLR_v1.5-a_videos\CISLR_v1.5-a_videos`
- Indexed MP4 files: 7050
- Metadata: 234.84 KB
- Raw videos on disk: 1.19 GB (7050 files)
- Matched metadata videos: 7050
- Missing metadata videos: 0
- Projected landmark tensors: 606.83 MB
- Recommended working space: 1.78 GB

## Video Format (sampled)
- Sampled videos: 25
- Resolutions: 300x300
- FPS values: 25.0, 30.0
- Frame count range: 50 – 342
- Avg duration (sample): 4.69s

## Pipeline Suitability
- **mediapipe_landmarks**: YES (high confidence) — Dictionary-style cropped signer videos are a strong fit for 2D hand landmarks.
- **sequence_windows**: YES (high confidence) — Use 24-frame padded sequences to match existing SAKSHAM preprocessing.
- **gru_training**: NO (low confidence) — Full 4.7k vocabulary is not realistic for vanilla supervised GRU. Filter to high-count glosses or use transfer/augmentation.
- **onnx_export**: YES (high confidence) — GRU with fixed input [batch, 24, 128] exports cleanly to ONNX Runtime.
- **saksham_runtime**: YES (medium confidence) — Runtime recognizer is not yet wired to ONNX/GRU; integration is a separate Phase 5 task. Live webcam domain shift from dictionary videos remains a validation risk.

## Risks
- Most glosses have fewer than 10 videos; standard supervised multi-class GRU training will be severely data-starved without filtering or augmentation.
- 71% of gloss classes appear to be singletons (1 video). Prototype/one-shot or transfer learning is more realistic than plain GRU softmax.
- Limited signer diversity increases overfitting risk for signer-based splits.

## Strengths
- Word-level RGB videos are compatible with offline MediaPipe Hands extraction.
- SAKSHAM already stores accepted clips as NPZ landmark tensors with left/right hands.
- Existing normalization and GRU preprocessing modules align with CISLR landmark conversion.
- ONNX export path is feasible once a compact GRU classifier is trained on fixed-length tensors.
- Short average clip duration (~4.7s) suits fixed window GRU inference.
- 8 SAKSHAM V1 target glosses have direct CISLR rows for bootstrap validation.
