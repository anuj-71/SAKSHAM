"""
src/sign_language/normalization.py
==================================
4-Layer Invariant Normalization for SAKSHAM Sign Language Pipeline.

Transforms raw MediaPipe Holistic landmarks into a scale-invariant,
translation-invariant, and body-centered coordinate space.

Layers:
  1. Translation Invariance: Body-Centric Mid-Shoulder Anchor
  2. Scale Invariance: Shoulder-Width Distance Normalization
  3. Hand-Shape Invariance: Wrist-Relative Finger Coordinates
  4. Kinematic Motion Dynamics: Bounded Trajectories

Output shape: (30, 225) or (225,) float32 array:
  - [0..62]   : Left Hand (21 landmarks x 3 coords, wrist-relative shape + body-relative wrist)
  - [63..125] : Right Hand (21 landmarks x 3 coords, wrist-relative shape + body-relative wrist)
  - [126..224]: Pose (33 landmarks x 3 coords, shoulder-anchored & shoulder-scaled)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

FEATURE_DIM = 225
SEQ_LEN = 30


def normalize_holistic_frame(
    left_hand_lms: Optional[np.ndarray],  # Shape (21, 3) or None
    right_hand_lms: Optional[np.ndarray], # Shape (21, 3) or None
    pose_lms: Optional[np.ndarray],       # Shape (33, 3) or None
) -> np.ndarray:
    """
    Normalizes a single frame of (left_hand, right_hand, pose) landmarks
    into a (225,) float32 invariant feature vector.
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    # 1. Compute Body Anchor (Mid-Shoulder) & Shoulder Scale
    anchor = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    scale = 0.35  # Default realistic shoulder scale in normalized frame

    has_pose = pose_lms is not None and len(pose_lms) >= 33 and np.any(pose_lms)
    if has_pose:
        left_shoulder = pose_lms[11]
        right_shoulder = pose_lms[12]

        # If both shoulders have non-zero coordinates
        if np.any(left_shoulder[:2]) and np.any(right_shoulder[:2]):
            anchor = (left_shoulder + right_shoulder) / 2.0
            dist = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
            if dist > 0.05:
                scale = float(dist)
        elif np.any(pose_lms[0][:2]):  # Fallback: nose
            anchor = pose_lms[0].copy()

    # 2. Normalize Pose Landmarks (Indices 126..224)
    if has_pose:
        pose_norm = (pose_lms - anchor) / scale
        features[126:225] = pose_norm.flatten()

    # 3. Normalize Left Hand (Indices 0..62)
    has_left = left_hand_lms is not None and len(left_hand_lms) >= 21 and np.any(left_hand_lms)
    if has_left:
        left_wrist = left_hand_lms[0]
        # Hand scale: distance from wrist (0) to MCP middle finger (9)
        hand_dist = np.linalg.norm(left_hand_lms[0][:2] - left_hand_lms[9][:2])
        hand_scale = float(hand_dist) if hand_dist > 0.02 else 0.15

        # Wrist position in body-relative coordinates
        wrist_body_pos = (left_wrist - anchor) / scale

        # Finger positions relative to wrist
        fingers_shape = (left_hand_lms - left_wrist) / hand_scale
        # Overwrite landmark 0 with body-relative wrist position so trajectory is preserved
        fingers_shape[0] = wrist_body_pos

        features[0:63] = fingers_shape.flatten()

    # 4. Normalize Right Hand (Indices 63..125)
    has_right = right_hand_lms is not None and len(right_hand_lms) >= 21 and np.any(right_hand_lms)
    if has_right:
        right_wrist = right_hand_lms[0]
        hand_dist = np.linalg.norm(right_hand_lms[0][:2] - right_hand_lms[9][:2])
        hand_scale = float(hand_dist) if hand_dist > 0.02 else 0.15

        wrist_body_pos = (right_wrist - anchor) / scale
        fingers_shape = (right_hand_lms - right_wrist) / hand_scale
        fingers_shape[0] = wrist_body_pos

        features[63:126] = fingers_shape.flatten()

    return features


def normalize_raw_225_sequence(seq_225: np.ndarray) -> np.ndarray:
    """
    Takes an unnormalized sequence of shape (T, 225) and applies invariant normalization
    to all frames in place.
    """
    T = seq_225.shape[0]
    out_seq = np.zeros((T, FEATURE_DIM), dtype=np.float32)

    for t in range(T):
        frame = seq_225[t]
        left_flat = frame[0:63]
        right_flat = frame[63:126]
        pose_flat = frame[126:225]

        left_lms = left_flat.reshape((21, 3)) if np.any(left_flat) else None
        right_lms = right_flat.reshape((21, 3)) if np.any(right_flat) else None
        pose_lms = pose_flat.reshape((33, 3)) if np.any(pose_flat) else None

        out_seq[t] = normalize_holistic_frame(left_lms, right_lms, pose_lms)

    return out_seq
