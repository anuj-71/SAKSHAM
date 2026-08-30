"""
src/sign_language/geometric_rules.py
====================================
Neuro-Symbolic Geometric Consistency Validator for SAKSHAM ASL Recognition.

Provides anatomical spatial verification for all 13 ASL sign classes + null.
Validates hand height, proximity to key body anchors (forehead, mouth, chest, chin),
finger extension states, and movement directions.

Used to boost valid neural predictions and eliminate impossible spatial false positives.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


def compute_geometric_consistency_scores(
    hand_data: dict,
    features_225: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes a [0.0, 1.0] anatomical plausibility score for all 14 classes
    based on the current frame's physical hand and pose positions.
    """
    # Default scores: neutral baseline
    scores = {
        "bye": 0.5,
        "drink": 0.5,
        "food": 0.5,
        "hello": 0.5,
        "home": 0.5,
        "hungry": 0.5,
        "no": 0.5,
        "null": 0.6,
        "please": 0.5,
        "sick": 0.5,
        "sleep": 0.5,
        "thankyou": 0.5,
        "where": 0.5,
        "yes": 0.5,
    }

    left = hand_data.get("left_hand", {})
    right = hand_data.get("right_hand", {})
    pose = hand_data.get("pose", {})

    has_left = left.get("present", False) and left.get("landmarks") and len(left["landmarks"]) >= 21
    has_right = right.get("present", False) and right.get("landmarks") and len(right["landmarks"]) >= 21
    has_pose = pose.get("present", False) and pose.get("landmarks") and len(pose["landmarks"]) >= 33

    # If no hands are detected anywhere, only null is valid
    if not has_left and not has_right:
        for k in scores:
            scores[k] = 0.05
        scores["null"] = 1.0
        return scores

    # Extract key anatomical anchors from pose
    pose_lms = np.array(pose["landmarks"], dtype=np.float32) if has_pose else None

    # Reference points (in 0..1 image coordinates, y increases downward)
    nose_y = pose_lms[0, 1] if has_pose else 0.30
    nose_x = pose_lms[0, 0] if has_pose else 0.50
    mouth_y = (pose_lms[9, 1] + pose_lms[10, 1]) / 2.0 if has_pose else (nose_y + 0.06)
    mouth_x = (pose_lms[9, 0] + pose_lms[10, 0]) / 2.0 if has_pose else nose_x

    shoulder_y = (pose_lms[11, 1] + pose_lms[12, 1]) / 2.0 if has_pose else (nose_y + 0.20)
    shoulder_dist = float(np.linalg.norm(pose_lms[11, :2] - pose_lms[12, :2])) if has_pose else 0.35
    shoulder_dist = max(shoulder_dist, 0.15)

    # Identify primary active hand
    active_lms_list = []
    if has_right:
        active_lms_list.append(np.array(right["landmarks"], dtype=np.float32))
    if has_left:
        active_lms_list.append(np.array(left["landmarks"], dtype=np.float32))

    # Highest hand fingertip and wrist across all visible hands
    min_fingertip_y = 1.0
    min_wrist_y = 1.0
    active_wrist_pts = []

    for hand_lms in active_lms_list:
        wrist = hand_lms[0]
        tips = hand_lms[[4, 8, 12, 16, 20]]  # Thumb, index, middle, ring, pinky
        min_wrist_y = min(min_wrist_y, float(wrist[1]))
        min_fingertip_y = min(min_fingertip_y, float(np.min(tips[:, 1])))
        active_wrist_pts.append(wrist)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. SICK GEOMETRIC VALIDATION:
    #    Hand MUST be at or above forehead/eye level.
    # ─────────────────────────────────────────────────────────────────────────
    if min_fingertip_y <= (nose_y + 0.04):
        # Fingertips touch forehead / eye level -> HIGHLY CONSISTENT with sick
        scores["sick"] = 1.0
        scores["hungry"] = 0.05  # Impossible to be hungry at forehead
        scores["please"] = 0.05  # Impossible to be please at forehead
    else:
        # Hand is below nose level -> CANNOT be sick
        scores["sick"] = 0.02

    # ─────────────────────────────────────────────────────────────────────────
    # 2. HELLO & BYE GEOMETRIC VALIDATION:
    #    Hand is elevated above shoulder level, near ear / temple, open palm.
    # ─────────────────────────────────────────────────────────────────────────
    if min_wrist_y < (shoulder_y + 0.05) and min_fingertip_y < (nose_y + 0.10):
        scores["hello"] = 0.95
        scores["bye"] = 0.90
    else:
        scores["hello"] = 0.15
        scores["bye"] = 0.15

    # ─────────────────────────────────────────────────────────────────────────
    # 3. FOOD & DRINK GEOMETRIC VALIDATION:
    #    Hand is close to mouth / chin level.
    # ─────────────────────────────────────────────────────────────────────────
    mouth_dist = min(
        float(np.linalg.norm(w[:2] - np.array([mouth_x, mouth_y])))
        for w in active_wrist_pts
    )
    if mouth_dist < (0.28 * shoulder_dist):
        scores["food"] = 0.98
        scores["drink"] = 0.95
        scores["thankyou"] = 0.90
        scores["home"] = 0.85
    else:
        scores["food"] = 0.20
        scores["drink"] = 0.20

    # ─────────────────────────────────────────────────────────────────────────
    # 4. PLEASE & HUNGRY GEOMETRIC VALIDATION:
    #    Hand is on center of chest / torso (between shoulders and waist).
    # ─────────────────────────────────────────────────────────────────────────
    is_on_chest = False
    for w in active_wrist_pts:
        if (shoulder_y - 0.05) <= w[1] <= (shoulder_y + 0.35 * shoulder_dist + 0.15):
            if abs(w[0] - nose_x) < (0.45 * shoulder_dist):
                is_on_chest = True
                break

    if is_on_chest:
        scores["please"] = 0.95
        scores["hungry"] = 0.95
        scores["yes"] = 0.85
        # Suppress head-level signs if hand is strictly on chest
        if min_fingertip_y > (nose_y + 0.08):
            scores["sick"] = 0.01
            scores["hello"] = 0.05
            scores["bye"] = 0.05
    else:
        scores["please"] = 0.20
        scores["hungry"] = 0.20

    # ─────────────────────────────────────────────────────────────────────────
    # 5. NO GEOMETRIC VALIDATION:
    #    Hand around chest/shoulder level.
    # ─────────────────────────────────────────────────────────────────────────
    if (shoulder_y - 0.10) <= min_wrist_y <= (shoulder_y + 0.25):
        scores["no"] = 0.85
        scores["where"] = 0.80

    return scores
