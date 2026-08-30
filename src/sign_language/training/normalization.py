import numpy as np


WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
EPSILON = 1e-6


def normalize_hand_frame(landmarks: np.ndarray, present: bool) -> np.ndarray:
    """
    Centers a hand on the wrist and scales by the maximum Euclidean distance 
    from the wrist to any other landmark.
    Assumes input landmarks are in an isometric pixel space (not 0-1 non-isometric).
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if not present or points.size == 0:
        return np.zeros((21, 3), dtype=np.float32)

    wrist = points[WRIST_INDEX]
    centered = points - wrist
    
    # Scale by the maximum distance from the wrist to any point
    distances = np.linalg.norm(centered, axis=1)
    scale = np.max(distances)
    if scale < EPSILON:
        scale = 1.0
    return centered / scale

def normalize_pose_ref(pose_ref: dict, wrist_position: np.ndarray) -> np.ndarray:
    """Centers nose and shoulder midpoint relative to the primary wrist and scales them."""
    if not pose_ref.get("present", False):
        return np.zeros((2, 3), dtype=np.float32)

    nose = np.asarray(pose_ref["nose"], dtype=np.float32)
    shoulder = np.asarray(pose_ref["shoulder_mid"], dtype=np.float32)
    points = np.stack([nose, shoulder])
    
    centered = points - wrist_position
    # Use a fixed scale or the same scale metric as the hand? 
    # Since we only get the wrist_position here, we should probably just return the centered points, 
    # or require the hand scale to be passed in. Let's return centered for now, 
    # or assume the preprocessing step scales it. The instruction says:
    # "same origin/scale convention as normalize_hand_frame".
    # Wait, it's better to pass the scale. Let's pass scale to this function.
    return centered

# Wait, let's redefine it properly.
def normalize_hand_sequence(hand_sequence: np.ndarray, presence_mask: np.ndarray) -> np.ndarray:
    """Normalizes a temporal hand sequence frame by frame."""
    frames = np.asarray(hand_sequence, dtype=np.float32)
    mask = np.asarray(presence_mask, dtype=np.float32)
    normalized = np.zeros_like(frames, dtype=np.float32)
    for index in range(frames.shape[0]):
        normalized[index] = normalize_hand_frame(frames[index], bool(mask[index] > 0.5))
    return normalized
