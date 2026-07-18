"""
Landmark Processing Module
Processes raw parquet landmark files into normalized tensors
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


class LandmarkProcessor:
    """Process landmark data from parquet files"""
    
    # Landmark counts for each type
    LANDMARK_COUNTS = {
        'face': 468,
        'left_hand': 21,
        'right_hand': 21,
        'pose': 33
    }
    
    def __init__(self, use_face: bool = False, use_pose: bool = True, 
                 use_hands: bool = True):
        """
        Initialize processor
        
        Args:
            use_face: Include face landmarks (468 points)
            use_pose: Include pose landmarks (33 points)
            use_hands: Include hand landmarks (21 + 21 points)
        """
        self.use_face = use_face
        self.use_pose = use_pose
        self.use_hands = use_hands
        
        # Calculate feature dimension
        self.feature_dim = 0
        if use_face:
            self.feature_dim += self.LANDMARK_COUNTS['face'] * 3
        if use_pose:
            self.feature_dim += self.LANDMARK_COUNTS['pose'] * 3
        if use_hands:
            self.feature_dim += (self.LANDMARK_COUNTS['left_hand'] + 
                                self.LANDMARK_COUNTS['right_hand']) * 3
    
    def load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load parquet file and handle NaN values"""
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # Fill NaN values with 0 (undetected landmarks)
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].fillna(0.0)
        
        return df
    
    def extract_landmarks(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and organize landmarks into sequence tensor
        
        Args:
            df: DataFrame with columns [frame, type, landmark_index, x, y, z]
            
        Returns:
            Array of shape (num_frames, feature_dim)
        """
        # Get number of frames
        num_frames = df['frame'].max() + 1
        
        # Initialize output array
        sequence = np.zeros((num_frames, self.feature_dim), dtype=np.float32)
        
        # Process each landmark type
        offset = 0
        
        if self.use_pose:
            pose_df = df[df['type'] == 'pose'].copy()
            if len(pose_df) > 0:
                sequence = self._fill_landmarks(sequence, pose_df, offset, 
                                                self.LANDMARK_COUNTS['pose'])
            offset += self.LANDMARK_COUNTS['pose'] * 3
        
        if self.use_hands:
            # Left hand
            left_df = df[df['type'] == 'left_hand'].copy()
            if len(left_df) > 0:
                sequence = self._fill_landmarks(sequence, left_df, offset,
                                               self.LANDMARK_COUNTS['left_hand'])
            offset += self.LANDMARK_COUNTS['left_hand'] * 3
            
            # Right hand
            right_df = df[df['type'] == 'right_hand'].copy()
            if len(right_df) > 0:
                sequence = self._fill_landmarks(sequence, right_df, offset,
                                               self.LANDMARK_COUNTS['right_hand'])
            offset += self.LANDMARK_COUNTS['right_hand'] * 3
        
        if self.use_face:
            face_df = df[df['type'] == 'face'].copy()
            if len(face_df) > 0:
                sequence = self._fill_landmarks(sequence, face_df, offset,
                                               self.LANDMARK_COUNTS['face'])
            offset += self.LANDMARK_COUNTS['face'] * 3
        
        return sequence
    
    def _fill_landmarks(self, sequence: np.ndarray, df: pd.DataFrame, 
                       offset: int, num_landmarks: int) -> np.ndarray:
        """Fill landmarks into sequence array"""
        for frame_idx in df['frame'].unique():
            frame_data = df[df['frame'] == frame_idx].sort_values('landmark_index')
            
            for idx, row in frame_data.iterrows():
                landmark_idx = row['landmark_index']
                if landmark_idx < num_landmarks:
                    base_idx = offset + landmark_idx * 3
                    sequence[frame_idx, base_idx:base_idx+3] = [row['x'], row['y'], row['z']]
        
        return sequence
    
    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize sequence using reference point (nose/torso center)
        
        Args:
            sequence: Array of shape (num_frames, feature_dim)
            
        Returns:
            Normalized array of same shape
        """
        if len(sequence) == 0:
            return sequence
        
        # Use first pose landmark (nose) as reference if available
        if self.use_pose:
            # Nose is at index 0 in pose landmarks (first 3 values)
            reference = sequence[:, 0:2]  # x, y coordinates
        else:
            # Use mean of all points as reference
            # Reshape to (frames, landmarks, 3)
            reshaped = sequence.reshape(len(sequence), -1, 3)
            reference = reshaped[:, :, :2].mean(axis=1)  # Mean x, y
        
        # Center coordinates
        normalized = sequence.copy()
        num_landmarks = self.feature_dim // 3
        
        for i in range(num_landmarks):
            normalized[:, i*3:i*3+2] -= reference
        
        return normalized
    
    def interpolate_missing_frames(self, sequence: np.ndarray, 
                                   max_len: Optional[int] = None) -> np.ndarray:
        """
        Interpolate or pad sequence to fixed length
        
        Args:
            sequence: Input sequence of shape (num_frames, feature_dim)
            max_len: Target length (if None, no padding/truncation)
            
        Returns:
            Processed sequence
        """
        if max_len is None:
            return sequence
        
        curr_len = len(sequence)
        
        if curr_len == max_len:
            return sequence
        elif curr_len > max_len:
            # Uniformly sample frames
            indices = np.linspace(0, curr_len - 1, max_len).astype(int)
            return sequence[indices]
        else:
            # Pad with zeros or repeat last frame
            padding = np.zeros((max_len - curr_len, self.feature_dim), dtype=np.float32)
            return np.vstack([sequence, padding])
    
    def process_file(self, file_path: str, max_len: Optional[int] = None,
                    normalize: bool = True) -> np.ndarray:
        """
        Complete processing pipeline for a single file
        
        Args:
            file_path: Path to parquet file
            max_len: Target sequence length
            normalize: Whether to normalize coordinates
            
        Returns:
            Processed sequence array
        """
        # Load data
        df = self.load_parquet(file_path)
        
        # Extract landmarks
        sequence = self.extract_landmarks(df)
        
        # Normalize
        if normalize:
            sequence = self.normalize_sequence(sequence)
        
        # Interpolate/pad
        if max_len is not None:
            sequence = self.interpolate_missing_frames(sequence, max_len)
        
        return sequence


def compute_sequence_statistics(dataset_root: str, csv_path: str, 
                                sample_size: int = 1000) -> Dict:
    """
    Compute statistics over dataset sequences
    
    Returns:
        Dict with mean, std, min_len, max_len, median_len
    """
    import random
    
    df = pd.read_csv(csv_path)
    
    # Sample random files
    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df
    
    processor = LandmarkProcessor(use_face=False, use_pose=True, use_hands=True)
    
    lengths = []
    all_values = []
    
    print(f"Computing statistics from {len(sample_df)} samples...")
    
    for idx, row in sample_df.iterrows():
        file_path = Path(dataset_root) / row['path']
        try:
            sequence = processor.process_file(str(file_path), normalize=False)
            lengths.append(len(sequence))
            all_values.extend(sequence.flatten())
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    all_values = np.array(all_values)
    
    stats = {
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'min_len': int(np.min(lengths)),
        'max_len': int(np.max(lengths)),
        'median_len': int(np.median(lengths)),
        'p25_len': int(np.percentile(lengths, 25)),
        'p75_len': int(np.percentile(lengths, 75)),
        'p95_len': int(np.percentile(lengths, 95))
    }
    
    return stats


if __name__ == "__main__":
    # Test processing
    dataset_root = "E:/gessture/SAKSHAM_DATASET"
    csv_path = f"{dataset_root}/train_filtered.csv"
    
    # Compute statistics
    stats = compute_sequence_statistics(dataset_root, csv_path, sample_size=500)
    
    print("\nDataset Statistics:")
    print(f"  Value mean: {stats['mean']:.4f}")
    print(f"  Value std: {stats['std']:.4f}")
    print(f"  Sequence lengths:")
    print(f"    Min: {stats['min_len']}")
    print(f"    P25: {stats['p25_len']}")
    print(f"    Median: {stats['median_len']}")
    print(f"    P75: {stats['p75_len']}")
    print(f"    P95: {stats['p95_len']}")
    print(f"    Max: {stats['max_len']}")
    
    # Save statistics
    output_path = "E:/gessture/reports/preprocessing_stats.json"
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {output_path}")
