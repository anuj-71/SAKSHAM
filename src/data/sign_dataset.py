"""
PyTorch Dataset for Sign Language Recognition
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from sklearn.model_selection import GroupShuffleSplit
import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.landmark_processor import LandmarkProcessor


class SignLanguageDataset(Dataset):
    """PyTorch Dataset for Sign Language sequences"""
    
    def __init__(self, 
                 csv_data: pd.DataFrame,
                 dataset_root: str,
                 sign_map: Dict[str, int],
                 max_seq_len: int = 128,
                 use_face: bool = False,
                 use_pose: bool = True,
                 use_hands: bool = True,
                 normalize: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Args:
            csv_data: DataFrame with columns [path, participant_id, sequence_id, sign]
            dataset_root: Root directory of dataset
            sign_map: Dictionary mapping sign names to integer labels
            max_seq_len: Maximum sequence length (for padding/truncation)
            use_face: Include face landmarks
            use_pose: Include pose landmarks
            use_hands: Include hand landmarks
            normalize: Whether to normalize coordinates
            cache_dir: Directory to cache processed tensors
        """
        self.csv_data = csv_data.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.sign_map = sign_map
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize processor
        self.processor = LandmarkProcessor(
            use_face=use_face,
            use_pose=use_pose,
            use_hands=use_hands
        )
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.csv_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
            sequence: Tensor of shape (max_seq_len, feature_dim)
            label: Integer label
            seq_len: Actual sequence length (before padding)
        """
        row = self.csv_data.iloc[idx]
        
        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"{row['sequence_id']}.npz"
            if cache_file.exists():
                cached = np.load(cache_file)
                sequence = torch.from_numpy(cached['sequence'])
                seq_len = int(cached['seq_len'])
                label = self.sign_map[row['sign']]
                return sequence, label, seq_len
        
        # Load and process file
        file_path = self.dataset_root / row['path']
        
        try:
            # Process landmarks
            sequence = self.processor.process_file(
                str(file_path),
                max_len=self.max_seq_len,
                normalize=self.normalize
            )
            
            # Get actual length before padding
            seq_len = min(len(sequence), self.max_seq_len)
            
            # Convert to tensor
            sequence_tensor = torch.from_numpy(sequence).float()
            
            # Get label
            label = self.sign_map[row['sign']]
            
            # Cache if enabled
            if self.cache_dir:
                cache_file = self.cache_dir / f"{row['sequence_id']}.npz"
                np.savez_compressed(
                    cache_file,
                    sequence=sequence,
                    seq_len=seq_len
                )
            
            return sequence_tensor, label, seq_len
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero tensor on error
            sequence_tensor = torch.zeros(self.max_seq_len, self.processor.feature_dim)
            return sequence_tensor, 0, 1
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample"""
        row = self.csv_data.iloc[idx]
        return {
            'path': row['path'],
            'participant_id': row['participant_id'],
            'sequence_id': row['sequence_id'],
            'sign': row['sign'],
            'label': self.sign_map[row['sign']]
        }


def create_participant_splits(csv_path: str,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create participant-aware train/val/test splits
    
    Ensures no participant overlap between splits
    
    Args:
        csv_path: Path to CSV file
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed
        
    Returns:
        train_df, val_df, test_df
    """
    df = pd.read_csv(csv_path)
    
    # Get unique participants
    participants = df['participant_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(participants)
    
    # Split participants
    n_train = int(len(participants) * train_ratio)
    n_val = int(len(participants) * val_ratio)
    
    train_participants = participants[:n_train]
    val_participants = participants[n_train:n_train + n_val]
    test_participants = participants[n_train + n_val:]
    
    # Split data
    train_df = df[df['participant_id'].isin(train_participants)].copy()
    val_df = df[df['participant_id'].isin(val_participants)].copy()
    test_df = df[df['participant_id'].isin(test_participants)].copy()
    
    print(f"Participant split:")
    print(f"  Train: {len(train_participants)} participants, {len(train_df)} samples")
    print(f"  Val:   {len(val_participants)} participants, {len(val_df)} samples")
    print(f"  Test:  {len(test_participants)} participants, {len(test_df)} samples")
    
    return train_df, val_df, test_df


def create_dataloaders(dataset_root: str,
                      csv_path: str,
                      sign_map_path: str,
                      batch_size: int = 32,
                      max_seq_len: int = 128,
                      num_workers: int = 4,
                      cache_dir: Optional[str] = None,
                      use_face: bool = False,
                      use_pose: bool = True,
                      use_hands: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load sign map
    with open(sign_map_path, 'r') as f:
        sign_map = json.load(f)
    
    # Create splits
    train_df, val_df, test_df = create_participant_splits(csv_path)
    
    # Create datasets
    train_dataset = SignLanguageDataset(
        train_df, dataset_root, sign_map, max_seq_len,
        use_face=use_face, use_pose=use_pose, use_hands=use_hands,
        normalize=True, cache_dir=cache_dir
    )
    
    val_dataset = SignLanguageDataset(
        val_df, dataset_root, sign_map, max_seq_len,
        use_face=use_face, use_pose=use_pose, use_hands=use_hands,
        normalize=True, cache_dir=cache_dir
    )
    
    test_dataset = SignLanguageDataset(
        test_df, dataset_root, sign_map, max_seq_len,
        use_face=use_face, use_pose=use_pose, use_hands=use_hands,
        normalize=True, cache_dir=cache_dir
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")
    print(f"  Feature dim: {train_dataset.processor.feature_dim}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    dataset_root = "E:/gessture/SAKSHAM_DATASET"
    csv_path = f"{dataset_root}/train_filtered.csv"
    sign_map_path = f"{dataset_root}/sign_map.json"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=dataset_root,
        csv_path=csv_path,
        sign_map_path=sign_map_path,
        batch_size=8,
        max_seq_len=128,
        num_workers=0,
        cache_dir="E:/gessture/dataset/processed/cache"
    )
    
    # Test loading a batch
    print("\nTesting batch loading...")
    for sequences, labels, seq_lens in train_loader:
        print(f"Batch shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Seq lens shape: {seq_lens.shape}")
        print(f"Sample seq lens: {seq_lens[:5]}")
        break
