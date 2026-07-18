"""
SAKSHAM Dataset Validator
Comprehensive validation of the ISL dataset including file existence, corruption detection,
and data quality checks.
"""

import json
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
from collections import Counter
import hashlib

from src.utils.logger import logger
from configs.config import (
    DATASET_ROOT, TRAIN_CSV, SIGN_MAP_JSON, LANDMARK_FILES_DIR,
    EXPECTED_SIGN_COUNT, EXPECTED_SAMPLE_COUNT_APPROX
)


@dataclass
class ValidationResults:
    """Container for validation results."""
    total_samples: int
    existing_files: int
    missing_files: int
    corrupted_files: int
    duplicate_entries: int
    class_distribution: Dict[str, int]
    missing_file_paths: List[str]
    corrupted_file_paths: List[str]
    duplicate_paths: List[str]
    label_mismatches: List[str]
    sign_map: Dict[str, int]
    participants: Set[str]
    dataset_size_mb: float
    validation_passed: bool
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['participants'] = list(data['participants'])  # Convert set to list
        return data


class DatasetValidator:
    """Validates SAKSHAM dataset integrity and quality."""
    
    def __init__(self):
        self.dataset_root = DATASET_ROOT
        self.train_csv = TRAIN_CSV
        self.sign_map_json = SIGN_MAP_JSON
        self.landmark_dir = LANDMARK_FILES_DIR
        
        self.df = None
        self.sign_map = None
        
    def load_metadata(self) -> None:
        """Load CSV and sign map."""
        logger.info(f"Loading train_filtered.csv from {self.train_csv}")
        self.df = pd.read_csv(self.train_csv)
        logger.success(f"Loaded {len(self.df)} entries from CSV")
        
        logger.info(f"Loading sign_map.json from {self.sign_map_json}")
        with open(self.sign_map_json, 'r') as f:
            self.sign_map = json.load(f)
        logger.success(f"Loaded sign map with {len(self.sign_map)} classes")
    
    def verify_sign_map(self) -> List[str]:
        """Verify that all labels in CSV exist in sign_map."""
        logger.info("Verifying labels match sign_map.json...")
        
        csv_signs = set(self.df['sign'].unique())
        map_signs = set(self.sign_map.keys())
        
        mismatches = []
        
        # Check for signs in CSV not in map
        missing_in_map = csv_signs - map_signs
        if missing_in_map:
            msg = f"Signs in CSV but not in sign_map: {missing_in_map}"
            logger.error(msg)
            mismatches.append(msg)
        
        # Check for signs in map not in CSV
        unused_in_csv = map_signs - csv_signs
        if unused_in_csv:
            msg = f"Signs in sign_map but not used in CSV: {unused_in_csv}"
            logger.warning(msg)
            mismatches.append(msg)
        
        if not mismatches:
            logger.success("All labels match sign_map.json ✓")
        
        return mismatches
    
    def check_file_existence(self) -> Tuple[List[str], List[str]]:
        """Check which parquet files exist and which are missing."""
        logger.header("Checking file existence...")
        
        existing = []
        missing = []
        
        with logger.progress_bar() as progress:
            task = progress.add_task(
                "[cyan]Verifying parquet files...",
                total=len(self.df)
            )
            
            for idx, row in self.df.iterrows():
                file_path = self.dataset_root / row['path']
                
                if file_path.exists():
                    existing.append(str(file_path))
                else:
                    missing.append(row['path'])
                
                progress.update(task, advance=1)
        
        logger.success(f"Found {len(existing)} existing files")
        if missing:
            logger.error(f"Missing {len(missing)} files")
        
        return existing, missing
    
    def check_file_corruption(self, file_paths: List[str]) -> List[str]:
        """Check for corrupted parquet files."""
        logger.header("Checking for corrupted parquet files...")
        
        corrupted = []
        
        with logger.progress_bar() as progress:
            task = progress.add_task(
                "[cyan]Validating parquet integrity...",
                total=len(file_paths)
            )
            
            for file_path in file_paths:
                try:
                    # Try to read the parquet file
                    table = pq.read_table(file_path)
                    
                    # Additional validation: check if it has data
                    if table.num_rows == 0:
                        logger.warning(f"Empty parquet file: {file_path}")
                        corrupted.append(file_path)
                    
                except Exception as e:
                    logger.error(f"Corrupted file {file_path}: {str(e)}")
                    corrupted.append(file_path)
                
                progress.update(task, advance=1)
        
        if corrupted:
            logger.error(f"Found {len(corrupted)} corrupted files")
        else:
            logger.success("No corrupted files detected ✓")
        
        return corrupted
    
    def detect_duplicates(self) -> List[str]:
        """Detect duplicate entries in the CSV."""
        logger.info("Checking for duplicate entries...")
        
        # Check for duplicate paths
        duplicates = self.df[self.df.duplicated(subset=['path'], keep=False)]
        
        if len(duplicates) > 0:
            logger.warning(f"Found {len(duplicates)} duplicate path entries")
            return duplicates['path'].tolist()
        else:
            logger.success("No duplicate entries detected ✓")
            return []
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Calculate class distribution."""
        logger.info("Calculating class distribution...")
        
        distribution = self.df['sign'].value_counts().to_dict()
        
        # Sort by sign name
        distribution = dict(sorted(distribution.items()))
        
        return distribution
    
    def get_participant_stats(self) -> Set[str]:
        """Get unique participants."""
        participants = set(self.df['participant_id'].astype(str).unique())
        logger.info(f"Found {len(participants)} unique participants")
        return participants
    
    def calculate_dataset_size(self, existing_files: List[str]) -> float:
        """Calculate total dataset size in MB."""
        logger.info("Calculating dataset size...")
        
        total_size_bytes = 0
        for file_path in existing_files:
            try:
                total_size_bytes += Path(file_path).stat().st_size
            except Exception as e:
                logger.warning(f"Could not get size of {file_path}: {e}")
        
        size_mb = total_size_bytes / (1024 * 1024)
        logger.info(f"Total dataset size: {size_mb:.2f} MB")
        
        return size_mb
    
    def print_statistics(self, results: ValidationResults) -> None:
        """Print comprehensive statistics."""
        logger.header("=" * 60)
        logger.header("SAKSHAM DATASET VALIDATION REPORT")
        logger.header("=" * 60)
        
        # Overview
        logger.panel(
            f"Total Samples: {results.total_samples}\n"
            f"Expected (approx): ~{EXPECTED_SAMPLE_COUNT_APPROX}\n"
            f"Unique Participants: {len(results.participants)}\n"
            f"Total Dataset Size: {results.dataset_size_mb:.2f} MB\n"
            f"Number of Classes: {len(results.sign_map)}",
            title="📊 Dataset Overview",
            style="cyan"
        )
        
        # File Status
        status_color = "green" if results.missing_files == 0 and results.corrupted_files == 0 else "red"
        logger.panel(
            f"Existing Files: {results.existing_files}\n"
            f"Missing Files: {results.missing_files}\n"
            f"Corrupted Files: {results.corrupted_files}\n"
            f"Duplicate Entries: {results.duplicate_entries}",
            title="📁 File Status",
            style=status_color
        )
        
        # Class Distribution Table
        logger.header("\n📈 Class Distribution")
        class_data = [
            [sign, count, f"{(count/results.total_samples)*100:.2f}%"]
            for sign, count in results.class_distribution.items()
        ]
        logger.table(
            "Sign Classes",
            ["Sign", "Count", "Percentage"],
            class_data
        )
        
        # Validation Status
        if results.validation_passed:
            logger.panel(
                "✓ Dataset validation PASSED\n"
                "All files are present and valid.",
                title="✅ Validation Result",
                style="green"
            )
        else:
            issues = []
            if results.missing_files > 0:
                issues.append(f"• {results.missing_files} missing files")
            if results.corrupted_files > 0:
                issues.append(f"• {results.corrupted_files} corrupted files")
            if results.duplicate_entries > 0:
                issues.append(f"• {results.duplicate_entries} duplicate entries")
            if results.label_mismatches:
                issues.append(f"• Label mismatches detected")
            
            logger.panel(
                "✗ Dataset validation FAILED\n\n" + "\n".join(issues),
                title="❌ Validation Result",
                style="red"
            )
    
    def validate(self) -> ValidationResults:
        """Run complete validation pipeline."""
        logger.panel(
            "Starting SAKSHAM Dataset Validation\n"
            f"Dataset: {self.dataset_root}",
            title="🚀 Dataset Validator",
            style="bold blue"
        )
        
        # Load metadata
        self.load_metadata()
        
        # Verify sign map
        label_mismatches = self.verify_sign_map()
        
        # Check file existence
        existing_files, missing_files = self.check_file_existence()
        
        # Check for corruption
        corrupted_files = self.check_file_corruption(existing_files)
        
        # Detect duplicates
        duplicate_paths = self.detect_duplicates()
        
        # Get statistics
        class_distribution = self.get_class_distribution()
        participants = self.get_participant_stats()
        dataset_size = self.calculate_dataset_size(existing_files)
        
        # Determine if validation passed
        validation_passed = (
            len(missing_files) == 0 and
            len(corrupted_files) == 0 and
            len(label_mismatches) == 0
        )
        
        # Create results object
        results = ValidationResults(
            total_samples=len(self.df),
            existing_files=len(existing_files),
            missing_files=len(missing_files),
            corrupted_files=len(corrupted_files),
            duplicate_entries=len(duplicate_paths),
            class_distribution=class_distribution,
            missing_file_paths=missing_files,
            corrupted_file_paths=corrupted_files,
            duplicate_paths=duplicate_paths,
            label_mismatches=label_mismatches,
            sign_map=self.sign_map,
            participants=participants,
            dataset_size_mb=dataset_size,
            validation_passed=validation_passed
        )
        
        # Print statistics
        self.print_statistics(results)
        
        return results
