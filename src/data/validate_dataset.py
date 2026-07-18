"""
Dataset Validation Script
Validates integrity of SAKSHAM dataset including full parquet file checks
"""

import pandas as pd
import pyarrow.parquet as pq
import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import sys

class DatasetValidator:
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.csv_path = self.dataset_root / "train_filtered.csv"
        self.sign_map_path = self.dataset_root / "sign_map.json"
        self.landmark_dir = self.dataset_root / "train_landmark_files"
        
        self.results = {
            "total_samples": 0,
            "missing_files": [],
            "corrupted_files": [],
            "schema_issues": [],
            "valid_files": 0,
            "duplicate_entries": [],
            "label_mismatches": [],
            "class_distribution": {},
            "participant_distribution": {},
        }
    
    def load_metadata(self):
        """Load CSV and sign map"""
        print("📂 Loading metadata files...")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        
        if not self.sign_map_path.exists():
            raise FileNotFoundError(f"Sign map not found: {self.sign_map_path}")
        
        self.df = pd.read_csv(self.csv_path)
        with open(self.sign_map_path, 'r') as f:
            self.sign_map = json.load(f)
        
        self.results["total_samples"] = len(self.df)
        print(f"✓ Loaded {len(self.df)} samples from CSV")
        print(f"✓ Loaded {len(self.sign_map)} sign classes from sign_map.json")
    
    def check_duplicates(self):
        """Check for duplicate sequence IDs"""
        print("\n🔍 Checking for duplicate entries...")
        
        duplicate_sequences = self.df[self.df.duplicated(subset=['sequence_id'], keep=False)]
        if len(duplicate_sequences) > 0:
            self.results["duplicate_entries"] = duplicate_sequences['sequence_id'].tolist()
            print(f"⚠️  Found {len(duplicate_sequences)} duplicate sequence IDs")
        else:
            print("✓ No duplicate entries found")
    
    def verify_labels(self):
        """Verify all labels exist in sign_map"""
        print("\n🏷️  Verifying labels...")
        
        csv_labels = set(self.df['sign'].unique())
        map_labels = set(self.sign_map.keys())
        
        missing_in_map = csv_labels - map_labels
        if missing_in_map:
            self.results["label_mismatches"] = list(missing_in_map)
            print(f"⚠️  Labels in CSV but not in sign_map: {missing_in_map}")
        else:
            print("✓ All labels match sign_map.json")
        
        # Class distribution
        self.results["class_distribution"] = self.df['sign'].value_counts().to_dict()
    
    def validate_parquet_files(self):
        """Validate every parquet file - check existence, readability, and schema"""
        print("\n🗂️  Validating parquet file integrity...")
        print("   This may take a few minutes...\n")
        
        # Expected columns for Kaggle landmark format (long format)
        expected_columns = ['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z']
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Validating parquet files"):
            file_path = self.dataset_root / row['path']
            
            # Check file existence
            if not file_path.exists():
                self.results["missing_files"].append({
                    "path": row['path'],
                    "sequence_id": row['sequence_id'],
                    "sign": row['sign']
                })
                continue
            
            # Try to read and validate parquet file
            try:
                table = pq.read_table(file_path)
                df_parquet = table.to_pandas()
                
                # Check if dataframe is empty
                if len(df_parquet) == 0:
                    self.results["corrupted_files"].append({
                        "path": row['path'],
                        "error": "Empty parquet file",
                        "sequence_id": row['sequence_id']
                    })
                    continue
                
                # Check for expected columns
                columns = set(df_parquet.columns.tolist())
                required_cols = set(expected_columns)
                
                if not required_cols.issubset(columns):
                    missing_cols = required_cols - columns
                    self.results["schema_issues"].append({
                        "path": row['path'],
                        "error": f"Missing required columns: {missing_cols}",
                        "columns": list(columns)
                    })
                    continue
                
                # Check for valid landmark types
                if 'type' in df_parquet.columns:
                    landmark_types = df_parquet['type'].unique()
                    expected_types = {'face', 'left_hand', 'right_hand', 'pose'}
                    if not any(t in expected_types for t in landmark_types):
                        self.results["schema_issues"].append({
                            "path": row['path'],
                            "error": f"Invalid landmark types: {landmark_types}",
                            "sequence_id": row['sequence_id']
                        })
                        continue
                
                # Validation passed
                self.results["valid_files"] += 1
                
            except Exception as e:
                self.results["corrupted_files"].append({
                    "path": row['path'],
                    "error": str(e),
                    "sequence_id": row['sequence_id']
                })
    
    def compute_statistics(self):
        """Compute dataset statistics"""
        print("\n📊 Computing statistics...")
        
        # Participant distribution
        self.results["participant_distribution"] = self.df['participant_id'].value_counts().to_dict()
        print(f"✓ Found {len(self.results['participant_distribution'])} unique participants")
    
    def generate_report(self, output_path: str):
        """Generate markdown report"""
        print(f"\n📝 Generating report: {output_path}")
        
        report = []
        report.append("# SAKSHAM Dataset Validation Report\n")
        report.append(f"**Generated:** {pd.Timestamp.now()}\n")
        report.append("---\n\n")
        
        # Summary
        report.append("## Summary\n\n")
        report.append(f"- **Total Samples:** {self.results['total_samples']}\n")
        report.append(f"- **Valid Files:** {self.results['valid_files']}\n")
        report.append(f"- **Missing Files:** {len(self.results['missing_files'])}\n")
        report.append(f"- **Corrupted Files:** {len(self.results['corrupted_files'])}\n")
        report.append(f"- **Schema Issues:** {len(self.results['schema_issues'])}\n")
        report.append(f"- **Duplicate Entries:** {len(self.results['duplicate_entries'])}\n")
        report.append(f"- **Unique Participants:** {len(self.results['participant_distribution'])}\n")
        report.append(f"- **Sign Classes:** {len(self.sign_map)}\n\n")
        
        # Integrity Status
        total_issues = (len(self.results['missing_files']) + 
                       len(self.results['corrupted_files']) + 
                       len(self.results['schema_issues']))
        
        if total_issues == 0:
            report.append("### ✅ DATASET INTEGRITY: PASSED\n\n")
            report.append("All parquet files validated successfully. Dataset is ready for training.\n\n")
        else:
            report.append("### ⚠️  DATASET INTEGRITY: ISSUES FOUND\n\n")
            report.append(f"Found {total_issues} problematic files. Review details below.\n\n")
        
        # Class Distribution
        report.append("## Class Distribution\n\n")
        report.append("| Sign | Count | Percentage |\n")
        report.append("|------|-------|------------|\n")
        for sign, count in sorted(self.results['class_distribution'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / self.results['total_samples']) * 100
            report.append(f"| {sign} | {count} | {pct:.2f}% |\n")
        
        # Participant Distribution
        report.append("\n## Participant Distribution\n\n")
        report.append(f"Total unique participants: {len(self.results['participant_distribution'])}\n\n")
        report.append("Top 10 participants by sample count:\n\n")
        report.append("| Participant ID | Sample Count |\n")
        report.append("|----------------|-------------|\n")
        top_participants = sorted(self.results['participant_distribution'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
        for pid, count in top_participants:
            report.append(f"| {pid} | {count} |\n")
        
        # Missing Files
        if self.results['missing_files']:
            report.append("\n## Missing Files\n\n")
            report.append(f"Total: {len(self.results['missing_files'])}\n\n")
            if len(self.results['missing_files']) <= 20:
                for item in self.results['missing_files']:
                    report.append(f"- `{item['path']}` (sign: {item['sign']}, seq: {item['sequence_id']})\n")
            else:
                for item in self.results['missing_files'][:20]:
                    report.append(f"- `{item['path']}` (sign: {item['sign']})\n")
                report.append(f"\n... and {len(self.results['missing_files']) - 20} more\n")
        
        # Corrupted Files
        if self.results['corrupted_files']:
            report.append("\n## Corrupted Files\n\n")
            report.append(f"Total: {len(self.results['corrupted_files'])}\n\n")
            if len(self.results['corrupted_files']) <= 20:
                for item in self.results['corrupted_files']:
                    report.append(f"- `{item['path']}`\n")
                    report.append(f"  - Error: {item['error']}\n")
            else:
                for item in self.results['corrupted_files'][:20]:
                    report.append(f"- `{item['path']}`: {item['error']}\n")
                report.append(f"\n... and {len(self.results['corrupted_files']) - 20} more\n")
        
        # Schema Issues
        if self.results['schema_issues']:
            report.append("\n## Schema Issues\n\n")
            report.append(f"Total: {len(self.results['schema_issues'])}\n\n")
            for item in self.results['schema_issues'][:10]:
                report.append(f"- `{item['path']}`\n")
                report.append(f"  - Error: {item['error']}\n")
        
        # Duplicates
        if self.results['duplicate_entries']:
            report.append("\n## Duplicate Entries\n\n")
            report.append(f"Found {len(self.results['duplicate_entries'])} duplicate sequence IDs:\n\n")
            for seq_id in self.results['duplicate_entries'][:20]:
                report.append(f"- {seq_id}\n")
        
        # Sign Map
        report.append("\n## Sign Map\n\n")
        report.append("| Sign | Label Index |\n")
        report.append("|------|-------------|\n")
        for sign, idx in sorted(self.sign_map.items(), key=lambda x: x[1]):
            report.append(f"| {sign} | {idx} |\n")
        
        report.append("\n---\n")
        report.append("\n## Next Steps\n\n")
        
        if total_issues == 0:
            report.append("✅ Dataset validation complete with no issues.\n\n")
            report.append("**Ready to proceed with:**\n")
            report.append("1. Preprocessing pipeline implementation\n")
            report.append("2. PyTorch Dataset/DataLoader creation\n")
            report.append("3. Model architecture development\n")
            report.append("4. Training pipeline execution\n")
        else:
            report.append("⚠️  Address the issues found before training:\n\n")
            report.append("1. Remove or fix corrupted files\n")
            report.append("2. Verify missing file paths\n")
            report.append("3. Re-run validation\n")
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"✓ Report saved to {output_path}")
    
    def run(self, output_path: str):
        """Run full validation pipeline"""
        print("=" * 70)
        print("SAKSHAM DATASET VALIDATION")
        print("=" * 70)
        
        try:
            self.load_metadata()
            self.check_duplicates()
            self.verify_labels()
            self.validate_parquet_files()
            self.compute_statistics()
            self.generate_report(output_path)
            
            print("\n" + "=" * 70)
            print("VALIDATION COMPLETE")
            print("=" * 70)
            
            total_issues = (len(self.results['missing_files']) + 
                           len(self.results['corrupted_files']) + 
                           len(self.results['schema_issues']))
            
            if total_issues == 0:
                print("\n✅ ALL CHECKS PASSED - Dataset is ready for training!")
                return True
            else:
                print(f"\n⚠️  Found {total_issues} issues - Review report for details")
                return False
                
        except Exception as e:
            print(f"\n❌ Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    dataset_root = "E:/gessture/SAKSHAM_DATASET"
    output_path = "E:/gessture/reports/dataset_report.md"
    
    validator = DatasetValidator(dataset_root)
    success = validator.run(output_path)
    
    sys.exit(0 if success else 1)
