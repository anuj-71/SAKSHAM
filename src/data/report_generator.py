"""
Report Generator for SAKSHAM Dataset Validation
Generates comprehensive markdown reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.data.dataset_validator import ValidationResults
from configs.config import REPORTS_DIR


class ReportGenerator:
    """Generates markdown reports for dataset validation."""
    
    def __init__(self, results: ValidationResults):
        self.results = results
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_markdown(self) -> str:
        """Generate comprehensive markdown report."""
        
        md = []
        
        # Header
        md.append("# SAKSHAM Dataset Validation Report")
        md.append(f"\n**Generated:** {self.timestamp}")
        md.append(f"\n**Status:** {'✅ PASSED' if self.results.validation_passed else '❌ FAILED'}")
        md.append("\n---\n")
        
        # Executive Summary
        md.append("## Executive Summary\n")
        md.append(f"- **Total Samples:** {self.results.total_samples:,}")
        md.append(f"- **Existing Files:** {self.results.existing_files:,}")
        md.append(f"- **Missing Files:** {self.results.missing_files:,}")
        md.append(f"- **Corrupted Files:** {self.results.corrupted_files:,}")
        md.append(f"- **Duplicate Entries:** {self.results.duplicate_entries:,}")
        md.append(f"- **Unique Participants:** {len(self.results.participants):,}")
        md.append(f"- **Dataset Size:** {self.results.dataset_size_mb:.2f} MB")
        md.append(f"- **Number of Classes:** {len(self.results.sign_map)}\n")
        
        # Sign Classes
        md.append("## Sign Classes\n")
        md.append("The dataset contains the following sign language classes:\n")
        for sign, idx in sorted(self.results.sign_map.items(), key=lambda x: x[1]):
            count = self.results.class_distribution.get(sign, 0)
            percentage = (count / self.results.total_samples * 100) if self.results.total_samples > 0 else 0
            md.append(f"- **{idx}. {sign}:** {count:,} samples ({percentage:.2f}%)")
        md.append("")
        
        # Class Distribution Table
        md.append("## Class Distribution\n")
        md.append("| Sign | Count | Percentage | Label |")
        md.append("|------|-------|------------|-------|")
        for sign, count in sorted(self.results.class_distribution.items()):
            percentage = (count / self.results.total_samples * 100)
            label_idx = self.results.sign_map[sign]
            md.append(f"| {sign} | {count:,} | {percentage:.2f}% | {label_idx} |")
        md.append("")
        
        # Class Balance Analysis
        md.append("## Class Balance Analysis\n")
        counts = list(self.results.class_distribution.values())
        if counts:
            min_count = min(counts)
            max_count = max(counts)
            avg_count = sum(counts) / len(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            md.append(f"- **Minimum samples per class:** {min_count:,}")
            md.append(f"- **Maximum samples per class:** {max_count:,}")
            md.append(f"- **Average samples per class:** {avg_count:.1f}")
            md.append(f"- **Imbalance ratio:** {imbalance_ratio:.2f}x")
            md.append("")
            
            if imbalance_ratio < 2.0:
                md.append("✅ **Dataset is well-balanced** (imbalance ratio < 2.0)")
            elif imbalance_ratio < 5.0:
                md.append("⚠️ **Moderate imbalance detected** (2.0 ≤ imbalance ratio < 5.0)")
            else:
                md.append("❌ **Significant imbalance detected** (imbalance ratio ≥ 5.0)")
            md.append("")
        
        # File Validation
        md.append("## File Validation\n")
        if self.results.missing_files == 0 and self.results.corrupted_files == 0:
            md.append("✅ **All parquet files are present and valid.**\n")
        else:
            if self.results.missing_files > 0:
                md.append(f"### ❌ Missing Files ({self.results.missing_files})\n")
                md.append("<details>")
                md.append("<summary>Click to expand missing file list</summary>\n")
                for path in self.results.missing_file_paths[:100]:  # Limit to first 100
                    md.append(f"- `{path}`")
                if len(self.results.missing_file_paths) > 100:
                    md.append(f"\n... and {len(self.results.missing_file_paths) - 100} more")
                md.append("</details>\n")
            
            if self.results.corrupted_files > 0:
                md.append(f"### ❌ Corrupted Files ({self.results.corrupted_files})\n")
                md.append("<details>")
                md.append("<summary>Click to expand corrupted file list</summary>\n")
                for path in self.results.corrupted_file_paths:
                    md.append(f"- `{path}`")
                md.append("</details>\n")
        
        # Duplicate Detection
        md.append("## Duplicate Detection\n")
        if self.results.duplicate_entries == 0:
            md.append("✅ **No duplicate entries detected.**\n")
        else:
            md.append(f"⚠️ **Found {self.results.duplicate_entries} duplicate entries.**\n")
            md.append("<details>")
            md.append("<summary>Click to expand duplicate list</summary>\n")
            for path in self.results.duplicate_paths[:50]:  # Limit to first 50
                md.append(f"- `{path}`")
            if len(self.results.duplicate_paths) > 50:
                md.append(f"\n... and {len(self.results.duplicate_paths) - 50} more")
            md.append("</details>\n")
        
        # Label Validation
        md.append("## Label Validation\n")
        if not self.results.label_mismatches:
            md.append("✅ **All labels match sign_map.json.**\n")
        else:
            md.append("❌ **Label mismatches detected:**\n")
            for mismatch in self.results.label_mismatches:
                md.append(f"- {mismatch}")
            md.append("")
        
        # Participant Statistics
        md.append("## Participant Statistics\n")
        md.append(f"- **Total unique participants:** {len(self.results.participants)}")
        md.append(f"- **Average samples per participant:** {self.results.total_samples / len(self.results.participants):.1f}\n")
        
        # Recommendations
        md.append("## Recommendations\n")
        recommendations = []
        
        if self.results.missing_files > 0:
            recommendations.append("🔴 **Critical:** Resolve missing files before training")
        
        if self.results.corrupted_files > 0:
            recommendations.append("🔴 **Critical:** Fix or remove corrupted parquet files")
        
        if self.results.duplicate_entries > 0:
            recommendations.append("🟡 **Warning:** Remove duplicate entries to avoid data leakage")
        
        counts = list(self.results.class_distribution.values())
        if counts:
            imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            if imbalance_ratio >= 5.0:
                recommendations.append("🟡 **Warning:** Consider class balancing techniques (oversampling/undersampling)")
        
        if len(self.results.participants) < 10:
            recommendations.append("🟡 **Warning:** Limited number of participants may affect generalization")
        
        if self.results.validation_passed:
            recommendations.append("🟢 **Ready:** Dataset is validated and ready for preprocessing")
        
        if recommendations:
            for rec in recommendations:
                md.append(f"- {rec}")
        else:
            md.append("- ✅ No issues detected. Dataset is ready for use.")
        
        md.append("")
        
        # Footer
        md.append("---")
        md.append(f"\n*Report generated by SAKSHAM Dataset Validator on {self.timestamp}*")
        
        return "\n".join(md)
    
    def save_report(self, filename: str = "dataset_report.md") -> Path:
        """Save markdown report to file."""
        report_path = REPORTS_DIR / filename
        
        markdown = self.generate_markdown()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        return report_path
    
    def save_json(self, filename: str = "validation_results.json") -> Path:
        """Save validation results as JSON."""
        json_path = REPORTS_DIR / filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results.to_dict(), f, indent=2)
        
        return json_path
