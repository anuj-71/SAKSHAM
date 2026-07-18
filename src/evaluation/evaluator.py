"""
Model Evaluation and Metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm


class Evaluator:
    """Evaluate sign language recognition models"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: Trained model
            device: Device to run evaluation on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
    
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for entire dataset
        
        Returns:
            predictions: Predicted labels
            ground_truth: True labels
        """
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels, seq_lens in tqdm(dataloader, desc='Predicting'):
                sequences = sequences.to(self.device)
                seq_lens = seq_lens.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences, seq_lens)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Get predictions
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_predictions), np.array(all_labels)
    
    def compute_metrics(self, predictions: np.ndarray, 
                       ground_truth: np.ndarray) -> Dict:
        """
        Compute evaluation metrics
        
        Returns:
            Dictionary with accuracy, precision, recall, f1
        """
        metrics = {
            'accuracy': accuracy_score(ground_truth, predictions),
            'precision_macro': precision_score(ground_truth, predictions, average='macro', zero_division=0),
            'precision_weighted': precision_score(ground_truth, predictions, average='weighted', zero_division=0),
            'recall_macro': recall_score(ground_truth, predictions, average='macro', zero_division=0),
            'recall_weighted': recall_score(ground_truth, predictions, average='weighted', zero_division=0),
            'f1_macro': f1_score(ground_truth, predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def compute_confusion_matrix(self, predictions: np.ndarray,
                                ground_truth: np.ndarray,
                                normalize: bool = False) -> np.ndarray:
        """Compute confusion matrix"""
        cm = confusion_matrix(ground_truth, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return cm
    
    def evaluate(self, dataloader: DataLoader, 
                class_names: List[str] = None) -> Dict:
        """
        Complete evaluation pipeline
        
        Args:
            dataloader: Test/validation dataloader
            class_names: List of class names for reporting
            
        Returns:
            Dictionary with all metrics and confusion matrix
        """
        print("Running evaluation...")
        
        # Get predictions
        predictions, ground_truth = self.predict(dataloader)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, ground_truth)
        
        # Confusion matrix
        cm = self.compute_confusion_matrix(predictions, ground_truth)
        cm_normalized = self.compute_confusion_matrix(predictions, ground_truth, normalize=True)
        
        # Classification report
        if class_names:
            report = classification_report(
                ground_truth, predictions,
                target_names=class_names,
                zero_division=0,
                output_dict=True
            )
        else:
            report = classification_report(
                ground_truth, predictions,
                zero_division=0,
                output_dict=True
            )
        
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'classification_report': report,
            'predictions': predictions.tolist(),
            'ground_truth': ground_truth.tolist()
        }
        
        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Accuracy:        {metrics['accuracy']*100:.2f}%")
        print(f"Precision (macro):  {metrics['precision_macro']*100:.2f}%")
        print(f"Recall (macro):     {metrics['recall_macro']*100:.2f}%")
        print(f"F1 Score (macro):   {metrics['f1_macro']*100:.2f}%")
        print("=" * 70)
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                             class_names: List[str],
                             output_path: str,
                             normalize: bool = False,
                             figsize: Tuple[int, int] = (12, 10)):
        """Plot and save confusion matrix"""
        plt.figure(figsize=figsize)
        
        # Format based on normalization
        if normalize:
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {output_path}")
    
    def plot_training_curves(self, history: Dict, output_path: str):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curve
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {output_path}")
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def generate_evaluation_report(evaluator: Evaluator,
                              test_loader: DataLoader,
                              sign_map: Dict[str, int],
                              history: Dict,
                              output_dir: str):
    """
    Generate complete evaluation report with plots
    
    Args:
        evaluator: Evaluator instance
        test_loader: Test dataloader
        sign_map: Dictionary mapping sign names to indices
        history: Training history
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get class names sorted by index
    class_names = [k for k, v in sorted(sign_map.items(), key=lambda x: x[1])]
    
    # Run evaluation
    results = evaluator.evaluate(test_loader, class_names)
    
    # Plot confusion matrices
    cm = np.array(results['confusion_matrix'])
    cm_norm = np.array(results['confusion_matrix_normalized'])
    
    evaluator.plot_confusion_matrix(
        cm, class_names,
        output_dir / 'confusion_matrix.png',
        normalize=False
    )
    
    evaluator.plot_confusion_matrix(
        cm_norm, class_names,
        output_dir / 'confusion_matrix_normalized.png',
        normalize=True
    )
    
    # Plot training curves
    evaluator.plot_training_curves(history, output_dir / 'training_curves.png')
    
    # Save results
    evaluator.save_results(results, output_dir / 'evaluation_results.json')
    
    # Generate markdown report
    report_lines = []
    report_lines.append("# Model Evaluation Report\n\n")
    report_lines.append("## Overall Metrics\n\n")
    
    metrics = results['metrics']
    report_lines.append(f"- **Accuracy**: {metrics['accuracy']*100:.2f}%\n")
    report_lines.append(f"- **Precision (Macro)**: {metrics['precision_macro']*100:.2f}%\n")
    report_lines.append(f"- **Recall (Macro)**: {metrics['recall_macro']*100:.2f}%\n")
    report_lines.append(f"- **F1 Score (Macro)**: {metrics['f1_macro']*100:.2f}%\n\n")
    
    report_lines.append("## Per-Class Performance\n\n")
    report_lines.append("| Class | Precision | Recall | F1-Score | Support |\n")
    report_lines.append("|-------|-----------|--------|----------|----------|\n")
    
    for class_name in class_names:
        if class_name in results['classification_report']:
            cls_metrics = results['classification_report'][class_name]
            report_lines.append(
                f"| {class_name} | {cls_metrics['precision']*100:.2f}% | "
                f"{cls_metrics['recall']*100:.2f}% | {cls_metrics['f1-score']*100:.2f}% | "
                f"{cls_metrics['support']} |\n"
            )
    
    report_lines.append("\n## Training Summary\n\n")
    report_lines.append(f"- **Total Epochs**: {len(history['train_loss'])}\n")
    report_lines.append(f"- **Best Val Accuracy**: {max(history['val_acc']):.2f}%\n")
    report_lines.append(f"- **Final Train Loss**: {history['train_loss'][-1]:.4f}\n")
    report_lines.append(f"- **Final Val Loss**: {history['val_loss'][-1]:.4f}\n\n")
    
    report_lines.append("## Visualizations\n\n")
    report_lines.append("### Training Curves\n")
    report_lines.append("![Training Curves](training_curves.png)\n\n")
    report_lines.append("### Confusion Matrix\n")
    report_lines.append("![Confusion Matrix](confusion_matrix.png)\n\n")
    report_lines.append("### Normalized Confusion Matrix\n")
    report_lines.append("![Normalized Confusion Matrix](confusion_matrix_normalized.png)\n")
    
    with open(output_dir / 'evaluation_report.md', 'w') as f:
        f.writelines(report_lines)
    
    print(f"\nComplete evaluation report saved to {output_dir}")


if __name__ == "__main__":
    print("Evaluator module ready for use")
