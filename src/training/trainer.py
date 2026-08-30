"""
Training Pipeline for Sign Language Recognition
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from typing import Dict, List, Tuple
import time


class Trainer:
    """Training pipeline for sign language models"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 checkpoint_dir: str = 'checkpoints'):
        """
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: 'cuda' or 'cpu'
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            checkpoint_dir: Directory to save checkpoints
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for sequences, labels, seq_lens in pbar:
            # Move to device
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            seq_lens = seq_lens.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Handle different model outputs
            outputs = self.model(sequences, seq_lens)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # BiGRU_Attention returns (logits, attention)
            else:
                logits = outputs  # LSTM returns only logits
            
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for sequences, labels, seq_lens in pbar:
                # Move to device
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                seq_lens = seq_lens.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences, seq_lens)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (val_acc: {val_acc:.2f}%)")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 70)
        
        epochs_without_improvement = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Save history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch}/{num_epochs}] ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            if is_best:
                print(f"  ⭐ New best validation accuracy!")
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
                print(f"  Best val accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
                break
            
            print("-" * 70)
        
        # Save final history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


def compute_class_weights(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced datasets"""
    class_counts = torch.zeros(num_classes)
    
    for _, labels, _ in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)
    
    return class_weights


if __name__ == "__main__":
    # Test trainer setup
    from models import BiGRU_Attention
    from torch.utils.data import TensorDataset
    
    # Create dummy data
    num_samples = 100
    seq_len = 128
    input_dim = 225
    num_classes = 13  # Updated from 15 (Phase 1: removed mom/dad)
    
    X = torch.randn(num_samples, seq_len, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    lens = torch.randint(50, seq_len+1, (num_samples,))
    
    dataset = TensorDataset(X, y, lens)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = BiGRU_Attention(input_dim=input_dim, num_classes=num_classes)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',
        checkpoint_dir='test_checkpoints'
    )
    
    print("Trainer initialized successfully!")
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
