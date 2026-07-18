"""
Sign Language Recognition Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence models"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, gru_output: torch.Tensor, 
                seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gru_output: (batch, seq_len, hidden_dim)
            seq_lens: (batch,) - actual sequence lengths
            
        Returns:
            context: (batch, hidden_dim)
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention(gru_output).squeeze(-1)  # (batch, seq_len)
        
        # Create mask for padding
        batch_size, max_len = scores.shape
        mask = torch.arange(max_len, device=scores.device).expand(batch_size, max_len) < seq_lens.unsqueeze(1)
        
        # Apply mask
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        # Compute weighted context
        context = torch.bmm(attention_weights.unsqueeze(1), gru_output).squeeze(1)  # (batch, hidden_dim)
        
        return context, attention_weights


class BiGRU_Attention(nn.Module):
    """Bidirectional GRU with Attention for Sign Language Recognition"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 15,
                 dropout: float = 0.3):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for GRU
            num_layers: Number of GRU layers
            num_classes: Number of sign classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            seq_lens: (batch,) - actual sequence lengths
            
        Returns:
            logits: (batch, num_classes)
            attention_weights: (batch, seq_len)
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        # Pack padded sequence for efficiency
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # GRU forward pass
        packed_output, _ = self.gru(packed_input)
        
        # Unpack sequence
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # (batch, seq_len, hidden_dim*2)
        
        # Apply attention
        context, attention_weights = self.attention(gru_output, seq_lens)
        
        # Classification
        logits = self.classifier(context)  # (batch, num_classes)
        
        return logits, attention_weights


class SimpleLSTM(nn.Module):
    """Simple LSTM baseline model"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 15,
                 dropout: float = 0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        # Pack sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        packed_output, (hidden, _) = self.lstm(packed_input)
        
        # Use last hidden states from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden*2)
        
        # Classify
        logits = self.classifier(hidden)
        
        return logits


def create_model(model_name: str,
                input_dim: int,
                num_classes: int = 15,
                **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: 'bigru_attention' or 'lstm'
        input_dim: Input feature dimension
        num_classes: Number of classes
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    if model_name == 'bigru_attention':
        return BiGRU_Attention(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_name == 'lstm':
        return SimpleLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test model
    batch_size = 4
    seq_len = 128
    input_dim = 225  # 75 landmarks * 3 (x,y,z)
    num_classes = 15
    
    # Create model
    model = BiGRU_Attention(input_dim=input_dim, num_classes=num_classes)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    seq_lens = torch.tensor([128, 100, 80, 60])
    
    logits, attention = model(x, seq_lens)
    
    print(f"Model: BiGRU_Attention")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Attention shape: {attention.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
