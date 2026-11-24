"""
model.py
--------
PyTorch LSTM Regressor for multivariate time-series forecasting.
"""

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """
    Multivariate LSTM Regressor.
    Architecture:
      - 2-layer stacked LSTM with inter-layer dropout
      - Last timestep feature extraction
      - Fully connected regression head with ReLU and dropout
      - Output: scalar prediction of scaled temperature at horizon t + H
    """
    def __init__(
        self,
        input_size: int = 19,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_hidden_size: int = 32
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dense regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc_hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len=144, input_size=19)
        output shape: (batch_size, 1)
        """
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Select last timestep representation
        last_step = lstm_out[:, -1, :] # shape: (batch_size, hidden_size)
        out = self.head(last_step)     # shape: (batch_size, 1)
        return out
