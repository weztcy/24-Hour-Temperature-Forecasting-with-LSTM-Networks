"""
train.py
--------
Training script for the 24-hour temperature forecasting LSTM model.
Includes seed setting, early stopping, LR scheduling, checkpoint saving,
and training history logging.
"""

import os
import sys
import time
import random
import logging
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_preprocessing import run_preprocessing_pipeline
from src.dataset import create_dataloaders
from src.model import LSTMRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed: int = 42):
    """Sets random seeds for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Trains the model for one epoch and returns the average loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        
        # Gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)
        
    return running_loss / total_samples


def evaluate_val_loss(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Computes the average validation loss."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            running_loss += loss.item() * len(y_batch)
            total_samples += len(y_batch)
            
    return running_loss / total_samples


def train_lstm_pipeline(
    csv_path: str = "dataset/jena_climate_2009_2016.csv",
    models_dir: str = "models",
    outputs_dir: str = "outputs",
    batch_size: int = 128,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 20,
    patience: int = 5,
    stride: int = 6,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Complete training pipeline:
      1. Preprocesses data and extracts continuous segments
      2. Creates datasets and dataloaders
      3. Trains LSTM with EarlyStopping and LR scheduling
      4. Saves best model checkpoint and training history CSV
    """
    set_seed(seed)
    
    # Configure CPU threading for optimal performance
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    logging.info(f"PyTorch CPU threads set to {num_threads}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Preprocessing
    scaled_segments, prep_bundle, metadata = run_preprocessing_pipeline(
        csv_path=csv_path,
        models_dir=models_dir
    )
    
    train_end_dt = prep_bundle['train_end_dt']
    val_end_dt = prep_bundle['val_end_dt']
    
    # 2. Dataloaders
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
        segments=scaled_segments,
        train_end_dt=train_end_dt,
        val_end_dt=val_end_dt,
        batch_size=batch_size,
        stride=stride
    )
    
    logging.info(
        f"DataLoaders prepared: Train samples={len(train_ds):,} ({len(train_loader)} batches) | "
        f"Val samples={len(val_ds):,} ({len(val_loader)} batches) | "
        f"Test samples={len(test_ds):,} ({len(test_loader)} batches)"
    )
    
    # 3. Model setup
    input_size = len(prep_bundle['feature_cols'])
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model architecture: {model}\nTrainable parameters: {n_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5)
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = os.path.join(models_dir, "best_lstm_model.pt")
    
    history = []
    start_total_time = time.time()
    
    logging.info(f"Starting training for up to {max_epochs} epochs (Early stopping patience={patience})...")
    
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_val_loss(model, val_loader, criterion, device)
        epoch_time = time.time() - t0
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        logging.info(
            f"Epoch {epoch:2d}/{max_epochs:2d} [{epoch_time:4.1f}s] - "
            f"Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | LR: {current_lr:.6f}"
        )
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'epoch_duration_sec': epoch_time
        })
        
        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'seed': seed,
                    'stride': stride,
                    'window_size': 144,
                    'horizon': 144
                }
            }, best_model_path)
            logging.info(f" -> Best model saved at epoch {epoch} with Val MSE: {val_loss:.5f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered after {epoch} epochs (Best epoch: {best_epoch} with Val MSE: {best_val_loss:.5f})")
                break
                
    total_training_time = time.time() - start_total_time
    logging.info(f"Training completed in {total_training_time:.1f}s ({total_training_time/60.0:.2f} mins).")
    
    # Save training history CSV
    history_df = pd.DataFrame(history)
    history_path = os.path.join(outputs_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    logging.info(f"Training history saved to {history_path}")
    
    summary = {
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'total_epochs': len(history),
        'total_training_time_sec': float(total_training_time),
        'best_model_path': best_model_path,
        'history_path': history_path,
        'trainable_params': n_params,
        'device': str(device),
        'seed': seed
    }
    
    return summary


if __name__ == "__main__":
    train_lstm_pipeline()
