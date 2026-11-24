"""
dataset.py
----------
PyTorch Dataset and DataLoader generators for lazy windowing over continuous segments.
Supports configurable window size, forecast horizon, and forecast origin stride.
Prevents cross-segment contamination and enforces strict chronological target boundaries.
"""

import os
import sys
from typing import List, Tuple, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_preprocessing import FEATURE_COLS, TARGET_COL


class JenaClimateSequenceDataset(Dataset):
    """
    Lazy windowing dataset for multivariate time-series forecasting.
    Each sample corresponds to:
      - X: (window_size, n_features) scaled input observations ending at time t (origin).
      - y_scaled: (1,) scaled target at time t + horizon.
      - y_true: float, true unscaled target in degC.
      - baseline: float, true unscaled target at origin t (persistence forecast).
      - target_dt: Timestamp of the target.
    """
    def __init__(
        self,
        segments: List[pd.DataFrame],
        partition: str,
        train_end_dt: pd.Timestamp,
        val_end_dt: pd.Timestamp,
        window_size: int = 144,
        horizon: int = 144,
        stride: int = 6,
        target_scaler: Optional[Any] = None
    ):
        super().__init__()
        assert partition in ['train', 'val', 'test'], f"Invalid partition: {partition}"
        
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        self.partition = partition
        self.target_scaler = target_scaler
        
        # Precompute segment arrays in memory
        self.segment_features = []
        self.segment_targets_scaled = []
        self.segment_targets_unscaled = []
        self.segment_datetimes = []
        
        target_idx_in_features = FEATURE_COLS.index(TARGET_COL)
        scaled_col_names = [f"scaled_{col}" for col in FEATURE_COLS]
        
        for seg in segments:
            feats = seg[scaled_col_names].values.astype(np.float32)
            targets_scaled = seg[f"scaled_{TARGET_COL}"].values.astype(np.float32)
            targets_unscaled = seg[TARGET_COL].values.astype(np.float32)
            dts = seg.index.to_numpy()
            
            self.segment_features.append(feats)
            self.segment_targets_scaled.append(targets_scaled)
            self.segment_targets_unscaled.append(targets_unscaled)
            self.segment_datetimes.append(dts)
        
        # Build index mapping: list of (seg_idx, origin_idx, target_idx)
        self.indices = []
        for s_idx, dts in enumerate(self.segment_datetimes):
            n_steps = len(dts)
            # Origin can run from window_size - 1 to n_steps - 1 - horizon
            for origin_i in range(window_size - 1, n_steps - horizon, stride):
                tgt_i = origin_i + horizon
                tgt_dt = pd.Timestamp(dts[tgt_i])
                
                # Check partition based on target timestamp
                if partition == 'train':
                    if tgt_dt < train_end_dt:
                        self.indices.append((s_idx, origin_i, tgt_i))
                elif partition == 'val':
                    if train_end_dt <= tgt_dt < val_end_dt:
                        self.indices.append((s_idx, origin_i, tgt_i))
                elif partition == 'test':
                    if tgt_dt >= val_end_dt:
                        self.indices.append((s_idx, origin_i, tgt_i))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_idx, origin_i, tgt_i = self.indices[idx]
        start_i = origin_i - self.window_size + 1
        
        X = self.segment_features[s_idx][start_i : origin_i + 1] # shape (144, 19)
        y = self.segment_targets_scaled[s_idx][tgt_i : tgt_i + 1] # shape (1,)
        
        return torch.from_numpy(X), torch.from_numpy(y)

    def get_metadata_for_eval(self, idx: int) -> Dict[str, Any]:
        """
        Returns unscaled ground truth, persistence baseline, and timestamps for evaluation.
        """
        s_idx, origin_i, tgt_i = self.indices[idx]
        target_dt = pd.Timestamp(self.segment_datetimes[s_idx][tgt_i])
        origin_dt = pd.Timestamp(self.segment_datetimes[s_idx][origin_i])
        y_true = float(self.segment_targets_unscaled[s_idx][tgt_i])
        baseline = float(self.segment_targets_unscaled[s_idx][origin_i])
        
        return {
            'target_dt': target_dt,
            'origin_dt': origin_dt,
            'y_true': y_true,
            'baseline': baseline
        }


def create_dataloaders(
    segments: List[pd.DataFrame],
    train_end_dt: pd.Timestamp,
    val_end_dt: pd.Timestamp,
    batch_size: int = 128,
    window_size: int = 144,
    horizon: int = 144,
    stride: int = 6,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, JenaClimateSequenceDataset, JenaClimateSequenceDataset, JenaClimateSequenceDataset]:
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    Train DataLoader is shuffled; val and test are chronological.
    """
    train_dataset = JenaClimateSequenceDataset(
        segments=segments,
        partition='train',
        train_end_dt=train_end_dt,
        val_end_dt=val_end_dt,
        window_size=window_size,
        horizon=horizon,
        stride=stride
    )
    
    val_dataset = JenaClimateSequenceDataset(
        segments=segments,
        partition='val',
        train_end_dt=train_end_dt,
        val_end_dt=val_end_dt,
        window_size=window_size,
        horizon=horizon,
        stride=stride
    )
    
    test_dataset = JenaClimateSequenceDataset(
        segments=segments,
        partition='test',
        train_end_dt=train_end_dt,
        val_end_dt=val_end_dt,
        window_size=window_size,
        horizon=horizon,
        stride=stride
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
