"""
data_preprocessing.py
---------------------
Handles dataset inspection, deduplication, sentinel value replacement,
gap detection, timeline segmentation, feature engineering, and leakage-safe scaling
for 24-hour temperature forecasting on the Jena Climate dataset.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


FEATURE_COLS = [
    'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
    'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
    'wd_sin', 'wd_cos', 'day_sin', 'day_cos', 'year_sin', 'year_cos'
]
TARGET_COL = 'T (degC)'


def load_and_clean_raw_data(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw CSV, removes exact duplicate rows, parses timestamps,
    and replaces invalid sentinel values in wind speed.
    """
    logging.info(f"Loading raw dataset from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    n_raw = len(df_raw)
    
    # 1. Exact duplicate removal
    df_dedup = df_raw.drop_duplicates().copy()
    n_exact_dups = n_raw - len(df_dedup)
    logging.info(f"Raw rows: {n_raw:,} | Exact duplicates removed: {n_exact_dups:,} | Remaining: {len(df_dedup):,}")
    
    # 2. Parse timestamps and sort
    df_dedup['Date Time'] = pd.to_datetime(df_dedup['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df_dedup = df_dedup.sort_values('Date Time').reset_index(drop=True)
    
    # 3. Check duplicate timestamps after deduplication
    n_dt_dups = int(df_dedup.duplicated(subset=['Date Time']).sum())
    logging.info(f"Duplicate timestamps after deduplication: {n_dt_dups}")
    
    # 4. Handle invalid wind speed sentinels (-9999 -> NaN)
    wv_sentinels = int((df_dedup['wv (m/s)'] == -9999).sum())
    max_wv_sentinels = int((df_dedup['max. wv (m/s)'] == -9999).sum())
    logging.info(f"Sentinel -9999 counts: wv={wv_sentinels}, max. wv={max_wv_sentinels}")
    
    df_dedup.loc[df_dedup['wv (m/s)'] == -9999, 'wv (m/s)'] = np.nan
    df_dedup.loc[df_dedup['max. wv (m/s)'] == -9999, 'max. wv (m/s)'] = np.nan
    
    metadata = {
        'n_raw_records': n_raw,
        'n_exact_duplicates_removed': n_exact_dups,
        'n_records_after_dedup': len(df_dedup),
        'n_duplicate_timestamps': n_dt_dups,
        'wv_sentinel_count': wv_sentinels,
        'max_wv_sentinel_count': max_wv_sentinels,
        'start_datetime': str(df_dedup['Date Time'].min()),
        'end_datetime': str(df_dedup['Date Time'].max()),
    }
    return df_dedup, metadata


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes circular wind features and cyclical time features (diurnal and annual).
    """
    df = df.copy()
    
    # Circular wind direction (0 to 360 degrees -> sin/cos)
    wd_rad = df['wd (deg)'] * np.pi / 180.0
    df['wd_sin'] = np.sin(wd_rad)
    df['wd_cos'] = np.cos(wd_rad)
    
    # Diurnal cycle (period = 24 hours = 86400 seconds)
    seconds_in_day = 24.0 * 3600.0
    time_sec = df['Date Time'].dt.hour * 3600.0 + df['Date Time'].dt.minute * 60.0 + df['Date Time'].dt.second
    df['day_sin'] = np.sin(time_sec * (2.0 * np.pi / seconds_in_day))
    df['day_cos'] = np.cos(time_sec * (2.0 * np.pi / seconds_in_day))
    
    # Annual cycle (period = 365.25 days = 31,557,600 seconds)
    seconds_in_year = 365.25 * seconds_in_day
    year_sec = (df['Date Time'].dt.day_of_year - 1) * seconds_in_day + time_sec
    df['year_sin'] = np.sin(year_sec * (2.0 * np.pi / seconds_in_year))
    df['year_cos'] = np.cos(year_sec * (2.0 * np.pi / seconds_in_year))
    
    return df


def build_continuous_segments(
    df: pd.DataFrame,
    metadata: Dict[str, Any]
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Reindexes the timeline onto a complete 10-minute grid.
    Applies causal forward filling with limit=2 (<= 20 min) for short gaps.
    Splits the timeline into continuous segments across long gaps (> 30 min).
    """
    full_grid = pd.date_range(start=df['Date Time'].min(), end=df['Date Time'].max(), freq='10min')
    df_grid = df.set_index('Date Time').reindex(full_grid)
    df_grid.index.name = 'Date Time'
    
    total_grid_steps = len(df_grid)
    missing_before_fill = int(df_grid['T (degC)'].isna().sum())
    
    # Causal forward fill for short gaps (limit=2 steps = 20 minutes)
    df_grid_filled = df_grid.ffill(limit=2)
    missing_after_fill = int(df_grid_filled['T (degC)'].isna().sum())
    
    logging.info(
        f"Total 10-min grid timesteps: {total_grid_steps:,} | "
        f"Missing before fill: {missing_before_fill} | "
        f"Missing after causal ffill(limit=2): {missing_after_fill}"
    )
    
    # Find continuous valid blocks
    valid_mask = ~df_grid_filled['T (degC)'].isna()
    segment_ids = (~valid_mask).cumsum()
    
    segments = []
    segment_info = []
    for sid, group in df_grid_filled[valid_mask].groupby(segment_ids[valid_mask]):
        seg_df = group.copy()
        segments.append(seg_df)
        segment_info.append({
            'segment_id': int(len(segments)),
            'start_dt': str(seg_df.index.min()),
            'end_dt': str(seg_df.index.max()),
            'timesteps': int(len(seg_df)),
            'days': float(len(seg_df) / 144.0)
        })
        logging.info(
            f"Segment {len(segments)}: {seg_df.index.min()} to {seg_df.index.max()} "
            f"({len(seg_df):,} timesteps, {len(seg_df)/144.0:.1f} days)"
        )
        
    metadata['total_grid_steps'] = total_grid_steps
    metadata['missing_before_fill'] = missing_before_fill
    metadata['missing_after_fill'] = missing_after_fill
    metadata['segments'] = segment_info
    
    return segments, metadata


def compute_chronological_splits(
    min_dt: pd.Timestamp,
    max_dt: pd.Timestamp,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Computes chronological boundaries: 70% train, 15% validation, 15% test.
    """
    total_duration = max_dt - min_dt
    train_end_dt = min_dt + total_duration * train_ratio
    val_end_dt = min_dt + total_duration * (train_ratio + val_ratio)
    return train_end_dt, val_end_dt


def fit_and_scale_pipeline(
    segments: List[pd.DataFrame],
    train_end_dt: pd.Timestamp,
    val_end_dt: pd.Timestamp,
    models_dir: str = "models"
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Fits SimpleImputer and StandardScaler strictly on the training partition (Date Time < train_end_dt).
    Transforms all segments without data leakage.
    Saves scaler and imputer objects to models_dir.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Collect all rows in training partition for fitting
    train_dfs = [seg[seg.index < train_end_dt] for seg in segments if (seg.index < train_end_dt).any()]
    train_full = pd.concat(train_dfs)
    
    logging.info(f"Fitting imputer and scaler on {len(train_full):,} training rows (strictly < {train_end_dt})...")
    
    # 1. Fit median imputer on training features
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train_full[FEATURE_COLS])
    
    # 2. Fit feature scaler on imputed training features
    feature_scaler = StandardScaler()
    train_imputed = imputer.transform(train_full[FEATURE_COLS])
    feature_scaler.fit(train_imputed)
    
    # 3. Fit target scaler strictly on training target
    target_scaler = StandardScaler()
    target_scaler.fit(train_full[[TARGET_COL]])
    
    # Transform all segments
    scaled_segments = []
    target_col_idx = FEATURE_COLS.index(TARGET_COL)
    
    for i, seg in enumerate(segments):
        seg_features_imputed = imputer.transform(seg[FEATURE_COLS])
        seg_features_scaled = feature_scaler.transform(seg_features_imputed)
        
        # Save both scaled array and original unscaled target for evaluation/persistence
        seg_copy = seg.copy()
        for idx, col in enumerate(FEATURE_COLS):
            seg_copy[f"scaled_{col}"] = seg_features_scaled[:, idx]
        
        scaled_segments.append(seg_copy)
    
    # Preprocessing bundle
    prep_bundle = {
        'feature_cols': FEATURE_COLS,
        'target_col': TARGET_COL,
        'target_col_idx': target_col_idx,
        'imputer': imputer,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'train_end_dt': train_end_dt,
        'val_end_dt': val_end_dt,
    }
    
    bundle_path = os.path.join(models_dir, "preprocessing_objects.joblib")
    joblib.dump(prep_bundle, bundle_path)
    logging.info(f"Saved preprocessing bundle to {bundle_path}")
    
    return scaled_segments, prep_bundle


def run_preprocessing_pipeline(
    csv_path: str = "dataset/jena_climate_2009_2016.csv",
    models_dir: str = "models",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[List[pd.DataFrame], Dict[str, Any], Dict[str, Any]]:
    """
    Executes the complete preprocessing pipeline.
    """
    df_clean, metadata = load_and_clean_raw_data(csv_path)
    df_engineered = engineer_features(df_clean)
    segments, metadata = build_continuous_segments(df_engineered, metadata)
    
    min_dt = segments[0].index.min()
    max_dt = segments[-1].index.max()
    train_end_dt, val_end_dt = compute_chronological_splits(min_dt, max_dt, train_ratio, val_ratio)
    
    metadata['train_end_dt'] = str(train_end_dt)
    metadata['val_end_dt'] = str(val_end_dt)
    metadata['test_start_dt'] = str(val_end_dt)
    
    scaled_segments, prep_bundle = fit_and_scale_pipeline(
        segments=segments,
        train_end_dt=train_end_dt,
        val_end_dt=val_end_dt,
        models_dir=models_dir
    )
    
    return scaled_segments, prep_bundle, metadata


if __name__ == "__main__":
    scaled_segs, bundle, meta = run_preprocessing_pipeline()
    print("Preprocessing completed successfully!")
    print(json.dumps(meta, indent=2))
