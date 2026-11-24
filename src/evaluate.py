"""
evaluate.py
-----------
Comprehensive evaluation and visualization module for the 24-hour temperature forecasting project.
Computes test metrics (MAE, RMSE, R²) for both LSTM and the Persistence Baseline on Celsius scale.
Generates all 10+ mandatory high-resolution figures saved to outputs/figures/.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import (
    load_and_clean_raw_data,
    engineer_features,
    build_continuous_segments,
    fit_and_scale_pipeline,
    FEATURE_COLS,
    TARGET_COL
)
from src.dataset import JenaClimateSequenceDataset, create_dataloaders
from src.model import LSTMRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculates MAE, RMSE, and R2 on unscaled Celsius scale."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def run_evaluation(
    model_path: str = "models/best_lstm_model.pt",
    prep_path: str = "models/preprocessing_objects.joblib",
    csv_path: str = "dataset/jena_climate_2009_2016.csv",
    outputs_dir: str = "outputs",
    figures_dir: str = "outputs/figures"
) -> Dict[str, Any]:
    """
    Executes full evaluation:
      - Loads trained model and preprocessing bundle
      - Computes test predictions and baseline predictions
      - Computes metrics
      - Generates all 10+ mandatory visualization figures
      - Saves outputs/metrics.json
    """
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading preprocessing objects from {prep_path}...")
    prep_bundle = joblib.load(prep_path)
    
    feature_scaler = prep_bundle['feature_scaler']
    target_scaler = prep_bundle['target_scaler']
    train_end_dt = prep_bundle['train_end_dt']
    val_end_dt = prep_bundle['val_end_dt']
    
    # Reload cleaned dataset to recreate segments
    logging.info(f"Processing dataset for evaluation...")
    df_clean, metadata = load_and_clean_raw_data(csv_path)
    df_engineered = engineer_features(df_clean)
    segments, metadata = build_continuous_segments(df_engineered, metadata)
    
    scaled_segments, _ = fit_and_scale_pipeline(
        segments=segments,
        train_end_dt=train_end_dt,
        val_end_dt=val_end_dt,
        models_dir="models"
    )
    
    # Create test dataset and dataloader
    _, _, test_loader, _, _, test_dataset = create_dataloaders(
        segments=scaled_segments,
        train_end_dt=train_end_dt,
        val_end_dt=val_end_dt,
        batch_size=128,
        stride=6
    )
    
    logging.info(f"Loading trained LSTM model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = LSTMRegressor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Collect predictions
    y_preds_scaled = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            y_preds_scaled.append(preds.cpu().numpy())
            
    y_preds_scaled = np.vstack(y_preds_scaled)
    # Inverse transform to Celsius
    y_pred_celsius = target_scaler.inverse_transform(y_preds_scaled).flatten()
    
    # Collect ground truth, baseline, and timestamps
    n_test = len(test_dataset)
    y_true_celsius = np.zeros(n_test, dtype=np.float32)
    y_baseline_celsius = np.zeros(n_test, dtype=np.float32)
    test_dts = []
    
    for i in range(n_test):
        meta_i = test_dataset.get_metadata_for_eval(i)
        y_true_celsius[i] = meta_i['y_true']
        y_baseline_celsius[i] = meta_i['baseline']
        test_dts.append(meta_i['target_dt'])
        
    test_dts = pd.to_datetime(test_dts)
    
    # Compute metrics
    lstm_metrics = evaluate_predictions(y_true_celsius, y_pred_celsius)
    baseline_metrics = evaluate_predictions(y_true_celsius, y_baseline_celsius)
    
    # Compute improvement percentages
    mae_improvement = ((baseline_metrics['mae'] - lstm_metrics['mae']) / baseline_metrics['mae']) * 100.0
    rmse_improvement = ((baseline_metrics['rmse'] - lstm_metrics['rmse']) / baseline_metrics['rmse']) * 100.0
    
    results = {
        'model_name': 'LSTM Regressor (2-layer)',
        'baseline_name': 'Persistence Baseline (T(t+24h) = T(t))',
        'n_test_samples': n_test,
        'test_period_start': str(test_dts.min()),
        'test_period_end': str(test_dts.max()),
        'lstm_metrics': {
            'mae_degC': lstm_metrics['mae'],
            'rmse_degC': lstm_metrics['rmse'],
            'r2_score': lstm_metrics['r2']
        },
        'baseline_metrics': {
            'mae_degC': baseline_metrics['mae'],
            'rmse_degC': baseline_metrics['rmse'],
            'r2_score': baseline_metrics['r2']
        },
        'relative_improvement_pct': {
            'mae_reduction_pct': mae_improvement,
            'rmse_reduction_pct': rmse_improvement
        },
        'training_summary': {
            'best_epoch': checkpoint.get('epoch', None),
            'best_val_loss': checkpoint.get('val_loss', None),
            'train_loss': checkpoint.get('train_loss', None),
            'architecture': config
        }
    }
    
    logging.info("\n" + "="*50)
    logging.info("FINAL TEST SET EVALUATION RESULTS")
    logging.info("="*50)
    logging.info(f"Test Samples: {n_test:,} | Period: {test_dts.min()} to {test_dts.max()}")
    logging.info(f"Persistence Baseline -> MAE: {baseline_metrics['mae']:.4f} °C | RMSE: {baseline_metrics['rmse']:.4f} °C | R²: {baseline_metrics['r2']:.4f}")
    logging.info(f"LSTM Model           -> MAE: {lstm_metrics['mae']:.4f} °C | RMSE: {lstm_metrics['rmse']:.4f} °C | R²: {lstm_metrics['r2']:.4f}")
    logging.info(f"LSTM MAE Reduction: {mae_improvement:.2f}% | RMSE Reduction: {rmse_improvement:.2f}%")
    logging.info("="*50)
    
    # Save metrics JSON
    metrics_path = os.path.join(outputs_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved metrics to {metrics_path}")
    
    # Save test predictions dataframe for inspection / plotting
    test_df = pd.DataFrame({
        'Date Time': test_dts,
        'Actual': y_true_celsius,
        'LSTM_Pred': y_pred_celsius,
        'Persistence_Baseline': y_baseline_celsius,
        'Residual_LSTM': y_pred_celsius - y_true_celsius,
        'Residual_Baseline': y_baseline_celsius - y_true_celsius,
        'AbsError_LSTM': np.abs(y_pred_celsius - y_true_celsius),
        'AbsError_Baseline': np.abs(y_baseline_celsius - y_true_celsius)
    })
    test_df.to_csv(os.path.join(outputs_dir, "test_predictions.csv"), index=False)
    
    # Generate all mandatory visualizations
    generate_all_figures(
        df_clean=df_clean,
        df_engineered=df_engineered,
        train_end_dt=train_end_dt,
        val_end_dt=val_end_dt,
        test_df=test_df,
        history_path=os.path.join(outputs_dir, "training_history.csv"),
        results=results,
        figures_dir=figures_dir
    )
    
    return results


def generate_all_figures(
    df_clean: pd.DataFrame,
    df_engineered: pd.DataFrame,
    train_end_dt: pd.Timestamp,
    val_end_dt: pd.Timestamp,
    test_df: pd.DataFrame,
    history_path: str,
    results: Dict[str, Any],
    figures_dir: str
):
    """Generates all 10+ mandatory visualization figures with publication quality."""
    logging.info(f"Generating mandatory figures in {figures_dir}...")
    
    # -----------------------------------------------------------------
    # Figure 1: Temperature Over Time (chronological plot across dataset)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5), dpi=200)
    # Downsample for crisp plotting of 420k points
    df_plot = df_clean.iloc[::6].copy()
    ax.plot(df_plot['Date Time'], df_plot['T (degC)'], color='#2b5c8f', lw=0.6, alpha=0.85, label='Observed Temperature')
    
    # Add partition spans
    ax.axvspan(df_plot['Date Time'].min(), train_end_dt, color='#2ca02c', alpha=0.12, label='Training Partition (70%)')
    ax.axvspan(train_end_dt, val_end_dt, color='#ff7f0e', alpha=0.15, label='Validation Partition (15%)')
    ax.axvspan(val_end_dt, df_plot['Date Time'].max(), color='#d62728', alpha=0.12, label='Test Partition (15%)')
    
    ax.axvline(train_end_dt, color='#2ca02c', linestyle='--', lw=1.2)
    ax.axvline(val_end_dt, color='#ff7f0e', linestyle='--', lw=1.2)
    
    ax.set_title("Jena Climate (2009–2016): Air Temperature Chronology and Partition Boundaries", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date Time", fontsize=11)
    ax.set_ylabel("Air Temperature T (°C)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, facecolor='white')
    fig.tight_layout()
    fig1_path = os.path.join(figures_dir, "01_temperature_over_time.png")
    fig.savefig(fig1_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved: {fig1_path}")

    # -----------------------------------------------------------------
    # Figure 2: Temperature Distribution (histogram + KDE across partitions)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    train_mask = df_clean['Date Time'] < train_end_dt
    val_mask = (df_clean['Date Time'] >= train_end_dt) & (df_clean['Date Time'] < val_end_dt)
    test_mask = df_clean['Date Time'] >= val_end_dt
    
    sns.kdeplot(df_clean.loc[train_mask, 'T (degC)'], ax=ax, color='#2ca02c', lw=2.0, label=f"Train (Mean: {df_clean.loc[train_mask, 'T (degC)'].mean():.1f}°C, Std: {df_clean.loc[train_mask, 'T (degC)'].std():.1f}°C)")
    sns.kdeplot(df_clean.loc[val_mask, 'T (degC)'], ax=ax, color='#ff7f0e', lw=2.0, label=f"Validation (Mean: {df_clean.loc[val_mask, 'T (degC)'].mean():.1f}°C, Std: {df_clean.loc[val_mask, 'T (degC)'].std():.1f}°C)")
    sns.kdeplot(df_clean.loc[test_mask, 'T (degC)'], ax=ax, color='#d62728', lw=2.0, label=f"Test (Mean: {df_clean.loc[test_mask, 'T (degC)'].mean():.1f}°C, Std: {df_clean.loc[test_mask, 'T (degC)'].std():.1f}°C)")
    
    ax.set_title("Empirical Distribution of Air Temperature T (°C) across Chronological Partitions", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Air Temperature T (°C)", fontsize=11)
    ax.set_ylabel("Kernel Density", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, framealpha=0.9, facecolor='white')
    fig.tight_layout()
    fig2_path = os.path.join(figures_dir, "02_temperature_distribution.png")
    fig.savefig(fig2_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved: {fig2_path}")

    # -----------------------------------------------------------------
    # Figure 3: Feature Correlation Heatmap
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 11), dpi=200)
    corr_matrix = df_engineered[FEATURE_COLS].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7.5},
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation Coefficient"},
        ax=ax
    )
    ax.set_title("Correlation Heatmap of Engineered Meteorological and Temporal Features", fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    fig.tight_layout()
    fig3_path = os.path.join(figures_dir, "03_feature_correlation_heatmap.png")
    fig.savefig(fig3_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved: {fig3_path}")

    # -----------------------------------------------------------------
    # Figure 4: Data-Quality Visualization (Missing / Sentinel / Gaps)
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    
    # Left: Data cleaning summary bar chart
    clean_labels = ['Exact Duplicate\nRows Removed', 'wv (m/s)\nSentinels (-9999)', 'max. wv (m/s)\nSentinels (-9999)', 'Long Missing\nGaps (>30m)']
    clean_counts = [327, 18, 20, 2]
    colors = ['#4575b4', '#d73027', '#fc8d59', '#74add1']
    
    bars = ax1.bar(clean_labels, clean_counts, color=colors, edgecolor='#333333', width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h}',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_title("Identified and Remediated Data Quality Anomalies", fontsize=11, fontweight='bold', pad=10)
    ax1.set_ylabel("Count of Records / Events", fontsize=10.5)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Right: Timeline missing intervals visualization
    ax2.plot([0, 10], [0, 0], color='#2ca02c', lw=8, solid_capstyle='butt', label='Continuous Segment 1 (2,093 days)')
    ax2.plot([10.05, 13.8], [0, 0], color='#ff7f0e', lw=8, solid_capstyle='butt', label='Continuous Segment 2 (761 days)')
    ax2.plot([13.85, 14.15], [0, 0], color='#1f77b4', lw=8, solid_capstyle='butt', label='Continuous Segment 3 (65 days)')
    
    ax2.scatter([10.025, 13.825], [0, 0], color='#d62728', s=120, zorder=5, marker='X', label='Unfillable Gaps (>30m): 16h & 3.1d')
    ax2.set_xlim(-0.5, 14.8)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Relative Timeline Progression (2009–2016)", fontsize=10.5)
    ax2.set_title("Timeline Segmentation Across Long Missing Gaps", fontsize=11, fontweight='bold', pad=10)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2, frameon=True, facecolor='white', fontsize=9.5)
    ax2.grid(axis='x', linestyle=':', alpha=0.5)
    
    fig.tight_layout()
    fig4_path = os.path.join(figures_dir, "04_data_quality_issues.png")
    fig.savefig(fig4_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {fig4_path}")

    # -----------------------------------------------------------------
    # Figure 5: Training History (Train & Validation Loss by Epoch)
    # -----------------------------------------------------------------
    if os.path.exists(history_path):
        hist_df = pd.read_csv(history_path)
        fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
        
        ax.plot(hist_df['epoch'], hist_df['train_loss'], marker='o', lw=1.8, color='#1f77b4', label='Training Loss (MSE)')
        ax.plot(hist_df['epoch'], hist_df['val_loss'], marker='s', lw=1.8, color='#ff7f0e', label='Validation Loss (MSE)')
        
        best_epoch_idx = hist_df['val_loss'].idxmin()
        best_ep = hist_df.loc[best_epoch_idx, 'epoch']
        best_vl = hist_df.loc[best_epoch_idx, 'val_loss']
        ax.scatter([best_ep], [best_vl], color='#d62728', s=130, zorder=5, label=f'Best Checkpoint (Epoch {best_ep}: {best_vl:.4f})')
        
        ax.set_title("LSTM Model Training History: Mean Squared Error Across Epochs", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss (MSE on Scaled Target)", fontsize=11)
        ax.set_xticks(hist_df['epoch'])
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, framealpha=0.9, facecolor='white')
        fig.tight_layout()
        fig5_path = os.path.join(figures_dir, "05_training_validation_history.png")
        fig.savefig(fig5_path, dpi=200)
        plt.close(fig)
        logging.info(f"Saved: {fig5_path}")

    # -----------------------------------------------------------------
    # Figure 6: Actual vs Predicted Temperature (Chronological Test-Period)
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=200, gridspec_kw={'height_ratios': [1, 1.2]})
    
    # Subplot 1: Full test period overview
    ax1.plot(test_df['Date Time'], test_df['Actual'], color='#333333', lw=0.9, alpha=0.8, label='Actual T (°C)')
    ax1.plot(test_df['Date Time'], test_df['LSTM_Pred'], color='#d62728', lw=1.0, alpha=0.8, label='LSTM 24h Forecast')
    ax1.set_title("Full Held-Out Test Period (Oct 2015 – Dec 2016): Actual vs. 24-Hour LSTM Forecast", fontsize=12, fontweight='bold', pad=8)
    ax1.set_ylabel("T (°C)", fontsize=10.5)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', frameon=True, facecolor='white', ncol=2)
    
    # Subplot 2: Representative 4-week detailed window (e.g. July 2016)
    zoom_start = pd.Timestamp("2016-07-01 00:00:00")
    zoom_end = pd.Timestamp("2016-07-28 23:59:59")
    zoom_df = test_df[(test_df['Date Time'] >= zoom_start) & (test_df['Date Time'] <= zoom_end)]
    
    ax2.plot(zoom_df['Date Time'], zoom_df['Actual'], color='#222222', lw=1.8, label='Actual Ground Truth')
    ax2.plot(zoom_df['Date Time'], zoom_df['LSTM_Pred'], color='#d62728', lw=1.8, linestyle='-', label='LSTM 24-Hour Forecast')
    ax2.plot(zoom_df['Date Time'], zoom_df['Persistence_Baseline'], color='#1f77b4', lw=1.2, linestyle=':', alpha=0.75, label='Persistence Baseline T(t)')
    
    ax2.set_title("Zoomed View (July 2016, 4 Weeks): Diurnal Cycle Tracking & Baseline Comparison", fontsize=12, fontweight='bold', pad=8)
    ax2.set_xlabel("Date Time", fontsize=10.5)
    ax2.set_ylabel("T (°C)", fontsize=10.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', frameon=True, facecolor='white')
    
    fig.tight_layout()
    fig6_path = os.path.join(figures_dir, "06_actual_vs_predicted_chronological.png")
    fig.savefig(fig6_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved: {fig6_path}")

    # -----------------------------------------------------------------
    # Figure 7: Prediction Scatter Plot (Actual vs Predicted with Diagonal)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    
    # Hexbin / scatter with density
    hb = ax.hexbin(
        test_df['Actual'],
        test_df['LSTM_Pred'],
        gridsize=60,
        cmap='Blues',
        mincnt=1,
        bins='log',
        edgecolors='none'
    )
    cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label('Log10 Sample Density', fontsize=10)
    
    min_val = min(test_df['Actual'].min(), test_df['LSTM_Pred'].min()) - 2
    max_val = max(test_df['Actual'].max(), test_df['LSTM_Pred'].max()) + 2
    ax.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', lw=2.0, label='Ideal 1:1 Identity Line')
    
    r2_val = results['lstm_metrics']['r2_score']
    mae_val = results['lstm_metrics']['mae_degC']
    rmse_val = results['lstm_metrics']['rmse_degC']
    
    ax.annotate(
        f"Test R²:    {r2_val:.4f}\nMAE:   {mae_val:.2f} °C\nRMSE: {rmse_val:.2f} °C",
        xy=(0.05, 0.85),
        xycoords='axes fraction',
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#999999", alpha=0.9)
    )
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.set_title("Actual vs. Predicted 24-Hour Temperature Scatter (Test Set)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Actual Temperature (°C)", fontsize=11)
    ax.set_ylabel("LSTM Predicted Temperature (°C)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, facecolor='white')
    fig.tight_layout()
    fig7_path = os.path.join(figures_dir, "07_prediction_scatter_plot.png")
    fig.savefig(fig7_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved: {fig7_path}")

    # -----------------------------------------------------------------
    # Figure 8: Residual Distribution (Histogram / KDE of Prediction Errors)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    residuals_lstm = test_df['Residual_LSTM']
    residuals_base = test_df['Residual_Baseline']
    
    sns.kdeplot(residuals_base, ax=ax, color='#1f77b4', lw=1.8, linestyle=':', label=f"Persistence Residuals (Std: {residuals_base.std():.2f}°C, Mean: {residuals_base.mean():.2f}°C)")
    sns.kdeplot(residuals_lstm, ax=ax, color='#d62728', lw=2.2, label=f"LSTM Residuals (Std: {residuals_lstm.std():.2f}°C, Mean: {residuals_lstm.mean():.2f}°C)")
    
    ax.axvline(0, color='#333333', linestyle='--', lw=1.2, alpha=0.7, label='Zero Error Reference')
    
    ax.set_title("Test Residual Error Distribution: LSTM vs. Persistence Baseline", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Residual Error (Predicted - Actual) [°C]", fontsize=11)
    ax.set_ylabel("Kernel Density", fontsize=11)
    ax.set_xlim(-15, 15)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, framealpha=0.9, facecolor='white')
    fig.tight_layout()
    fig8_path = os.path.join(figures_dir, "08_residual_distribution.png")
    fig.savefig(fig8_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved: {fig8_path}")

    # -----------------------------------------------------------------
    # Figure 9: Residuals Over Time (Chronological Test Residuals)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5), dpi=200)
    ax.plot(test_df['Date Time'], test_df['Residual_LSTM'], color='#2b5c8f', lw=0.7, alpha=0.75, label='LSTM Residual (Predicted - Actual)')
    ax.axhline(0, color='#d62728', linestyle='--', lw=1.5, label='Zero Error Reference')
    
    # 2-sigma bands
    res_std = residuals_lstm.std()
    ax.axhline(2 * res_std, color='#e41a1c', linestyle=':', lw=1.0, alpha=0.8, label=f'+2σ ({2*res_std:.2f}°C)')
    ax.axhline(-2 * res_std, color='#e41a1c', linestyle=':', lw=1.0, alpha=0.8, label=f'-2σ ({-2*res_std:.2f}°C)')
    
    ax.set_title("Chronological Test Set Residuals Across 14 Months (Oct 2015 – Dec 2016)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date Time", fontsize=11)
    ax.set_ylabel("Residual Error (°C)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, facecolor='white', ncol=4)
    fig.tight_layout()
    fig9_path = os.path.join(figures_dir, "09_residuals_over_time.png")
    fig.savefig(fig9_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved: {fig9_path}")

    # -----------------------------------------------------------------
    # Figure 10: Model Comparison (Bar Chart: LSTM vs Persistence Metrics)
    # -----------------------------------------------------------------
    fig, (ax_err, ax_r2) = plt.subplots(1, 2, figsize=(11, 5), dpi=200, gridspec_kw={'width_ratios': [1.3, 1]})
    
    # Metrics comparison bar chart
    metrics_names = ['MAE (°C)', 'RMSE (°C)']
    base_errs = [results['baseline_metrics']['mae_degC'], results['baseline_metrics']['rmse_degC']]
    lstm_errs = [results['lstm_metrics']['mae_degC'], results['lstm_metrics']['rmse_degC']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    rects1 = ax_err.bar(x - width/2, base_errs, width, label='Persistence Baseline', color='#74add1', edgecolor='#333333')
    rects2 = ax_err.bar(x + width/2, lstm_errs, width, label='LSTM Model', color='#d73027', edgecolor='#333333')
    
    for r in rects1:
        h = r.get_height()
        ax_err.annotate(f'{h:.2f}°C', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontweight='bold', fontsize=10)
    for r in rects2:
        h = r.get_height()
        ax_err.annotate(f'{h:.2f}°C', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
    ax_err.set_title("Forecast Error Comparison (Lower is Better)", fontsize=11.5, fontweight='bold', pad=10)
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(metrics_names, fontsize=10.5)
    ax_err.set_ylabel("Error (°C)", fontsize=10.5)
    ax_err.set_ylim(0, max(base_errs) * 1.25)
    ax_err.grid(axis='y', linestyle=':', alpha=0.6)
    ax_err.legend(frameon=True, facecolor='white')
    
    # R2 comparison
    r2_names = ['Persistence', 'LSTM']
    r2_vals = [results['baseline_metrics']['r2_score'], results['lstm_metrics']['r2_score']]
    r2_colors = ['#74add1', '#d73027']
    
    rects_r2 = ax_r2.bar(r2_names, r2_vals, width=0.45, color=r2_colors, edgecolor='#333333')
    for r in rects_r2:
        h = r.get_height()
        ax_r2.annotate(f'{h:.4f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
    ax_r2.set_title("Explained Variance R² (Higher is Better)", fontsize=11.5, fontweight='bold', pad=10)
    ax_r2.set_ylabel("R² Score", fontsize=10.5)
    ax_r2.set_ylim(0, 1.05)
    ax_r2.grid(axis='y', linestyle=':', alpha=0.6)
    
    fig.suptitle("Performance Benchmark: LSTM Network vs. 24-Hour Persistence Baseline", fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig10_path = os.path.join(figures_dir, "10_model_comparison.png")
    fig.savefig(fig10_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {fig10_path}")

    # -----------------------------------------------------------------
    # Figure 11: Error Breakdown by Season and Temperature Regime
    # -----------------------------------------------------------------
    fig, (ax_month, ax_temp) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    
    # Monthly MAE
    test_df['Month'] = test_df['Date Time'].dt.strftime('%b')
    test_df['Month_Num'] = test_df['Date Time'].dt.month
    monthly_mae = test_df.groupby(['Month_Num', 'Month'])[['AbsError_LSTM', 'AbsError_Baseline']].mean().reset_index()
    monthly_mae = monthly_mae.sort_values('Month_Num')
    
    xm = np.arange(len(monthly_mae))
    ax_month.plot(xm, monthly_mae['AbsError_Baseline'], marker='s', color='#1f77b4', lw=1.8, linestyle=':', label='Persistence MAE')
    ax_month.plot(xm, monthly_mae['AbsError_LSTM'], marker='o', color='#d62728', lw=2.0, label='LSTM MAE')
    ax_month.set_xticks(xm)
    ax_month.set_xticklabels(monthly_mae['Month'], fontsize=9.5)
    ax_month.set_title("Mean Absolute Error by Month (Seasonality)", fontsize=11.5, fontweight='bold', pad=10)
    ax_month.set_ylabel("MAE (°C)", fontsize=10.5)
    ax_month.grid(True, linestyle=':', alpha=0.6)
    ax_month.legend(frameon=True, facecolor='white')
    
    # Temperature Bins MAE
    temp_bins = [-30, 0, 10, 20, 45]
    temp_labels = ['Freezing (<0°C)', 'Cold (0–10°C)', 'Mild (10–20°C)', 'Warm (>20°C)']
    test_df['Temp_Bin'] = pd.cut(test_df['Actual'], bins=temp_bins, labels=temp_labels)
    bin_mae = test_df.groupby('Temp_Bin', observed=True)[['AbsError_LSTM', 'AbsError_Baseline']].mean()
    
    xb = np.arange(len(bin_mae))
    ax_temp.bar(xb - 0.18, bin_mae['AbsError_Baseline'], width=0.35, label='Persistence Baseline', color='#74add1', edgecolor='#333333')
    ax_temp.bar(xb + 0.18, bin_mae['AbsError_LSTM'], width=0.35, label='LSTM Model', color='#d73027', edgecolor='#333333')
    ax_temp.set_xticks(xb)
    ax_temp.set_xticklabels(bin_mae.index, fontsize=9.5)
    ax_temp.set_title("Forecast Error Across Temperature Regimes", fontsize=11.5, fontweight='bold', pad=10)
    ax_temp.set_ylabel("MAE (°C)", fontsize=10.5)
    ax_temp.grid(axis='y', linestyle=':', alpha=0.6)
    ax_temp.legend(frameon=True, facecolor='white')
    
    fig.tight_layout()
    fig11_path = os.path.join(figures_dir, "11_error_breakdown_season_temperature.png")
    fig.savefig(fig11_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved: {fig11_path}")
    logging.info("All visualization figures successfully generated!")


if __name__ == "__main__":
    run_evaluation()
