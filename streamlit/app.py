"""
app.py
------
Streamlit interactive web application for 24-Hour Temperature Forecasting with LSTM.
Provides interactive sequence inspection, 24-hour ahead temperature prediction,
persistence baseline comparison, and evaluation diagnostics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model import LSTMRegressor
import joblib

st.set_page_config(
    page_title="24-Hour Temperature Forecasting | LSTM",
    page_icon="🌡️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2b5c8f;
        margin-bottom: 12px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 16px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_prep():
    prep_path = "models/preprocessing_objects.joblib"
    model_path = "models/best_lstm_model.pt"
    
    if not os.path.exists(prep_path) or not os.path.exists(model_path):
        return None, None, None
        
    prep = joblib.load(prep_path)
    device = torch.device("cpu")
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
    
    return model, prep, checkpoint


@st.cache_data
def load_test_predictions():
    pred_path = "outputs/test_predictions.csv"
    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path)
        df['Date Time'] = pd.to_datetime(df['Date Time'])
        return df
    return None


@st.cache_data
def load_metrics():
    metrics_path = "outputs/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def main():
    st.title("🌡️ 24-Hour Temperature Forecasting with LSTM")
    st.markdown("""
    **Deep Learning — Multivariate Time-Series Forecasting on Jena Climate**  
    Predicts atmospheric air temperature (`T (degC)`) exactly **24 hours into the future** using the preceding **24 hours of weather observations** (144 timesteps at 10-minute resolution).
    """)
    
    model, prep, checkpoint = load_model_and_prep()
    metrics = load_metrics()
    test_df = load_test_predictions()
    
    if model is None or prep is None:
        st.error("Model artifacts or preprocessing objects not found in `models/`. Please train the model first.")
        return

    # Top KPI Metrics Row
    st.subheader("Model Performance Benchmark (Held-Out Test Set)")
    col1, col2, col3, col4 = st.columns(4)
    
    if metrics:
        lstm_mae = metrics['lstm_metrics']['mae_degC']
        base_mae = metrics['baseline_metrics']['mae_degC']
        mae_imp = metrics['relative_improvement_pct']['mae_reduction_pct']
        
        lstm_rmse = metrics['lstm_metrics']['rmse_degC']
        base_rmse = metrics['baseline_metrics']['rmse_degC']
        rmse_imp = metrics['relative_improvement_pct']['rmse_reduction_pct']
        
        lstm_r2 = metrics['lstm_metrics']['r2_score']
        base_r2 = metrics['baseline_metrics']['r2_score']
        
        col1.metric("LSTM Test MAE", f"{lstm_mae:.2f} °C", f"-{mae_imp:.1f}% vs Baseline", delta_color="inverse")
        col2.metric("LSTM Test RMSE", f"{lstm_rmse:.2f} °C", f"-{rmse_imp:.1f}% vs Baseline", delta_color="inverse")
        col3.metric("LSTM Test R²", f"{lstm_r2:.4f}", f"+{(lstm_r2 - base_r2):.4f} vs Baseline")
        col4.metric("Baseline MAE", f"{base_mae:.2f} °C", "Persistence T(t)")
    
    st.markdown("---")
    
    # Interactive Tab Navigation
    tab1, tab2, tab3 = st.tabs(["🔍 Interactive Forecast Inspector", "📊 Model Comparison & Diagnostics", "ℹ️ Architecture & Data Specs"])
    
    with tab1:
        st.subheader("Interactive 24-Hour Forecast Explorer")
        st.write("Select an observation timestamp from the held-out test partition to inspect the 24-hour prediction.")
        
        if test_df is not None:
            min_date = test_df['Date Time'].min().date()
            max_date = test_df['Date Time'].max().date()
            
            col_sel1, col_sel2 = st.columns([2, 1])
            with col_sel1:
                selected_idx = st.slider(
                    "Select Test Observation Index:",
                    min_value=0,
                    max_value=len(test_df) - 1,
                    value=min(1250, len(test_df) - 1),
                    step=1
                )
            
            sample_row = test_df.iloc[selected_idx]
            target_time = sample_row['Date Time']
            actual_temp = sample_row['Actual']
            lstm_pred = sample_row['LSTM_Pred']
            baseline_pred = sample_row['Persistence_Baseline']
            
            with col_sel2:
                st.info(f"**Target Forecast Time:**  \n📅 `{target_time.strftime('%Y-%m-%d %H:%M:%S')}`")
            
            # Forecast Result Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ground Truth Actual", f"{actual_temp:.2f} °C")
            c2.metric("LSTM Forecast (t + 24h)", f"{lstm_pred:.2f} °C", f"{lstm_pred - actual_temp:+.2f} °C error")
            c3.metric("Persistence Forecast", f"{baseline_pred:.2f} °C", f"{baseline_pred - actual_temp:+.2f} °C error")
            c4.metric("LSTM Absolute Error", f"{abs(lstm_pred - actual_temp):.2f} °C", 
                      f"-{abs(baseline_pred - actual_temp) - abs(lstm_pred - actual_temp):.2f} °C advantage", delta_color="inverse")
            
            # Context window plot
            st.markdown("#### Local Context: 24-Hour Horizon Comparison")
            # Show window around selected index (+/- 24 hours)
            window_slice = test_df.iloc[max(0, selected_idx - 72) : min(len(test_df), selected_idx + 72)]
            
            fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)
            ax.plot(window_slice['Date Time'], window_slice['Actual'], label='Ground Truth Actual', color='#222222', lw=1.8)
            ax.plot(window_slice['Date Time'], window_slice['LSTM_Pred'], label='LSTM 24h Forecast', color='#d62728', lw=1.8)
            ax.plot(window_slice['Date Time'], window_slice['Persistence_Baseline'], label='Persistence Baseline', color='#1f77b4', lw=1.2, linestyle=':')
            
            # Highlight selected point
            ax.scatter([target_time], [actual_temp], color='#222222', s=90, zorder=6, label='Selected Actual')
            ax.scatter([target_time], [lstm_pred], color='#d62728', s=90, zorder=6, marker='s', label='Selected LSTM Forecast')
            ax.axvline(target_time, color='#888888', linestyle='--', alpha=0.7)
            
            ax.set_ylabel("Air Temperature (°C)")
            ax.set_title(f"24-Hour Forecast Trajectory around {target_time.strftime('%Y-%m-%d %H:%M')}", fontweight='bold')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower left', frameon=True, facecolor='white', ncol=3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        st.subheader("Model Diagnostic Visualizations")
        fig_options = {
            "Model Comparison Benchmark": "outputs/figures/10_model_comparison.png",
            "Actual vs. Predicted Chronology": "outputs/figures/06_actual_vs_predicted_chronological.png",
            "Prediction Scatter Plot (1:1 Fit)": "outputs/figures/07_prediction_scatter_plot.png",
            "Residual Error Distribution": "outputs/figures/08_residual_distribution.png",
            "Chronological Residuals": "outputs/figures/09_residuals_over_time.png",
            "Seasonal and Regime Error Breakdown": "outputs/figures/11_error_breakdown_season_temperature.png",
            "Feature Correlation Heatmap": "outputs/figures/03_feature_correlation_heatmap.png",
            "Data Quality Remediation": "outputs/figures/04_data_quality_issues.png",
            "Training History": "outputs/figures/05_training_validation_history.png"
        }
        
        selected_fig = st.selectbox("Select Diagnostic Figure:", list(fig_options.keys()))
        fig_path = fig_options[selected_fig]
        if os.path.exists(fig_path):
            st.image(fig_path, caption=selected_fig, use_container_width=True)
        else:
            st.warning(f"Figure file `{fig_path}` will be available once evaluation completes.")

    with tab3:
        st.subheader("Architecture, Features & Reproducibility Specifications")
        col_spec1, col_spec2 = st.columns(2)
        with col_spec1:
            st.markdown("""
            **Model Architecture:**
            - **Type:** PyTorch Stacked LSTM Regressor
            - **Layers:** 2 LSTM layers
            - **Hidden Dimension:** 64 units per layer
            - **Inter-layer Dropout:** 0.20
            - **Head:** Linear(64, 32) -> ReLU -> Dropout(0.1) -> Linear(32, 1)
            - **Trainable Parameters:** 57,153
            - **Loss Function:** Mean Squared Error (MSE)
            - **Optimizer:** Adam (Initial LR: 1e-3, ReduceLROnPlateau)
            """)
        with col_spec2:
            st.markdown("""
            **Input Sequence & Features:**
            - **Input Window:** 144 timesteps (24 hours at 10-minute cadence)
            - **Forecast Horizon:** 144 timesteps (24 hours ahead)
            - **Origin Stride:** 6 timesteps (1 hour)
            - **Features (19):**
              - Raw meteorological measurements (13)
              - Circular wind direction: `wd_sin`, `wd_cos`
              - Cyclical diurnal time: `day_sin`, `day_cos`
              - Cyclical annual time: `year_sin`, `year_cos`
            - **Scaling:** Fit strictly on Training partition (no test leakage)
            """)

    st.markdown("---")
    st.caption("24-Hour Temperature Forecasting Project | Built with PyTorch and Streamlit")


if __name__ == "__main__":
    main()
