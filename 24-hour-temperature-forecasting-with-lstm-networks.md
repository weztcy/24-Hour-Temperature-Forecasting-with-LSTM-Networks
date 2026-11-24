# Portfolio Project Data — 24-Hour Temperature Forecasting with LSTM Networks

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/24_hour_temperature_forecasting/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image10.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "24-hour-temperature-forecasting-with-lstm-networks",
  "code": "PRO-38 // ML-23",
  "title": "24-Hour Temperature Forecasting with LSTM Networks",
  "category": "machine-learning",
  "categoryLabel": "Deep Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/24-Hour-Temperature-Forecasting-with-LSTM-Networks",
  "license": "MIT License",
  "banner": "/images/projects/24_hour_temperature_forecasting/banner.webp",
  "summary": "24-hour multivariate temperature forecasting using a 2-layer PyTorch LSTM, cutting test MAE by 10.44% over persistence on Jena telemetry.",
  "projectDescription": "An end-to-end PyTorch pipeline for 24-hour-ahead air temperature forecasting on 8 years of 10-minute Jena Climate telemetry. Combines gap-aware timeline segmentation, circular encodings, and a stacked LSTM regressor achieving 2.24 °C MAE and 0.8676 R² on 10,399 test samples.",
  "architectureSummary": "Raw CSV (420k rows) -> Deduplication & -9999 Cleanup -> 3 Continuous Segments -> Train-Only Imputer/Scaler -> 144x19 Windows -> 2-Layer LSTM (57,153 params) -> 24h Forecast & Streamlit UI",
  "techStack": [
    "Python",
    "PyTorch",
    "NumPy",
    "Pandas",
    "SciPy",
    "Scikit-Learn",
    "Matplotlib",
    "Seaborn",
    "Joblib",
    "Streamlit"
  ],
  "telemetryMatrix": {
    "projectType": "Deep Learning Time-Series Pipeline",
    "targetPlatform": "Python Runtime & Streamlit Dashboard",
    "primaryDomain": "Atmospheric Temperature Forecasting",
    "architecturePipeline": "2-Layer Stacked LSTM Regressor (57k Params)",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Multivariate Sequence Forecasting Pipeline",
    "learningParadigm": "Deep Learning",
    "modelDomain": "Meteorological Time Series",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Stacked LSTM Regressor",
      "Leakage Safe Preprocessing",
      "Streamlit Forecast Explorer"
    ]
  },
  "mlProblemContext": "Forecasting ambient air temperature 24 hours ahead is critical for power grid dispatch, heating load planning, and frost warning. Univariate baselines miss oncoming cold or warm fronts, whereas a multivariate LSTM integrates pressure, humidity, air density, and wind vectors to anticipate synoptic transitions before surface temperatures shift.",
  "mlDatasetOverview": {
    "profileBadge": "Jena Climate 2009–2016",
    "source": "Max Planck Institute for Biogeochemistry (Jena, Germany)",
    "modality": "Multivariate Time Series",
    "scale": "420,551 Raw Rows / 69,897 Sequence Windows",
    "inputShape": "144 Timesteps x 19 Features",
    "dataFormat": "CSV",
    "description": "Eight calendar years (2009-01-01 to 2017-01-01) of 10-minute meteorological station readings partitioned chronologically by target timestamp into 49,042 training (70%), 10,456 validation (15%), and 10,399 held-out test (15%) sequences with a 1-hour stride.",
    "provenanceNote": "Public benchmark meteorological archive stored locally at dataset/original/jena_climate_2009_2016.csv with deduplication and sensor-fault curation."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "19 Meteorological & Cyclical Channels",
    "spaceNote": "13 thermodynamic and wind sensors combined with 6 circular/cyclical temporal projections.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "T (degC) at t + 144",
      "type": "Continuous Scalar",
      "distribution": "-23.01 °C to +37.28 °C (Test Mean: 9.6 °C, Std: 7.8 °C)"
    },
    "inputSignals": [
      {
        "name": "T (degC) & Tpot (K)",
        "type": "Thermodynamic Numerical",
        "importanceOrRole": "Primary autoregressive & potential temperature state (r = 0.997)",
        "description": "Preceding 24-hour dry-bulb and potential temperature trajectory encoding diurnal heating and cooling momentum."
      },
      {
        "name": "rho (g/m**3) & p (mbar)",
        "type": "Atmospheric State",
        "importanceOrRole": "Strong inverse density coupling (r = -0.96) & barometric front precursor",
        "description": "Surface air density and barometric pressure signaling approaching low-pressure troughs and air-mass advections."
      },
      {
        "name": "VPmax, VPdef, Tdew & rh (%)",
        "type": "Moisture & Vapor Balance",
        "importanceOrRole": "High Clausius-Clapeyron correlation (r = 0.896 to 0.951)",
        "description": "Saturation vapor pressure, vapor deficit, dew point, and relative humidity capturing latent heat and condensation regimes."
      },
      {
        "name": "wv, max. wv, wd_sin & wd_cos",
        "type": "Circular Wind Vectors",
        "importanceOrRole": "Continuous 0/360° directional advection indicators",
        "description": "Wind velocity, gust speed, and sine/cosine directional projections replacing discontinuous raw degree angles."
      },
      {
        "name": "day_sin, day_cos, year_sin & year_cos",
        "type": "Cyclical Temporal Encodings",
        "importanceOrRole": "Diurnal (24h) and annual (365.25d) solar phase anchors (year_cos r = -0.76)",
        "description": "Periodic trigonometric encodings anchoring solar elevation and seasonal position across the 144-step window."
      }
    ],
    "dataCharacteristicsSummary": "Temperature displays strong 24-hour diurnal oscillation and annual solar seasonality, tightly coupled with air density (r = -0.96) and vapor pressure (r = 0.95). Exact duplicates (327), -9999 wind sentinels (38), and two multi-hour outages require causal segmentation."
  },
  "mlLearningApproach": {
    "paradigm": "Deep Learning — Multivariate Sequence-to-Scalar Regression",
    "methodologyDescription": "A 2-layer stacked PyTorch LSTM processes 144 consecutive 10-minute observations (24 hours) across 19 standardized channels, extracting the final hidden representation to regress scaled temperature 144 steps ahead and comparing directly against 24-hour persistence.",
    "justification": "Gated LSTM memory cells retain both short-term hourly trajectories and multi-day synoptic precursors without vanishing gradients, outperforming persistence extrapolation that lags whenever atmospheric fronts alter day-to-day temperatures."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Exact Deduplication & Sentinel Masking",
      "description": "Removed 327 exact duplicate rows from 420,551 raw observations, sorted timestamps chronologically, and replaced 18 wv and 20 max. wv -9999.0 sensor fault sentinels with NaN.",
      "badge": "Data Hygiene"
    },
    {
      "step": "02",
      "title": "Circular & Cyclical Feature Engineering",
      "description": "Transformed wind direction into wd_sin and wd_cos and projected Date Time onto diurnal (86,400s) and annual (365.25d) sine/cosine pairs, excluding raw timestamp strings and degree discontinuities.",
      "badge": "Feature Engineering"
    },
    {
      "step": "03",
      "title": "Causal Grid Fill & Timeline Segmentation",
      "description": "Reindexed onto a 420,768-step 10-minute grid, applied causal forward filling (limit=2 steps) for <=30-minute gaps, and partitioned across 16h and 3.1d station outages into 3 continuous segments.",
      "badge": "Gap Isolation"
    },
    {
      "step": "04",
      "title": "Training-Only Imputation & Standardization",
      "description": "Fitted SimpleImputer(strategy='median') and StandardScaler strictly on records preceding 2014-08-08 09:39:00 before transforming validation and test segments into serialized joblib artifacts.",
      "badge": "Zero Leakage"
    }
  ],
  "mlDataPreparationNote": "Every sequence window [t - 143, t] and target t + 144 must lie within the same continuous segment, and split assignment is governed strictly by target timestamp to eliminate future leakage.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "2-Layer Stacked PyTorch LSTM Regressor",
    "architectureSummary": "Accepts (B, 144, 19) tensors into a 2-layer LSTM (hidden size 64, inter-layer dropout 0.2), extracts the final timestep hidden vector h_144, and passes through Linear(64, 32) -> ReLU -> Dropout(0.1) -> Linear(32, 1) totaling 57,153 trainable parameters.",
    "configurationBadge": "57,153 Parameters",
    "hyperparameters": [
      {
        "name": "window_size / horizon / stride",
        "value": "144 / 144 / 6",
        "description": "24-hour input lookback, 24-hour forecast horizon, and 1-hour stride between origins."
      },
      {
        "name": "hidden_size / num_layers",
        "value": "64 / 2",
        "description": "Stacked recurrent representation capacity balancing temporal memory with low overfitting risk."
      },
      {
        "name": "dropout / head_dropout",
        "value": "0.20 / 0.10",
        "description": "Inter-layer LSTM dropout and dense regression head regularization."
      },
      {
        "name": "batch_size / max_epochs / patience",
        "value": "128 / 20 / 5",
        "description": "Mini-batch size with validation-monitored early stopping terminating at Epoch 12."
      },
      {
        "name": "learning_rate / weight_decay",
        "value": "1e-3 / 1e-5",
        "description": "Initial Adam step size with ReduceLROnPlateau (factor=0.5, patience=2) and L2 decay."
      },
      {
        "name": "grad_clip_max_norm / seed",
        "value": "5.0 / 42",
        "description": "Gradient norm clipping for recurrent stability and fixed seed across Python, NumPy, and PyTorch."
      }
    ],
    "objectiveCriterion": "Mean Squared Error (MSE) on Standardized Target",
    "optimizationSolver": "Adam Optimizer with ReduceLROnPlateau & Gradient Clipping",
    "trainingProcess": "Trained on CPU for 12 epochs (403.5s) with seed 42 and validation early stopping. Train MSE fell from 0.1680 to 0.1135 at Epoch 7, achieving best validation MSE of 0.1109 before LR halving (0.0005 to 0.00025) and early stopping preserved the optimal weights."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Chronological 70/15/15 Holdout & Persistence Benchmark",
    "description": "Model checkpoints were selected on the 10,456-sample chronological validation split and evaluated once on 10,399 held-out test samples (Oct 2015 – Dec 2016) in unscaled degrees Celsius against the 24-hour persistence baseline.",
    "metrics": [
      {
        "label": "TEST MAE",
        "value": "2.24 °C",
        "benchmark": "Baseline < 2.51°C"
      },
      {
        "label": "TEST RMSE",
        "value": "2.84 °C",
        "benchmark": "Baseline < 3.26°C"
      },
      {
        "label": "TEST R²",
        "value": "0.8676",
        "benchmark": "Baseline > 0.8261"
      },
      {
        "label": "VAL MSE",
        "value": "0.1109",
        "benchmark": "Target < 0.1200"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Benchmark KPIs & Forecast Explorer",
      "badge": "Interactive UI",
      "imageUrl": "/images/projects/24_hour_temperature_forecasting/image01.webp",
      "telemetryNote": "Displays 2.24 °C MAE (-10.4% vs baseline) and sample #1250 forecast (4.90 °C vs 4.30 °C actual)."
    },
    {
      "title": "Local 24-Hour Forecast Trajectory Comparison",
      "badge": "Time-Series UI",
      "imageUrl": "/images/projects/24_hour_temperature_forecasting/image02.webp",
      "telemetryNote": "24-hour context curve around 2015-12-11 18:50 comparing LSTM, Ground Truth, and Persistence."
    },
    {
      "title": "Model Architecture & Feature Specifications Tab",
      "badge": "System Specs",
      "imageUrl": "/images/projects/24_hour_temperature_forecasting/image03.webp",
      "telemetryNote": "Documents 2-layer LSTM (57,153 parameters), 144-step horizon, and 19 engineered input channels."
    },
    {
      "title": "Partition KDE, Residual Density & 1:1 Prediction Scatter",
      "badge": "Diagnostic Grid",
      "imageUrl": "/images/projects/24_hour_temperature_forecasting/image04.webp",
      "telemetryNote": "Test R² 0.8676 hexbin fit along 1:1 line and tighter LSTM residual std (2.84 °C vs 3.26 °C)."
    },
    {
      "title": "8-Year Temperature Chronology & Test Residuals",
      "badge": "Chronological Audit",
      "imageUrl": "/images/projects/24_hour_temperature_forecasting/image05.webp",
      "telemetryNote": "70/15/15 chronological split across 2009–2016 and stationary test residuals within ±5.68 °C (±2σ)."
    },
    {
      "title": "Full Test Tracking & July 2016 Diurnal Zoom",
      "badge": "Forecast Tracking",
      "imageUrl": "/images/projects/24_hour_temperature_forecasting/image06.webp",
      "telemetryNote": "Accurate reproduction of July 2016 diurnal peaks and troughs ahead of persistence lag."
    },
    {
      "title": "Seasonal Monthly & Temperature Regime MAE Breakdown",
      "badge": "Error Stratification",
      "imageUrl": "/images/projects/24_hour_temperature_forecasting/image07.webp",
      "telemetryNote": "LSTM outperforms persistence across all 12 months and Cold/Mild/Warm bins; freezing (<0 °C) remains challenging."
    },
    {
      "title": "Model Benchmark & Data Quality Remediation Grid",
      "badge": "Benchmark & Audit",
      "imageUrl": "/images/projects/24_hour_temperature_forecasting/image08.webp",
      "telemetryNote": "Summarizes 10.44% MAE / 12.76% RMSE gains alongside 327 duplicate removals and 3 continuous segments."
    },
    {
      "title": "LSTM Training & Validation MSE Curve",
      "badge": "Convergence Curve",
      "imageUrl": "/images/projects/24_hour_temperature_forecasting/image09.webp",
      "telemetryNote": "Best checkpoint captured at Epoch 7 (Val MSE 0.1109) before early stopping at Epoch 12."
    },
    {
      "title": "Engineered Meteorological Feature Correlation Heatmap",
      "badge": "Feature Analysis",
      "imageUrl": "/images/projects/24_hour_temperature_forecasting/image10.webp",
      "telemetryNote": "Highlights strong couplings: Tpot (1.00), VPmax (0.95), Tdew (0.90), rho (-0.96), and year_cos (-0.76)."
    }
  ],
  "mlVisualizationTelemetryNote": "Validation MSE (0.1109) and held-out test generalization (2.24 °C MAE, 0.8676 R²) remain closely aligned across 14 unseen months without temporal drift.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "The LSTM combines local temperature persistence with barometric pressure drops, vapor pressure deficit, and circular wind shifts to anticipate frontal advection 24 hours ahead. Errors stay lowest in stable summer regimes and rise during winter transitions and sub-zero inversions.",
    "keyDrivers": [
      {
        "factor": "Diurnal & Annual Cyclical Anchors (day_sin/cos, year_cos)",
        "impact": "Captures baseline solar radiative cycles (year_cos r = -0.76 with T) across all seasons."
      },
      {
        "factor": "Thermodynamic & Moisture Coupling (Tpot, VPmax, VPdef, Tdew)",
        "impact": "Strong Clausius-Clapeyron relationships (r = 0.896 to 0.997) stabilize 24h enthalpy trajectory."
      },
      {
        "factor": "Barometric Pressure & Circular Wind Vectors (p, wv, wd_sin/cos)",
        "impact": "Provides leading indicators of incoming warm or cold air-mass advections missed by persistence."
      }
    ],
    "limitations": [
      {
        "title": "Single-Station Spatial Horizon Limit",
        "description": "Point observations from one tower in Jena cannot detect synoptic systems 500 km upwind without gridded NWP or satellite fields."
      },
      {
        "title": "Sub-Zero & Mesoscale Shock Error",
        "description": "Error increases in Freezing (<0 °C) conditions (~3.08 °C MAE) and abrupt convective storms due to snow-albedo feedbacks and radiative inversions."
      },
      {
        "title": "Site-Specific Climate Calibration",
        "description": "Weights are calibrated to Thuringia's temperate continental climate and require fine-tuning before transfer to coastal or alpine stations."
      }
    ],
    "futureImprovements": [
      {
        "title": "Gridded Spatial Weather Integration",
        "target": "Incorporate ERA5 reanalysis or upwind station telemetry to reduce frontal passage spikes."
      },
      {
        "title": "Probabilistic Quantile & MC Dropout Intervals",
        "target": "Output calibrated 10th/90th percentile forecast bounds for risk-aware grid dispatch."
      },
      {
        "title": "Modern Sequence Architecture Benchmarking",
        "target": "Benchmark Temporal Convolutional Networks (TCN) and PatchTST against the 2-layer LSTM baseline."
      }
    ],
    "discoveryNote": "Error stratification across months and temperature bins pinpoints winter frontal passages and freezing radiative inversions as primary residual drivers."
  },
  "mlModelArtifacts": [
    {
      "name": "best_lstm_model.pt",
      "format": "PT"
    },
    {
      "name": "preprocessing_objects.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "metrics.json",
      "format": "JSON"
    },
    {
      "name": "training_history.csv",
      "format": "CSV"
    },
    {
      "name": "test_predictions.csv",
      "format": "CSV"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Deep Learning",
    "modelArchitecture": "Stacked PyTorch LSTM",
    "coreFramework": "PyTorch",
    "validationMethod": "Chronological Holdout Benchmark"
  }
}
```
