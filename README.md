
# Auto Fuel Efficiency Predictor

![CI](https://github.com/aishvadla/auto-fuel-efficiency-predictor/actions/workflows/ci.yaml/badge.svg)

A **production-grade ML pipeline** that predicts vehicle fuel efficiency (MPG) using a Deep Neural Network — built with engineering best practices including modular architecture, config-driven training, artifact management, REST API serving, and CI/CD.

> This project serves as a reusable ML pipeline template applied to the UCI Auto MPG dataset. The focus is on **production engineering patterns**, not model complexity.

---

## Results

| Metric | Score |
|--------|-------|
| Test R² | **0.86** |
| Test MAE | **2.38 MPG** |
| Test MSE | 10.15 |

Model explains 86% of variance in fuel efficiency on held-out test data. Predictions are off by ~2.4 MPG on average.

---

## Architecture

```
Raw Data (UCI Auto MPG)
        │
        ▼
┌─────────────────┐
│  Data Ingestion │  Downloads, cleans, three-way splits (train/val/test)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Data Preprocessing  │  ColumnTransformer: StandardScaler + OneHotEncoder
│                      │  Saves preprocessor.pkl + scaler_y.pkl to artifacts
└────────┬─────────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │  Dynamic DNN (configurable layers via config.yaml)
│                 │  Adam optimizer, MSE loss, val loss tracked per epoch
│                 │  Saves model.pth to artifacts
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  Model Evaluate  │  MSE, MAE, R² on test set (original MPG scale)
│                  │  Saves metrics.json to artifacts
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  FastAPI REST API    │  POST /predict → returns predicted MPG
│  (app/app.py)        │  GET  /health  → service health check
└──────────────────────┘
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML Framework | PyTorch |
| Data Processing | Pandas, NumPy, Scikit-learn |
| API Serving | FastAPI, Uvicorn |
| Experiment Tracking | Config-driven (YAML) |
| Testing | Pytest with session-scoped fixtures |
| Code Quality | Black, Ruff |
| Environment | uv, pyproject.toml |
| CI/CD | GitHub Actions |

---

## Project Structure

```
auto-fuel-efficiency-predictor/
├── app/
│   └── app.py                    # FastAPI REST API
├── configs/
│   └── config.yaml               # All hyperparameters — single source of truth
├── src/
│   ├── data/
│   │   ├── ingestion.py          # Download, clean, split data
│   │   └── preprocessing.py      # Feature engineering pipeline
│   ├── models/
│   │   ├── model_registry.py     # Dynamic DNN architecture (nn.Module)
│   │   ├── train.py              # Training loop with validation tracking
│   │   └── evaluate.py           # Test set evaluation + metrics.json
│   ├── pipelines/
│   │   ├── training_pipeline.py  # End-to-end training orchestration
│   │   └── prediction_pipeline.py# Single-sample inference pipeline
│   └── utils/
│       ├── logger.py             # Structured logging
│       ├── exception.py          # Custom exception handling
│       └── helper.py             # save/load object utilities
├── tests/
│   ├── conftest.py               # Session-scoped pytest fixtures
│   ├── test_data_ingestion.py
│   ├── test_data_preprocessing.py
│   ├── test_model_registry.py
│   ├── test_model_trainer.py
│   └── test_prediction_pipeline.py
├── .github/workflows/ci.yaml     # CI pipeline — lint + test on every push
├── main.py                       # Entry point — runs full training pipeline
└── pyproject.toml                # Dependencies + build config (uv)
```

---

## Engineering Highlights

**Config-driven training** — all hyperparameters (hidden units, learning rate, epochs, batch size, train/val/test splits) live in `configs/config.yaml`. No hardcoded values in source code.

**Proper train/val/test separation** — three-way split with no data leakage. Preprocessing fitted on train only, applied to val and test.

**Target scaling** — `StandardScaler` applied to MPG target during training, inverse-transformed at evaluation and inference for interpretable metrics.

**Artifact management** — three artifacts saved at training time: `model.pth`, `preprocessor.pkl`, `scaler_y.pkl`. All three loaded at inference time ensuring identical preprocessing between training and serving.

**Dynamic architecture** — DNN hidden layer configuration via a list in config (`[64, 32]`). Change architecture by editing YAML, no code changes needed.

**Session-scoped test fixtures** — `conftest.py` runs ingestion and preprocessing once per test session, shared across all tests. Avoids redundant data loading.

**Production exception handling** — all pipeline stages wrapped in `CustomException` with full traceback context for debugging.

---

## Quick Start

**1. Clone and install:**
```bash
git clone https://github.com/aishvadla/auto-fuel-efficiency-predictor.git
cd auto-fuel-efficiency-predictor
uv venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

**2. Train the model:**
```bash
python main.py
```

**3. Start the API:**
```bash
python app/app.py
```

**4. Make a prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Cylinders": 4,
       "Displacement": 120.0,
       "Horsepower": 79.0,
       "Weight": 2625.0,
       "Acceleration": 18.6,
       "ModelYear": 82,
       "Origin": 1
     }'
```

**Response:**
```json
{"predicted_mpg": 29.61}
```

**5. Interactive API docs:**
```
http://localhost:8000/docs
```

**6. Run tests:**
```bash
pytest tests/ -v
```

---

## Configuration

All hyperparameters in `configs/config.yaml`:

```yaml
model:
  hidden_units: [64, 32]
  output_features: 1

training:
  epochs: 100
  batch_size: 32
  eta: 0.005
  log_epochs: 10

data:
  test_size: 0.15
  val_size: 0.15
  random_state: 42
```

---

## API Reference

### `POST /predict`

Predict fuel efficiency for a vehicle.

**Request body:**
```json
{
  "Cylinders": 4,
  "Displacement": 120.0,
  "Horsepower": 79.0,
  "Weight": 2625.0,
  "Acceleration": 18.6,
  "ModelYear": 82,
  "Origin": 1
}
```

**Response:**
```json
{"predicted_mpg": 29.61}
```

### `GET /health`
```json
{"status": "ok"}
```

---

## Dataset

[UCI Auto MPG Dataset](https://archive.ics.uci.edu/dataset/9/auto+mpg) — 392 samples, 7 features, predicting miles per gallon.

| Feature | Type | Preprocessing |
|---------|------|---------------|
| Cylinders | Numeric (discrete) | StandardScaler |
| Displacement | Numeric (continuous) | StandardScaler |
| Horsepower | Numeric (continuous) | StandardScaler + mean imputation |
| Weight | Numeric (continuous) | StandardScaler |
| Acceleration | Numeric (continuous) | StandardScaler |
| Model Year | Numeric (ordinal) | StandardScaler |
| Origin | Categorical (nominal) | OneHotEncoder (drop first) |

---

## Author

**Aishwarya Vadlamudi** — AI Software Architect transitioning to ML Engineering  
[GitHub](https://github.com/aishvadla) · [LinkedIn](https://www.linkedin.com/in/aishwaryavadlamudi/)
