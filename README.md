# LSTM Amazon Stock Price Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

A PyTorch implementation of a stacked **Long Short-Term Memory (LSTM)** network for predicting Amazon (AMZN) stock closing prices. The project covers the complete machine-learning pipeline — data preprocessing, model training with early stopping, evaluation, and a systematic hyperparameter sensitivity analysis.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Usage](#usage)
   - [Full Training Run](#full-training-run)
   - [Hyperparameter Sweep](#hyperparameter-sweep)
   - [Jupyter Notebook](#jupyter-notebook)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
8. [Documentation](#documentation)
9. [Contributing](#contributing)
10. [License](#license)

---

## Features

- **End-to-end pipeline** — raw CSV → log-scaled lagged features → MinMaxScaler → stacked LSTM → USD price predictions
- **GPU-accelerated training** — automatic CUDA detection; falls back to CPU seamlessly
- **Early stopping** — patience-based mechanism restores the best model weights automatically
- **Four evaluation metrics** — MSE, RMSE, MAE, and MAPE computed on the original USD price scale
- **Hyperparameter sensitivity analysis** — systematic grid sweep over learning rate, hidden size, and number of stacked layers with rich multi-panel visualization
- **Reproducible data paths** — all paths are resolved relative to the project root; no hardcoded machine-specific paths

---

## Project Structure

```
LSTM_air_prediction/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── .gitignore
├── requirements.txt
│
├── dataset/
│   └── Amazon_stock_data.csv       # Historical AMZN OHLCV data (1997 – present)
│
├── docs/
│   ├── architecture.md             # LSTM model design & forward pass
│   ├── data_preprocessing.md       # Feature engineering & normalization pipeline
│   ├── hyperparameter_tuning.md    # Sweep methodology & how to read the plots
│   ├── training_guide.md           # Step-by-step training & configuration reference
│   └── api_reference.md            # All public functions & classes
│
├── figs/                           # Output figures (generated at runtime)
│
└── src/
    ├── LSTM_training.py            # Main training script
    ├── hyper_LSTM.py               # Hyperparameter sweep script
    └── notebooks/
        └── non_statefull_training.ipynb   # Interactive Jupyter walkthrough
```

---

## Installation

### Prerequisites

- Python 3.10 or later
- NVIDIA GPU with CUDA 12.4 *(optional — CPU training is supported)*

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/RaulDan2001/LSTM_air_prediction.git
cd LSTM_air_prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` includes the CUDA 12.4 build of PyTorch. If you do not have a compatible GPU, install the CPU-only build instead:
> ```bash
> pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
> ```
> Then install the remaining dependencies:
> ```bash
> pip install pandas==3.0.1 numpy==2.3.5 matplotlib==3.10.8 scikit-learn==1.8.0 scipy==1.17.1
> ```

---

## Dataset

| Field   | Detail                                      |
|---------|---------------------------------------------|
| File    | `dataset/Amazon_stock_data.csv`             |
| Columns | `Date`, `Close`, `High`, `Low`, `Open`, `Volume` |
| Range   | 1997-05-15 → present                        |
| Source  | Amazon (AMZN) daily historical OHLCV data   |

Only the `Close` column is used as the prediction target. See [docs/data_preprocessing.md](docs/data_preprocessing.md) for the complete preprocessing pipeline.

---

## Usage

### Full Training Run

```bash
python src/LSTM_training.py
```

This script will:
1. Load and preprocess the dataset
2. Build a stacked LSTM model (hidden size 128, 1 layer by default)
3. Train for up to 120 epochs with early stopping (patience = 5)
4. Print validation metrics (MSE, RMSE, MAE, MAPE) in original USD price scale
5. Display a composite figure: actual vs predicted prices, epoch loss curves, and metric bar charts

**Key configurable constants** (top of `LSTM_training.py`):

| Constant         | Default | Description                             |
|------------------|---------|-----------------------------------------|
| `lookback`       | `30`    | Number of past time steps as features   |
| `batch_size`     | `16`    | Training batch size                     |
| `learning_rate`  | `1e-3`  | Adam optimizer learning rate            |
| `num_epochs`     | `120`   | Maximum number of training epochs       |
| `patience`       | `5`     | Early stopping patience                 |
| `min_delta`      | `1e-4`  | Minimum validation loss improvement     |

### Hyperparameter Sweep

```bash
python src/hyper_LSTM.py
```

Runs a one-at-a-time sensitivity analysis over three hyperparameters:

| Hyperparameter  | Values swept                      | Scale  |
|-----------------|-----------------------------------|--------|
| Learning Rate   | `1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0` | Log  |
| Hidden Size     | `4, 16, 64, 128, 256, 512`        | Log    |
| Stacked Layers  | `1, 2, 3, 4, 6, 8`               | Linear |

Each configuration is trained for 10 epochs. Results are printed as a formatted table and displayed as a 12-panel figure (3 sweeps × 4 metrics). See [docs/hyperparameter_tuning.md](docs/hyperparameter_tuning.md) for full details.

### Jupyter Notebook

```bash
jupyter notebook src/notebooks/non_statefull_training.ipynb
```

An interactive walkthrough of the full pipeline — data loading, preprocessing, model definition, a 20-epoch training loop, and visualization of actual vs. predicted prices with a date-labelled x-axis.

---

## Model Architecture

```
Input sequence: (batch_size, 30, 1)
        │
        ▼
┌──────────────────────────────────────────────────┐
│  nn.LSTM(input_size=1, hidden_size=H, num_layers=L) │
│  h₀, c₀ initialized to zeros each forward pass   │
└──────────────────────────────────────────────────┘
        │  last time-step hidden state: (batch_size, H)
        ▼
┌──────────────┐
│  nn.Linear(H, 1) │
└──────────────┘
        │
        ▼
Predicted Close (normalized) → inverse transform → USD price
```

- **`H`** — hidden size (default 128 in training script, 64 in sweep defaults)
- **`L`** — number of stacked LSTM layers (default 1 in training script, 2 in sweep defaults)
- Hidden and cell states are **reset to zeros at the start of every batch** (non-stateful / stateless mode)

For a deeper explanation see [docs/architecture.md](docs/architecture.md).

---

## Results

> Results below are illustrative placeholders. Run `src/LSTM_training.py` and fill in your own numbers.

| Metric | Value       |
|--------|-------------|
| MSE    | — USD²      |
| RMSE   | — USD       |
| MAE    | — USD       |
| MAPE   | — %         |

All metrics are computed on the **validation set** (last 15 % of the chronologically sorted data) at the original USD price scale after inverse-transforming and exponentiating the log-scaled predictions.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | LSTM class design, forward pass, early stopping |
| [docs/data_preprocessing.md](docs/data_preprocessing.md) | Feature engineering, normalization, train/val/test splits |
| [docs/hyperparameter_tuning.md](docs/hyperparameter_tuning.md) | Sweep methodology, reading plots, choosing hyperparameters |
| [docs/training_guide.md](docs/training_guide.md) | Step-by-step guide to run and configure training |
| [docs/api_reference.md](docs/api_reference.md) | All public functions and classes with parameters and examples |

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
