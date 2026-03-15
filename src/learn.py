"""
src/hyperparameter_analysis.py
─────────────────────────────────────────────────────────────────────────────
Sweeps three LSTM hyperparameters one at a time (others held at default).
Outputs a printed table + one combined matplotlib window.

What is "Stacked Layers (Sequence Depth)" / num_stacked_layers?
───────────────────────────────────────────────────────────────
An LSTM unrolls over TIME along the sequence (lookback window), but it can
also be stacked VERTICALLY: the hidden-state output of layer k feeds in as
the *input* to layer k+1 at every time step.  This is the "depth" of the RNN.

  Input ──► [LSTM layer 1] ──► [LSTM layer 2] ──► ... ──► [LSTM layer n] ──► FC ──► prediction

More layers let the model learn hierarchical temporal abstractions, but
each extra layer adds hidden_size² parameters and deepens the gradient path,
making vanishing/exploding gradients more likely.  In practice the sweet-spot
for univariate time-series is 1–4 layers; values beyond ~8 rarely help.
The user-specified range [2, 512] is therefore sampled at the practical
subset [1, 2, 3, 4, 6, 8].
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc

# ─────────────────────────────────────────────────────────────────────────────
# Data  (identical pre-processing to test.py)
# ─────────────────────────────────────────────────────────────────────────────
data_path = Path(__file__).resolve().parents[1] / 'dataset' / 'Amazon_stock_data.csv'
data      = pd.read_csv(data_path)
device    = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop=True)

LOOKBACK = 30

def prepare_dataframe_for_lstm(df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    df = dc(df)
    df = df[['Close']]
    df['Close'] = np.log(df['Close'])
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

shifted_df   = prepare_dataframe_for_lstm(data, LOOKBACK)
shifted_np   = shifted_df.to_numpy()
num_features = shifted_np.shape[1]

scaler    = MinMaxScaler(feature_range=(-1, 1))
scaled_np = scaler.fit_transform(shifted_np)

X = dc(np.flip(scaled_np[:, 1:], axis=1))
Y = scaled_np[:, 0]

train_split = int(len(X) * 0.70)
test_split  = int(len(X) * 0.85)

X_train = X[:train_split].reshape((-1, LOOKBACK, 1))
X_test  = X[train_split:test_split].reshape((-1, LOOKBACK, 1))
X_val   = X[test_split:].reshape((-1, LOOKBACK, 1))
Y_train = Y[:train_split].reshape((-1, 1))
Y_test  = Y[train_split:test_split].reshape((-1, 1))
Y_val   = Y[test_split:].reshape((-1, 1))

to_t = lambda a: torch.tensor(a).float()
X_train_t, Y_train_t = to_t(X_train), to_t(Y_train)
X_test_t,  Y_test_t  = to_t(X_test),  to_t(Y_test)
X_val_t,   Y_val_t   = to_t(X_val),   to_t(Y_val)

def inverse_transform_col0(arr: np.ndarray) -> np.ndarray:
    dummy = np.zeros((len(arr), num_features))
    dummy[:, 0] = arr.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 16

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y
    def __len__(self):         return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.Y[i]

train_loader = DataLoader(TimeSeriesDataset(X_train_t, Y_train_t), BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TimeSeriesDataset(X_test_t,  Y_test_t),  BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class LSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.h = hidden_size
        self.n = num_layers
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        bs = x.size(0)
        h0 = torch.zeros(self.n, bs, self.h, device=device)
        c0 = torch.zeros(self.n, bs, self.h, device=device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# ─────────────────────────────────────────────────────────────────────────────
# Train one configuration and return validation metrics (original USD scale)
# ─────────────────────────────────────────────────────────────────────────────
loss_fn = nn.MSELoss()

def train_and_evaluate(lr: float, hidden_size: int, num_layers: int,
                        num_epochs: int = 10) -> dict:
    model = LSTM(hidden_size, num_layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(num_epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss_fn(model(x_b), y_b).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_val_t.to(device)).cpu().numpy()

    y_true = np.exp(inverse_transform_col0(Y_val))
    y_pred = np.exp(inverse_transform_col0(pred))

    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# ─────────────────────────────────────────────────────────────────────────────
# Sweep configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_LR     = 1e-3
DEFAULT_HIDDEN = 64
DEFAULT_LAYERS = 2
SWEEP_EPOCHS   = 10    # raise for more stable results; lower = faster

sweep_config = [
    {
        'label':   'Learning Rate',
        'param':   'lr',
        'values':  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        'default': DEFAULT_LR,
        'xscale':  'log',
        'fmt':     lambda v: f'{v:.0e}',
    },
    {
        'label':   'Hidden Size',
        'param':   'hidden_size',
        'values':  [4, 16, 64, 128, 256, 512],
        'default': DEFAULT_HIDDEN,
        'xscale':  'log',
        'fmt':     lambda v: str(int(v)),
    },
    {
        'label':   'Stacked Layers (Seq. Depth)',
        'param':   'num_layers',
        'values':  [1, 2, 3, 4, 6, 8],
        'default': DEFAULT_LAYERS,
        'xscale':  'linear',
        'fmt':     lambda v: str(int(v)),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Run sweeps
# ─────────────────────────────────────────────────────────────────────────────
print(f'Device : {device}')
print(f'Running hyperparameter sweeps  ({SWEEP_EPOCHS} epochs each)\n')
print(f'  {"Param":<35} {"Value":>10}    RMSE      MAE    MAPE%')
print('-' * 70)

table_rows = []
for cfg in sweep_config:
    cfg['results'] = []
    for v in cfg['values']:
        kwargs = dict(lr=DEFAULT_LR, hidden_size=DEFAULT_HIDDEN, num_layers=DEFAULT_LAYERS)
        kwargs[cfg['param']] = v
        m = train_and_evaluate(**kwargs, num_epochs=SWEEP_EPOCHS)
        cfg['results'].append({'value': v, **m})
        print(f"  {cfg['label']:<35} {cfg['fmt'](v):>10}  "
              f"{m['RMSE']:8.2f}  {m['MAE']:8.2f}  {m['MAPE']:6.2f}")
        table_rows.append({
            'Hyperparameter': cfg['label'],
            'Value':          cfg['fmt'](v),
            'MSE':            round(m['MSE'],  4),
            'RMSE':           round(m['RMSE'], 4),
            'MAE':            round(m['MAE'],  4),
            'MAPE (%)':       round(m['MAPE'], 4),
        })
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Printed summary table
# ─────────────────────────────────────────────────────────────────────────────
table_df = pd.DataFrame(table_rows)
pd.options.display.float_format = '{:.4f}'.format
print('=' * 72)
print('HYPERPARAMETER SWEEP – SUMMARY TABLE')
print('=' * 72)
print(table_df.to_string(index=False))
print('=' * 72)

# ─────────────────────────────────────────────────────────────────────────────
# Combined figure
#
#  Layout
#  ┌──────────────────────────────────────────────────────────────────────────┐
#  │  row 0  │  LR  → MSE  │  LR  → RMSE  │  LR  → MAE  │  LR  → MAPE %   │
#  │  row 1  │  Hidden→MSE │  Hidden→RMSE │  Hidden→MAE │  Hidden→MAPE %  │
#  │  row 2  │  Depth→MSE  │  Depth→RMSE  │  Depth→MAE  │  Depth→MAPE %   │
#  ├──────────────────────────────────────────────────────────────────────────┤
#  │  row 3  │              Summary table (all configurations)               │
#  └──────────────────────────────────────────────────────────────────────────┘
# ─────────────────────────────────────────────────────────────────────────────
METRICS  = ['MSE', 'RMSE', 'MAE', 'MAPE']
M_LABELS = ['MSE  (USD²)', 'RMSE  (USD)', 'MAE  (USD)', 'MAPE  (%)']
COLORS   = ['#e74c3c', '#e67e22', '#3498db', '#27ae60']

n_params   = len(sweep_config)
n_tbl_rows = len(table_rows)

fig = plt.figure(figsize=(22, 5 * n_params + 0.55 * n_tbl_rows + 3))
fig.suptitle(
    'LSTM Hyperparameter Sensitivity Analysis\n'
    f'defaults  →  LR={DEFAULT_LR},  hidden={DEFAULT_HIDDEN},  '
    f'stacked layers={DEFAULT_LAYERS},  epochs={SWEEP_EPOCHS}',
    fontsize=13, fontweight='bold', y=0.995,
)

outer = gridspec.GridSpec(
    n_params + 1, 1, figure=fig,
    hspace=0.75,
    height_ratios=[4] * n_params + [0.55 * n_tbl_rows + 1],
)

for row_i, cfg in enumerate(sweep_config):
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[row_i], wspace=0.38)
    xs = [r['value'] for r in cfg['results']]

    for col_i, (metric, mlabel, color) in enumerate(zip(METRICS, M_LABELS, COLORS)):
        ys = [r[metric] for r in cfg['results']]
        ax = fig.add_subplot(inner[col_i])

        ax.plot(xs, ys, marker='o', color=color, linewidth=2, markersize=6, zorder=3)
        ax.axvline(cfg['default'], color='#95a5a6', linestyle='--', linewidth=1.3,
                   label=f"default={cfg['fmt'](cfg['default'])}")

        best_i = int(np.argmin(ys))
        ax.plot(xs[best_i], ys[best_i], marker='*', color='gold',
                markersize=14, zorder=5, label=f"best={cfg['fmt'](xs[best_i])}")

        ax.set_xscale(cfg['xscale'])
        ax.set_xticks(xs)
        ax.set_xticklabels([cfg['fmt'](v) for v in xs], rotation=40, ha='right', fontsize=7)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f'{cfg["label"]}  →  {mlabel}', fontsize=8.5)
        ax.set_xlabel(cfg['label'], fontsize=8)
        ax.set_ylabel(mlabel, fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.legend(fontsize=7)

# ── embedded summary table ───────────────────────────────────────────────────
ax_t = fig.add_subplot(outer[n_params])
ax_t.axis('off')

col_labels = list(table_df.columns)
cell_data  = [[str(c) for c in row] for row in table_df.values.tolist()]

tbl = ax_t.table(cellText=cell_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.35)

# Colour header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row shading for readability
for i in range(1, len(cell_data) + 1):
    fc = '#eaf2ff' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(fc)

plt.show()