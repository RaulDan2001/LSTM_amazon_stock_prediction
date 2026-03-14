from pathlib import Path
from copy import deepcopy as dc

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    # Reproducibility: makes training runs comparable when changing code/hyperparameters.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Read dataset from project-relative path to avoid hardcoded machine-specific paths.
data_path = Path(__file__).resolve().parents[1] / 'dataset' / 'Amazon_stock_data.csv'
data = pd.read_csv(data_path)

data['Date'] = pd.to_datetime(data['Date'])
# Time series models must respect chronology.
data = data.sort_values('Date').reset_index(drop=True)


def prepare_dataframe_for_lstm(df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    df = dc(df)
    df = df[['Date', 'Close']]

    # Work in log-space and returns:
    # - log prices reduce scale effects
    # - returns are usually more stationary than raw prices
    df['LogClose'] = np.log(df['Close'])
    df['Return'] = df['LogClose'].diff()

    # Previous close is needed to map predicted returns back to price.
    df['PrevClose'] = df['Close'].shift(1)

    # Build lagged features: Return(t-1), Return(t-2), ..., Return(t-n_steps)
    # so the LSTM gets a rolling history window.
    for i in range(1, n_steps + 1):
        df[f'Return(t-{i})'] = df['Return'].shift(i)

    # First rows have NaNs due to differencing and shifts.
    df.dropna(inplace=True)
    return df


lookback = 30
shifted_df = prepare_dataframe_for_lstm(data, lookback)

# X uses past returns, Y is next return
X_raw = shifted_df[[f'Return(t-{i})' for i in range(lookback, 0, -1)]].to_numpy()
Y_raw = shifted_df['Return'].to_numpy().reshape(-1, 1)
prev_close_raw = shifted_df['PrevClose'].to_numpy().reshape(-1, 1)
dates_raw = shifted_df['Date'].to_numpy()

train_end = int(len(X_raw) * 0.70)
val_end = int(len(X_raw) * 0.85)

# Chronological split (no shuffling between sets) avoids leakage from future to past.
X_train_raw, X_val_raw, X_test_raw = X_raw[:train_end], X_raw[train_end:val_end], X_raw[val_end:]
Y_train_raw, Y_val_raw, Y_test_raw = Y_raw[:train_end], Y_raw[train_end:val_end], Y_raw[val_end:]
prev_close_train, prev_close_val, prev_close_test = (
    prev_close_raw[:train_end],
    prev_close_raw[train_end:val_end],
    prev_close_raw[val_end:],
)
test_dates = dates_raw[val_end:]

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit scalers on train only, then reuse on val/test to keep evaluation fair.
X_train = scaler_X.fit_transform(X_train_raw)
X_val = scaler_X.transform(X_val_raw)
X_test = scaler_X.transform(X_test_raw)

Y_train = scaler_Y.fit_transform(Y_train_raw)
Y_val = scaler_Y.transform(Y_val_raw)
Y_test = scaler_Y.transform(Y_test_raw)

X_train = X_train.reshape((-1, lookback, 1))
X_val = X_val.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

Y_train = Y_train.reshape((-1, 1))
Y_val = Y_val.reshape((-1, 1))
Y_test = Y_test.reshape((-1, 1))

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32)


class TimeSeriesDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


train_dataset = TimeSeriesDataset(X_train_t, Y_train_t)
val_dataset = TimeSeriesDataset(X_val_t, Y_val_t)
test_dataset = TimeSeriesDataset(X_test_t, Y_test_t)

batch_size = 32
# Shuffle only training batches. Val/test must preserve order for stable evaluation.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_stacked_layers: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_stacked_layers,
            batch_first=True,
            dropout=dropout if num_stacked_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Stateless training: hidden/cell state reset for each batch.
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Use last time step representation for one-step-ahead regression.
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


model = LSTM(1, 128, 2, dropout=0.3).to(device)
print(model)

learning_rate = 1e-3
num_epochs = 120
patience = 12
loss_function = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# Reduce LR when validation stalls to improve convergence near minima.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)


def train_one_epoch(epoch_idx: int) -> float:
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)

        optimizer.zero_grad()
        loss.backward()
        # Protect against exploding gradients in recurrent networks.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch_idx + 1:03d} | Train Loss: {avg_train_loss:.5f}')
    return avg_train_loss


def evaluate(loader: DataLoader) -> float:
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    return running_loss / len(loader)


best_val_loss = float('inf')
best_state = None
epochs_without_improvement = 0

for epoch in range(num_epochs):
    train_one_epoch(epoch)
    val_loss = evaluate(val_loader)
    scheduler.step(val_loss)

    print(f'           | Val Loss:   {val_loss:.5f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    print('------------------------------------------')

    if val_loss < best_val_loss:
        # Keep the best checkpoint based on validation, not last epoch.
        best_val_loss = val_loss
        best_state = dc(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f'Early stopping at epoch {epoch + 1}. Best val loss: {best_val_loss:.5f}')
        break

if best_state is not None:
    model.load_state_dict(best_state)

test_loss = evaluate(test_loader)
print(f'Test Loss (scaled): {test_loss:.5f}')

model.eval()
with torch.no_grad():
    predicted_scaled = model(X_test_t.to(device)).cpu().numpy()

actual_scaled = Y_test_t.cpu().numpy()

# Back to original price scale for meaningful metrics
predicted_return = scaler_Y.inverse_transform(predicted_scaled)
actual_return = scaler_Y.inverse_transform(actual_scaled)

# price_t = price_{t-1} * exp(log_return_t)
predicted_price = prev_close_test * np.exp(predicted_return)
actual_price = prev_close_test * np.exp(actual_return)

mae = np.mean(np.abs(predicted_price - actual_price))
rmse = np.sqrt(np.mean((predicted_price - actual_price) ** 2))

# Naive random-walk baseline: next close ~= previous close.
baseline_price = prev_close_test
baseline_mae = np.mean(np.abs(baseline_price - actual_price))
baseline_rmse = np.sqrt(np.mean((baseline_price - actual_price) ** 2))

# Positive values mean the model beats the baseline.
mae_improvement = baseline_mae - mae
rmse_improvement = baseline_rmse - rmse

print(f'Test MAE (price):  {mae:.4f}')
print(f'Test RMSE (price): {rmse:.4f}')
print(f'Baseline MAE:      {baseline_mae:.4f}')
print(f'Baseline RMSE:     {baseline_rmse:.4f}')
print(f'MAE improvement vs baseline:  {mae_improvement:.4f}')
print(f'RMSE improvement vs baseline: {rmse_improvement:.4f}')

plt.figure(figsize=(12, 5))
plt.plot(test_dates, actual_price.ravel(), label='Actual Close')
plt.plot(test_dates, predicted_price.ravel(), label='Predicted Close')
plt.plot(test_dates, baseline_price.ravel(), '--', alpha=0.7, label='Baseline (Prev Close)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Test Set Predictions (Original Price Scale)')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.text(
    0.02,
    0.05,
    f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nBase MAE: {baseline_mae:.2f}\nBase RMSE: {baseline_rmse:.2f}',
    transform=plt.gca().transAxes,
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
)
plt.tight_layout()
plt.show()