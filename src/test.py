import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from copy import deepcopy as dc

# read the csv
# Read dataset from project-relative path to avoid hardcoded machine-specific paths.
data_path = Path(__file__).resolve().parents[1] / 'dataset' / 'Amazon_stock_data.csv'
data = pd.read_csv(data_path)

# use the gpu if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# handle the date column
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop=True)

def prepare_dataframe_for_lstm(df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    df = dc(df)
    # remove all the columns except the Closing Value
    df = df[['Close']]
    df['Close'] = np.log(data['Close'])

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 30
shifted_df = prepare_dataframe_for_lstm(data, lookback)

# covert the dataframe to np
shifted_df_as_np = shifted_df.to_numpy()
print(shifted_df_as_np)

# normalize the numpy array
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
print(shifted_df_as_np)

X = shifted_df_as_np[:, 1:] # all the rows, from the second column on to the end
X = dc(np.flip(X, axis=1)) # mirror the data horizontically
Y = shifted_df_as_np[:, 0] # all the rows only the first column

print(X.shape, Y.shape)

train_split = int(len(X) * 0.70)
test_split = int(len(X) * 0.85)

print(train_split, test_split)

# split the data into train and test groups
X_train = X[:train_split]
X_test  = X[train_split:test_split]
X_val   = X[test_split:]

Y_train = Y[:train_split]
Y_test  = Y[train_split:test_split]
Y_val   = Y[test_split:]

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# adding an extra dimension on the matrices for pytorch LSTMS
X_train = X_train.reshape((-1, lookback, 1))
X_test  = X_test.reshape((-1, lookback, 1))
X_val   = X_val.reshape((-1, lookback, 1))

Y_train = Y_train.reshape((-1, 1))
Y_test  = Y_test.reshape((-1, 1))
Y_val   = Y_val.reshape((-1, 1))

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# making this data into tensors for pytorch
X_train = torch.tensor(X_train).float()
Y_train = torch.tensor(Y_train).float()
X_test  = torch.tensor(X_test).float()
Y_test  = torch.tensor(Y_test).float()
X_val   = torch.tensor(X_val).float()
Y_val   = torch.tensor(Y_val).float()

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    # pytorch needs the len and getitem magic methods to train the model
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
train_dataset = TimeSeriesDataset(X_train, Y_train)
test_dataset = TimeSeriesDataset(X_test, Y_test)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# inspecting the batch size and dimensions
for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_stacked_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        # maps the LSTM output to a single predicted value
        self.fc = nn.Linear(hidden_size, 1) # Input sequence  →  LSTM  →  hidden state (hidden_size)  →  fc  →  prediction (1 value)

    # TODO: Create a stateful training
    def forward(self, x):
        batch_size = x.size(0)
        # initial short-term memory (hidden state)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        # initial long-term memory (cell state)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        (out, (hn, cn)) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
model = LSTM(1, 64 , 2)
model.to(device)
print(model)

def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    # it is a tensor with value
    running_loss = 0.0
    total_loss   = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        total_loss   += loss.item()
        # returns the gradients to 0
        optimizer.zero_grad()
        # do a backward probagation throug the loss to calculate the gradient
        loss.backward()
        # take a step into the direction of the gradient
        optimizer.step()

        if batch_index % 100 == 99: # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print(f'Batch {batch_index+1}, Average Loss: {avg_loss_across_batches:.3f}')
            running_loss = 0.0
    print()
    return total_loss / len(train_loader)

def validate_one_epoch():
    # put the model in evaluation mode
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print(f'Val Loss: {avg_loss_across_batches:.3f}')
    print("*******************************************")
    print()
    return avg_loss_across_batches

learning_rate = 0.001
num_epochs = 20
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses   = []

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss   = validate_one_epoch()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# ─── Metrics ────────────────────────────────────────────────────────────────
# The scaler was fit on the full (n, lookback+1) matrix, so to invert just the
# first column (the target) we pad dummy zeros for the remaining columns.
num_features = shifted_df_as_np.shape[1]

def inverse_transform_col0(arr: np.ndarray) -> np.ndarray:
    """Inverse-transform a 1-D array that lived in column 0 of the scaled matrix."""
    dummy = np.zeros((len(arr), num_features))
    dummy[:, 0] = arr.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

val_dates = data['Date'].iloc[lookback + test_split:].values

# plot the actual value vs the predicted values
with torch.no_grad():
    predicted = model(X_val.to(device)).to('cpu').numpy()

# Back to log(Close), then exponentiate to get actual USD price
Y_val_price  = np.exp(inverse_transform_col0(Y_val.numpy()))
pred_price   = np.exp(inverse_transform_col0(predicted))

# ── Four regression metrics ──────────────────────────────────────────────────
#
#  MSE  (Mean Squared Error)
#    Average of squared differences between prediction and truth.
#    Heavily penalises large errors; same unit as Close².
#
#  RMSE (Root Mean Squared Error)
#    Square root of MSE → same unit as Close (USD).
#    Easier to interpret: "on average, predictions are off by $X".
#
#  MAE  (Mean Absolute Error)
#    Average absolute difference. Less sensitive to outliers than RMSE.
#    Also in USD — directly readable as average dollar error.
#
#  MAPE (Mean Absolute Percentage Error)
#    MAE expressed as a percentage of the actual value.
#    Scale-independent; useful for comparing across different stocks/assets.
#
mse  = float(np.mean((Y_val_price - pred_price) ** 2))
rmse = float(np.sqrt(mse))
mae  = float(np.mean(np.abs(Y_val_price - pred_price)))
mape = float(np.mean(np.abs((Y_val_price - pred_price) / Y_val_price)) * 100)

print('\nValidation Metrics (original USD price scale):')
print(f'  MSE  = {mse:.4f}')
print(f'  RMSE = {rmse:.4f}')
print(f'  MAE  = {mae:.4f}')
print(f'  MAPE = {mape:.4f} %')

fig = plt.figure(figsize=(18, 9))
fig.suptitle('LSTM Amazon Stock Price - Validation Metrics', fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38)

# ── Top-left (spans 3 cols): time-series prediction ─────────────────────────
ax_ts = fig.add_subplot(gs[0, :3])
ax_ts.plot(val_dates, Y_val_price, label='Actual Close',    color='steelblue',  linewidth=1.5)
ax_ts.plot(val_dates, pred_price,  label='Predicted Close', color='tomato',     linewidth=1.5, linestyle='--')
ax_ts.set_title('Actual vs Predicted (validation set)')
ax_ts.set_xlabel('Date')
ax_ts.set_ylabel('Close Price (USD)')
ax_ts.legend()
ax_ts.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ── Top-right: training vs validation loss per epoch ────────────────────────
ax_loss = fig.add_subplot(gs[0, 3])
epochs_range = range(1, num_epochs + 1)
ax_loss.plot(epochs_range, train_losses, label='Train', color='steelblue', linewidth=1.5)
ax_loss.plot(epochs_range, val_losses,   label='Val',   color='tomato',    linewidth=1.5, linestyle='--')
ax_loss.set_title('Loss per Epoch')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('MSE Loss')
ax_loss.legend()

# ── Bottom row: one panel per metric ────────────────────────────────────────
metric_info = [
    # (label,  value,   bar colour,  subtitle)
    ('MSE',  mse,  '#e74c3c', 'Mean Squared Error\nAvg of squared errors  (USD²)\nPenalises large errors heavily'),
    ('RMSE', rmse, '#e67e22', 'Root Mean Squared Error\nSame unit as target  (USD)\n"Average" dollar error'),
    ('MAE',  mae,  '#3498db', 'Mean Absolute Error\nAvg absolute deviation  (USD)\nRobust to outliers'),
    ('MAPE', mape, '#27ae60', 'Mean Abs Percentage Error\nRelative error  (%)\nScale-independent'),
]

for col, (label, value, color, subtitle) in enumerate(metric_info):
    ax = fig.add_subplot(gs[1, col])
    bar = ax.bar([label], [value], color=color, width=0.45, zorder=3)
    ax.set_title(subtitle, fontsize=7.5, pad=6)
    ax.set_ylim(0, value * 1.35)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
    unit = ' %' if label == 'MAPE' else ''
    ax.text(
        0, value + value * 0.04,
        f'{value:.3f}{unit}',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold', color=color,
    )
    ax.set_xticks([])

plt.show()