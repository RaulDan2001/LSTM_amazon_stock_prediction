import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# read the csv
# Read dataset from project-relative path to avoid hardcoded machine-specific paths.
data_path = Path(__file__).resolve().parents[1] / 'dataset' / 'Amazon_stock_data.csv'
data = pd.read_csv(data_path)

# use the gpu if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# handle the date column
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop=True)

from copy import deepcopy as dc

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

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
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

learning_rate = 0.001
num_epochs = 20
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

val_dates = data['Date'].iloc[lookback + test_split:].values

# plot the actual value vs the predicted values
with torch.no_grad():
    predicted = model(X_val.to(device)).to('cpu').numpy()

plt.figure(figsize=(12, 5))
plt.plot(val_dates, Y_val, label='Actual Close')
plt.plot(val_dates, predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.title('Validation Predictions')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.show()