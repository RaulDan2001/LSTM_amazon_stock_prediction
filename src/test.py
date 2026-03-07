import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# read the csv with correct separator
# corect decimal, columns and missing values interpretation
data = pd.read_csv('dataset/AirQuality.csv', sep=';', decimal=',', usecols=range(15), na_values=[-200])

# use the gpu if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

data['Date'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
data = data.drop(columns=['Time'])

plt.plot(data['Date'], data['CO(GT)'])