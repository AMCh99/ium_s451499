import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse
import torch.optim as optim

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--epochs', type=int, required=True)
args = parser.parse_args()

learning_rate = args.learning_rate
epochs = args.epochs

# Load data
data = pd.read_csv('data/btc_train.csv')
data = pd.DataFrame(data)

# Data preprocessing
le = LabelEncoder()
data['date'] = le.fit_transform(data['date'])
data['hour'] = le.fit_transform(data['hour'])
data['Volume BTC'] = data['Volume BTC'] / 10

# Convert strings to numbers
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Replace missing values with 0
data = data.fillna(0)

# Prepare tensors
inputs = torch.tensor(data.drop('Volume USD', axis=1).values, dtype=torch.float32)
targets = torch.tensor(data['Volume USD'].values, dtype=torch.float32).view(-1, 1)

# Create DataLoader
data_set = TensorDataset(inputs, targets)
data_loader = DataLoader(data_set, batch_size=64)

# Define model
model = nn.Sequential(
    nn.Linear(inputs.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
for epoch in range(epochs):
    for X, y in data_loader:
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Model has been trained.")

# Save model
torch.save(model.state_dict(), "model/model.pth")
print("Model saved to 'model/model.pth'.")
