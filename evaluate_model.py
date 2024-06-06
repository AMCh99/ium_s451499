import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('btc_dev.csv')
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

# Load model
model = nn.Sequential(
    nn.Linear(inputs.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

# Evaluate model
predictions = model(inputs)
mse = nn.MSELoss()(predictions, targets).item()

print(f"Mean Squared Error: {mse}")

# Save evaluation results
evaluation_results = pd.DataFrame({
    'Actual': targets.view(-1).numpy(),
    'Predicted': predictions.view(-1).detach().numpy()
})
evaluation_results.to_csv('evaluation_results/predictions.csv', index=False)
print("Evaluation results saved to 'evaluation_results/predictions.csv'.")
