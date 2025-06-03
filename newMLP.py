import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mode
import time
import torch.nn.functional as F
from datetime import datetime

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
input_size = 4
hidden_sizes = [128, 64, 32, 16]
num_epochs = 400
batch_size = 32
n_runs = 5  # Number of independent training + prediction runs
learning_rate = 0.001

# Data paths
train_data = pd.read_excel('/home/tl545/桌面/iibtl545/datasets/training_327_no9.xlsx')
test_data = pd.read_excel('/home/tl545/桌面/iibtl545/datasets/testing_327_no9.xlsx')
#train_path = '/home/tl545/桌面/iibtl545/datasets/training+testing/train_4filter_600_clean.csv'
#test_path  = '/home/tl545/桌面/iibtl545/datasets/training+testing/test_4filter_600_clean.csv'

# Load data
#train_data = pd.read_csv(train_path)
#test_data  = pd.read_csv(test_path)

X_train = train_data[['red', 'blue', 'green', 'black']].values
y_train = train_data['Material_Label'].values
X_test  = test_data[['red', 'blue', 'green', 'black']].values
y_test  = test_data['Material_Label'].values

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32).to(device)

# MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# === Multi-Run Ensemble Voting ===
all_preds = []

start_time = time.time()
for run in range(n_runs):
    print(f"\n=== Run {run + 1}/{n_runs} ===")
    model = MLP(input_size, hidden_sizes, output_size=len(np.unique(y_train))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} - Loss: {loss.item():.4f}")

    # Predict on test set
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
        all_preds.append(preds)

# === Voting: majority rule across runs ===
all_preds_array = np.array(all_preds)  # shape: [n_runs, n_samples]
final_preds = mode(all_preds_array, axis=0).mode.flatten()

# === Evaluation ===
accuracy = accuracy_score(y_test, final_preds)
print(f"\nFinal Test Accuracy (Multi-Run Voting): {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, final_preds))

# === Save to CSV ===
test_data['Predicted_Label'] = final_preds
output_path = f'/home/tl545/\u684c\u9762/iibtl545/datasets/results/MLPresults/MLP_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
test_data.to_csv(output_path, index=False)
print(f"\nPredictions saved")
print(f"Total time used: {time.time() - start_time:.2f} seconds")

