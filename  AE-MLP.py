import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import time

# Parameters
input_size = 9
latent_dim = 12
batch_size = 32
num_epochs_ae = 1000
num_epochs_mlp = 2000
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load Data ===
training_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/train_clean.csv')
testing_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/test_clean.csv')

start_time = time.time()

X_train = training_data[['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'E12']].values
X_test = testing_data[['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'E12']].values
y_train = training_data['Material'].values
y_test = testing_data['Material'].values

# === Standardize ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# === AE model ===
class AE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=12):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

ae = AE(input_dim=input_size, latent_dim=latent_dim).to(device)
optimizer_ae = optim.Adam(ae.parameters(), lr=learning_rate)
criterion_ae = nn.MSELoss()

# === AE Training ===
ae.train()
dataset = TensorDataset(X_train_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs_ae):
    total_loss = 0
    for batch in loader:
        x_batch = batch[0]
        optimizer_ae.zero_grad()
        recon = ae(x_batch)
        loss = criterion_ae(recon, x_batch)
        loss.backward()
        optimizer_ae.step()
        total_loss += loss.item()
    if (epoch+1) % 100 == 0:
        print(f"AE Epoch [{epoch+1}/{num_epochs_ae}], Loss: {total_loss / len(loader):.6f}")

# === Extract latent ===
ae.eval()
with torch.no_grad():
    X_train_latent = ae.encoder(X_train_tensor)
    X_test_latent = ae.encoder(X_test_tensor)

# === MLP Classifier ===
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

mlp = MLPClassifier(input_dim=latent_dim, num_classes=len(np.unique(y_train))).to(device)
optimizer_mlp = optim.Adam(mlp.parameters(), lr=learning_rate)
criterion_mlp = nn.CrossEntropyLoss()

# === MLP Training ===
train_data_mlp = TensorDataset(X_train_latent, y_train_tensor)
train_loader_mlp = DataLoader(train_data_mlp, batch_size=batch_size, shuffle=True)

mlp.train()
for epoch in range(num_epochs_mlp):
    total_loss = 0
    for batch_X, batch_y in train_loader_mlp:
        optimizer_mlp.zero_grad()
        outputs = mlp(batch_X)
        loss = criterion_mlp(outputs, batch_y)
        loss.backward()
        optimizer_mlp.step()
        total_loss += loss.item()
    if (epoch+1) % 200 == 0:
        print(f"MLP Epoch [{epoch+1}/{num_epochs_mlp}], Loss: {total_loss / len(train_loader_mlp):.6f}")

# === Evaluation ===
mlp.eval()
with torch.no_grad():
    predictions = mlp(X_test_latent)
    y_pred = torch.argmax(predictions, dim=1).cpu().numpy()

accuracy = accuracy_score(y_test, y_pred)
print(f"AE-MLP Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

testing_data['Predicted_Label'] = y_pred  
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"/home/tl545/桌面/iibtl545/datasets/results/AE_MLP_report/AE_{timestamp}.csv"
testing_data.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")