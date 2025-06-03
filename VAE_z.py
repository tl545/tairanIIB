import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model Parameters
input_size = 4  # 4 spectral filters (RGB + Mono)
latent_dim = 12  # Latent space dimension
num_epochs = 1000  # Increased training epochs
batch_size = 32
learning_rate = 0.001
mlp_steps = 2000
kl_weight = 0.000005

# Mode selection: 'plastics_only' (exclude label 0) or 'all_materials' (include label 0)
AE_mode = 'all_materials'  # Train with all materials

# Load training and testing data
training_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/train_clean.csv')
testing_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/test_noisy.csv')
#training_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/train_4filter_600_clean.csv')
#testing_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/test_4filter_600_clean.csv')

print("Training and testing data loaded")
start_time = time.time()
print("Start analyzing ...")

# Extract features
X_train = training_data[['red', 'green', 'blue', 'black']].values
X_test = testing_data[['red', 'green', 'blue', 'black']].values
y_train = training_data['Material'].values
y_test = testing_data['Material'].values

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define Variational Autoencoder (VAE) Model
class VAE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=10):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4_mean = nn.Linear(128, latent_dim)
        self.fc4_logvar = nn.Linear(128, latent_dim)
        self.fc5 = nn.Linear(latent_dim, 128)
        self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 512)
        self.fc8 = nn.Linear(512, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return self.fc4_mean(h), self.fc4_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc5(z))
        h = torch.relu(self.fc6(h))
        h = torch.relu(self.fc7(h))
        return self.fc8(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Instantiate model, loss function, and optimizer
model = VAE(input_size, latent_dim).to(device)
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_divergence

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        batch_X = batch[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch_X)
        loss = loss_function(recon_batch, batch_X, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.6f}")

print("Training finished.")

# Extract latent space representations (use z instead of mu)
model.eval()
with torch.no_grad():
    mu_train, logvar_train = model.encode(X_train_tensor)
    z_train = model.reparameterize(mu_train, logvar_train)
    X_train_latent = z_train.to(device)

    mu_test, logvar_test = model.encode(X_test_tensor)
    z_test = model.reparameterize(mu_test, logvar_test)
    X_test_latent = z_test.cpu()

# Define MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            #nn.Dropout(p=0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            #nn.Dropout(p=0.25),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# Define and train MLP Classifier (unchanged)
train_dataset_mlp = TensorDataset(X_train_latent, y_train_tensor)
train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=batch_size, shuffle=True)

mlp = MLPClassifier(latent_dim, len(set(y_train))).to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(mlp.parameters(), lr=0.001)

mlp.train()
for epoch in range(mlp_steps):
    epoch_loss = 0
    for batch_X, batch_y in train_loader_mlp:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer_cls.zero_grad()
        outputs = mlp(batch_X)
        loss = criterion_cls(outputs, batch_y)
        loss.backward()
        optimizer_cls.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"MLP Epoch [{epoch+1}/{mlp_steps}], Loss: {epoch_loss / len(train_loader_mlp):.6f}")

print("MLP training (with z input) finished.")

# Evaluate Classification with MLP
mlp.eval()
with torch.no_grad():
    test_outputs = mlp(X_test_latent.to(device))
    predicted_labels = torch.argmax(test_outputs, dim=1).cpu().numpy()

testing_data['Predicted_Label'] = predicted_labels

accuracy = accuracy_score(y_test, predicted_labels)
print(f"MLP Classifier Accuracy (using z): {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))

# Save results
output_file_path = f'/home/tl545/桌面/iibtl545/datasets/results/VAE_z_report/VAE_z_{datetime.now().strftime("%Y%m%d_%H%M%S")}_.xlsx'
testing_data.to_excel(output_file_path, index=False)
print(f"Classification results (using z) saved to {output_file_path}")

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")
