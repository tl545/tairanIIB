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
input_size = 3  # 4 spectral filters (RGB + Mono)
latent_dim = 10  # Latent space dimension
num_epochs = 450  # Increased training epochs
batch_size = 32
learning_rate = 0.001

# Mode selection: 'plastics_only' (exclude label 0) or 'all_materials' (include label 0)
AE_mode = "all_materials"  # Train with all materials

# Load training and testing data
training_data_path = '/home/tl545/\u684c\u9762/iibtl545/datasets/training+testing/photocurrent_results_random_all_training.xlsx'
testing_data_path = '/home/tl545/\u684c\u9762/iibtl545/datasets/training+testing/photocurrent_results_random_all_testing.xlsx'

training_data = pd.read_excel(training_data_path)
testing_data = pd.read_excel(testing_data_path)

print("Training and testing data loaded")

start_time = time.time()
print("Start analyzing ...")

# Extract features
X_train = training_data[['red', 'blue', 'green']].values
X_test = testing_data[['red', 'blue', 'green']].values
y_train = training_data['Material_Label'].values
y_test = testing_data['Material_Label'].values

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

# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim=4, latent_dim=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Instantiate model, loss function, and optimizer
model = Autoencoder(input_size, latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        batch_X = batch[0].to(device)
        optimizer.zero_grad()
        encoded, decoded = model(batch_X)
        loss = criterion(decoded, batch_X)  # Reconstruction loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.6f}")

print("Training finished.")

# Extract latent space representations
model.eval()
with torch.no_grad():
    X_train_latent = model.encoder(X_train_tensor).cpu()
    X_test_latent = model.encoder(X_test_tensor).cpu()

# Define MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

# Train MLP Classifier
mlp = MLPClassifier(latent_dim, len(set(y_train))).to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(mlp.parameters(), lr=0.001)

train_dataset_mlp = TensorDataset(X_train_latent.to(device), y_train_tensor.to(device))
train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=batch_size, shuffle=True)

mlp.train()
for epoch in range(350):  # Train for 300 epochs
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
        print(f"MLP Epoch [{epoch+1}/350], Loss: {epoch_loss / len(train_loader_mlp):.6f}")

print("MLP training finished.")

# Evaluate MLP Classifier
mlp.eval()
with torch.no_grad():
    test_outputs = mlp(X_test_latent.to(device))
    predicted_labels = torch.argmax(test_outputs, dim=1).cpu().numpy()

testing_data['Predicted_Label'] = predicted_labels

accuracy = accuracy_score(y_test, predicted_labels)
print(f"MLP Classifier Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))


# Plot Latent Space with True Labels
plt.figure(figsize=(8, 6))
plt.scatter(X_test_latent[:, 0], X_test_latent[:, 1], c=y_test, cmap='viridis', alpha=0.7)
plt.colorbar(label="True Material Labels")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Latent Space Representation of Test Data")
plot_path = f'/home/tl545/\u684c\u9762/iibtl545/datasets/results/AE_results/latent_space_true_labels_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
plt.savefig(plot_path)
print(f"Latent space plot saved at {plot_path}")


# Save results
output_file_path = f'/home/tl545/\u684c\u9762/iibtl545/datasets/results/AE_results/AE_MLP_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}_.xlsx'
testing_data.to_excel(output_file_path, index=False)
print(f"Classification results saved to {output_file_path}")

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")



