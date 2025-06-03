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
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
input_size = 4
batch_size = 32
learning_rate = 0.001
cnn_epochs = 1200
mlp_epochs = 1800

num_class = 10
latent_d = 12

# Load data
training_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/train_clean.csv')
testing_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/test_scale.csv')
#training_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/train_new.csv')
#testing_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/test_new.csv')
#training_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/train_4filter_600_clean.csv')
#testing_data = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/test_4filter_600_clean.csv')

start_time = time.time()

# Extract features and labels
X_train = training_data[['red', 'green', 'blue', 'black']].values
X_test = testing_data[['red', 'green', 'blue', 'black']].values
y_train = training_data['Material'].values
y_test = testing_data['Material'].values

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors and reshape for CNN: (batch, channel=1, length=4)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2),   # (32, 3)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),                   # Dropout layer

            nn.Conv1d(32, 64, kernel_size=2),  # (64, 2)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=1),  # (128, 2)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),                    # Prevent overfitting

            nn.Flatten(),                       # (128*2 = 256)
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, latent_d),                 # Output latent dim
            nn.ReLU()
        )

    def forward(self, x):
        return self.cnn(x)

# Initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Initialize CNN
cnn_model = CNNFeatureExtractor().to(device)
cnn_model.apply(init_weights)

# CNN optimizer
optimizer_cnn = optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9) #87.94%
#optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=learning_rate)              #87.15%
#optimizer_cnn = optim.RMSprop(cnn_model.parameters(), lr=learning_rate)            #80.24%

criterion_cnn = nn.CrossEntropyLoss()

# Train CNN + final linear classifier
print("Training CNN feature extractor...")
for epoch in range(cnn_epochs):
    cnn_model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer_cnn.zero_grad()
        features = cnn_model(batch_X)
        loss = criterion_cnn(features, batch_y)  # Use features directly as logits
        loss.backward()
        optimizer_cnn.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{cnn_epochs}], Loss: {epoch_loss / len(train_loader):.6f}")

# Freeze CNN and extract latent features
cnn_model.eval()
with torch.no_grad():
    X_train_latent = cnn_model(X_train_tensor)
    X_test_latent = cnn_model(X_test_tensor)

# MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            #nn.Dropout(0.3),         # dropout layer
            nn.Linear(128, 64),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

mlp = MLPClassifier(latent_d, num_class).to(device)

mlp.apply(init_weights)

# MLP optimizer
optimizer_mlp = optim.Adam(mlp.parameters(), lr=learning_rate)
#optimizer_mlp = optim.SGD(mlp.parameters(), lr=learning_rate)

criterion_mlp = nn.CrossEntropyLoss()

train_dataset_mlp = TensorDataset(X_train_latent, y_train_tensor)
train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=batch_size, shuffle=True)

print("Training MLP classifier...")
for epoch in range(mlp_epochs):
    mlp.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader_mlp:
        optimizer_mlp.zero_grad()
        outputs = mlp(batch_X)
        loss = criterion_mlp(outputs, batch_y)
        loss.backward()
        optimizer_mlp.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 50 == 0:
        print(f"MLP Epoch [{epoch+1}/{mlp_epochs}], Loss: {epoch_loss / len(train_loader_mlp):.6f}")

# Evaluation
mlp.eval()
with torch.no_grad():
    test_outputs = mlp(X_test_latent)
    predicted_labels = torch.argmax(test_outputs, dim=1).cpu().numpy()

print("start plotting")
X_latent_np = X_test_latent.cpu().numpy()
y_pred_np = predicted_labels

# Randomly pick two component indices
latent_dim = X_latent_np.shape[1]
random_indices = random.sample(range(latent_dim), 2)
i1, i2 = random_indices

print(f"Using latent components {i1} and {i2} for visualization")

# Color by predicted label
plt.figure(figsize=(10, 7))
for label in np.unique(y_pred_np):
    idx = y_pred_np == label
    plt.scatter(X_latent_np[idx, i1], X_latent_np[idx, i2], label=f"Predicted {label}", alpha=0.7)

plt.title("Latent Space Visualization")
plt.xlabel("Component {i1}")
plt.ylabel("Component {i2}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#plt.savefig("latent_space_visualization_2.png", dpi=300)
#print("Plot saved")


accuracy = accuracy_score(y_test, predicted_labels)
print(f"CNN + MLP Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))

# Plot (2D latent space, use only if latent dim = 2 or apply PCA)
# Save predictions
testing_data['Predicted_Label'] = predicted_labels
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
testing_data.to_excel(f"cnn_mlp_results_{timestamp}.xlsx", index=False)

output_file_path = f'/home/tl545/桌面/iibtl545/datasets/results/CNN_report/CNN_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
testing_data['Predicted_Label'] = predicted_labels
testing_data.to_excel(output_file_path, index=False)
print(f"Classification results saved to {output_file_path}")

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")