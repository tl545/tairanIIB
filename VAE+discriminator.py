import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import time
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
input_size = 8
latent_dim = 15
num_epochs = 1100
batch_size = 32
learning_rate = 0.001
mlp_steps = 2500
loss_weight = 0.5
kl_weight = 0.00001
domain_weight = 0.1  # weight for domain-adversarial loss

# Load data
train_df = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/train_8filter_500.csv')
test_df = pd.read_csv('/home/tl545/桌面/iibtl545/datasets/training+testing/test_8filter_500.csv')

# Add domain labels manually (1: distorted / 0: clean)
train_df["domain"] = train_df["Sample"].apply(lambda x: 1 if "scale" in str(x) or "shift" in str(x) else 0)
test_df["domain"] = test_df["Sample"].apply(lambda x: 1 if "scale" in str(x) or "shift" in str(x) else 0)

# Features and labels
X_train = train_df[['red','green','blue','black','N1','N2','N3','N4']].values
y_train = train_df['Material'].values
d_train = train_df['domain'].values

X_test = test_df[['red','green','blue','black','N1','N2','N3','N4']].values
y_test = test_df['Material'].values
d_test = test_df['domain'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
d_train_tensor = torch.tensor(d_train, dtype=torch.long).to(device)
d_test_tensor = torch.tensor(d_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Gradient Reversal Layer
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_):
    return GradientReversal.apply(x, lambda_)

# VAE with Encoder
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4_mean = nn.Linear(128, latent_dim)
        self.fc4_logvar = nn.Linear(128, latent_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return self.fc4_mean(h), self.fc4_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

# Domain Discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

# MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Instantiate models
vae = VAE(input_size, latent_dim).to(device)
discriminator = DomainDiscriminator(latent_dim).to(device)
mlp = MLPClassifier(latent_dim, len(set(y_train))).to(device)

optimizer_vae = optim.Adam(vae.parameters(), lr=learning_rate)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_cls = optim.Adam(mlp.parameters(), lr=learning_rate)

criterion_cls = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

# VAE pretraining (optional, can be skipped if desired)
vae.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        x = batch[0]
        optimizer_vae.zero_grad()
        z, mu, logvar = vae(x)
        recon = z @ z.T  # dummy recon to reuse VAE structure
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = kl_weight * kl
        loss.backward()
        optimizer_vae.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 100 == 0:
        print(f"VAE Epoch [{epoch+1}/{num_epochs}], KL Loss: {epoch_loss:.6f}")

# Encode all data
vae.eval()
with torch.no_grad():
    mu_train, _ = vae.encode(X_train_tensor)
    mu_test, _ = vae.encode(X_test_tensor)

# Train MLP + Domain Discriminator
mlp.train()
discriminator.train()

for epoch in range(mlp_steps):
    optimizer_cls.zero_grad()
    optimizer_disc.zero_grad()

    # MLP classification loss
    out_cls = mlp(mu_train)
    loss_cls = criterion_cls(out_cls, y_train_tensor)

    # Domain loss with gradient reversal
    z_rev = grad_reverse(mu_train, 1.0)
    out_domain = discriminator(z_rev)
    loss_domain = criterion_domain(out_domain, d_train_tensor)

    total_loss = loss_cls + domain_weight * loss_domain
    total_loss.backward()
    optimizer_cls.step()
    optimizer_disc.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{mlp_steps}], Class Loss: {loss_cls.item():.4f}, Domain Loss: {loss_domain.item():.4f}")

# Evaluate
mlp.eval()
with torch.no_grad():
    preds = mlp(mu_test.to(device))
    predicted = torch.argmax(preds, dim=1).cpu().numpy()

acc = accuracy_score(y_test, predicted)
print(f"\nFinal Test Accuracy: {acc:.4f}")
print(classification_report(y_test, predicted))
