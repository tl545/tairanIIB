import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
file_path = '/home/tl545/桌面/iibtl545/datasets/training+testing/mixup_expanded_600.csv'  
df = pd.read_csv(file_path)

# Extract features and labels
wavelength_columns = [str(wl) for wl in range(400, 1001)]
X = df[wavelength_columns].values.astype(np.float32)
y = df["Material"].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define VAE model
class VAE(nn.Module):
    def __init__(self, input_dim=601, latent_dim=10):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Define loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Train VAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in range(50):
    vae.train()
    total_loss = 0
    for batch in dataloader:
        x_batch = batch[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(x_batch)
        loss = vae_loss(recon_batch, x_batch, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

# Per-class sampling to augment to 700
augmented_samples = []
unique_labels = np.unique(y)

for label in unique_labels:
    X_class = X_scaled[y == label]
    X_class_orig = X[y == label]  # for cosine similarity check in original scale
    num_existing = X_class.shape[0]
    num_needed = max(0, 700 - num_existing)
    if num_needed == 0:
        continue

    with torch.no_grad():
        inputs = torch.tensor(X_class, dtype=torch.float32).to(device)
        mu, logvar = vae.encode(inputs)
        mu = mu.cpu().numpy()
        std = torch.exp(0.5 * logvar).cpu().numpy()

    mean = np.mean(mu, axis=0)
    cov = np.cov(mu.T)

    seen = set()
    n_generated = 0
    max_attempts = 300

    while n_generated < num_needed and max_attempts > 0:
        z = np.random.multivariate_normal(mean, cov)
        z_tensor = torch.tensor(z, dtype=torch.float32).to(device)
        with torch.no_grad():
            generated = vae.decode(z_tensor.unsqueeze(0)).cpu().numpy().flatten()

        generated_orig = scaler.inverse_transform(generated.reshape(1, -1)).flatten()
        recon_str = tuple(np.round(generated_orig, 5))

        # Check uniqueness
        if recon_str in seen:
            max_attempts -= 1
            continue
        seen.add(recon_str)

        # Similarity check (cosine similarity ≥ 0.98)
        sim = cosine_similarity(generated_orig.reshape(1, -1), X_class_orig).max()
        if sim >= 0.98:
            sample_dict = {str(wl): generated_orig[i] for i, wl in enumerate(range(400, 1001))}
            sample_dict["Material"] = label
            sample_dict["Sample"] = f"aug_vae_{label}_{np.random.randint(1e6)}"
            augmented_samples.append(sample_dict)
            n_generated += 1

        max_attempts -= 1

aug_df = pd.DataFrame(augmented_samples)
output_path = "/home/tl545/桌面/iibtl545/datasets/training+testing/vae_generated_700.csv"
aug_df.to_csv(output_path, index=False)
print('Finished.')


