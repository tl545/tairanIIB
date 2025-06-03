import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, recall_score
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy

# === Adjustable Parameters ===
train_path = '/home/tl545/桌面/iibtl545/datasets/training+testing/train_9filter_600_clean.csv' #ag
test_path  = '/home/tl545/桌面/iibtl545/datasets/training+testing/test_9filter_600_moisy.csv'  #ag

#train_path = '/home/tl545/桌面/iibtl545/datasets/training+testing/train_10filter_600_gold.csv' #au
#test_path  = '/home/tl545/桌面/iibtl545/datasets/training+testing/test_10filter_600_gold_noisy.csv'  #au

# === Parameters ===
input_dim = 10
dropout_rate = 0.1
num_epochs = 320
batch_size = 32
learning_rate = 0.001
noise_std = 0.005
mixup_alpha = 0
n_votes = 20
loss_weight = 1
patience = 350
L2_decay = 1e-5

# === Gaussian Noise Layer ===
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.stddev
        return x

# === Gradient Reversal Layer ===
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

# === CNN Encoder (1D Conv) ===
class CNN_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(CNN_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # (B, 1, 10) -> (B, 32, 10)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), # -> (B, 64, 10)
            nn.ReLU(),
            nn.Flatten(),                               # -> (B, 64*10)
            nn.Linear(64 * input_dim, latent_dim),      # -> (B, latent_dim)
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim: (B, 10) -> (B, 1, 10)
        z = self.encoder(x)
        return z

# === DANN Model using CNN Encoder ===
class DANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3, noise_std=0.0):
        super(DANN, self).__init__()
        self.noise = GaussianNoise(noise_std)
        self.encoder_module = CNN_Encoder(input_dim, latent_dim=hidden_dim)

        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, alpha):
        x = self.noise(x)
        features = self.encoder_module(x)
        reverse_features = GRL.apply(features, alpha)
        class_outputs = self.label_classifier(features)
        domain_outputs = self.domain_discriminator(reverse_features)
        return class_outputs, domain_outputs

# === MixUp Functions ===
def mixup_data(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# === Training Function ===
def train_dann(model, train_loader, val_loader, device, optimizer, criterion_cls, criterion_domain):
    best_model = None
    best_val_acc = 0
    stale_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        total_correct = 0
        total_samples = 0

        for step, (x_batch, y_batch, d_batch) in enumerate(train_loader):
            p = float(step + epoch * len(train_loader)) / (num_epochs * len(train_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            x_batch, y_batch, d_batch = x_batch.to(device), y_batch.to(device), d_batch.to(device)

            optimizer.zero_grad()
            x_mix, y_a, y_b, lam = mixup_data(x_batch, y_batch, mixup_alpha)
            class_outputs, domain_outputs = model(x_mix, alpha)

            loss_cls = mixup_criterion(criterion_cls, class_outputs, y_a, y_b, lam)
            loss_domain = criterion_domain(domain_outputs, d_batch)

            weights = 1 - d_batch + d_batch * loss_weight
            loss_cls = (loss_cls * weights).mean()

            total_loss = loss_cls + loss_domain
            total_loss.backward()
            optimizer.step()

            total_correct += (torch.argmax(class_outputs, dim=1) == y_batch).sum().item()
            total_samples += y_batch.size(0)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val, _ in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds, _ = model(x_val, alpha=0.0)
                pred_labels = torch.argmax(preds, dim=1)
                correct += (pred_labels == y_val).sum().item()
                total += y_val.size(0)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stale_epochs = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}, best val acc: {best_val_acc:.4f}")
                break

        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Acc: {total_correct / total_samples:.4f}  Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_model)
    return model

# === Evaluation with Voting ===
def evaluate(model, X_test, y_test, device, scaler):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        votes = []
        for _ in range(n_votes):
            noisy_input = X_test_tensor + torch.randn_like(X_test_tensor) * noise_std
            logits, _ = model(noisy_input, alpha=0.0)
            votes.append(F.softmax(logits, dim=1))

        avg_probs = torch.stack(votes).mean(dim=0)
        pred_labels = torch.argmax(avg_probs, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, pred_labels)
    macro_recall = recall_score(y_test, pred_labels, average='macro')

    print(f"\nFinal Test Accuracy (Voting): {accuracy:.4f}")
    print(f"Final Macro Recall (Voting): {macro_recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred_labels, target_names=[str(c) for c in le.classes_]))
    return accuracy, macro_recall

# === Load and Prepare Data ===
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

#X_train = train_df[['E12', 'R', 'G', 'B', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6']].values
X_train = train_df[["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "E12"]].values
y_train = train_df['Material'].values
is_aug_train = train_df['Sample'].str.startswith("aug_").astype(int).values

#X_test = test_df[['E12', 'R', 'G', 'B', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6']].values
X_test = test_df[["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "E12"]].values
y_test = test_df['Material'].values
is_aug_test = test_df['Sample'].str.startswith("aug_").astype(int).values

le = LabelEncoder()
y = le.fit_transform(np.concatenate([y_train, y_test]))
y_train = y[:len(y_train)]
y_test = y[len(y_train):]

domain_labels = np.concatenate([is_aug_train, is_aug_test])
X = np.vstack([X_train, X_test])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
domain_tensor = torch.tensor(domain_labels, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor, domain_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Initialize and Train Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DANN(input_dim=input_dim, hidden_dim=128,
             num_classes=len(le.classes_),
             dropout=dropout_rate,
             noise_std=noise_std).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_decay)
criterion_cls = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

model = train_dann(model, train_loader, val_loader, device, optimizer, criterion_cls, criterion_domain)
evaluate(model, X_test, y_test, device, scaler)
