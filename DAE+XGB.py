import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# 检查是否可用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 参数设置
input_size = 8
latent_dim = 15
num_epochs = 1500
batch_size = 32
learning_rate = 0.001

# 加载数据
train_path = '/home/tl545/桌面/iibtl545/datasets/training+testing/train_8filter_500.csv'
test_path = '/home/tl545/桌面/iibtl545/datasets/training+testing/test_8filter_500_gaussian.csv'
training_data = pd.read_csv(train_path)
testing_data = pd.read_csv(test_path)

# 特征与标签
features = ['red', 'green', 'blue', 'black', 'N1', 'N2', 'N3', 'N4']
X_train = training_data[features].values
X_test = testing_data[features].values
y_train = training_data['Material'].values
y_test = testing_data['Material'].values

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义 Denoising Autoencoder（无KL项）
class DAE(nn.Module):
    def __init__(self, input_dim=8, latent_dim=15):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# 初始化 DAE
dae = DAE(input_size, latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(dae.parameters(), lr=learning_rate)

# 训练 DAE
print("Training Denoising Autoencoder...")
start_time = time.time()
dae.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        batch_X = batch[0].to(device)
        noisy_input = batch_X + 0.01 * torch.randn_like(batch_X)
        optimizer.zero_grad()
        recon, _ = dae(noisy_input)
        loss = criterion(recon, batch_X)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.6f}")

# 提取编码后的特征
dae.eval()
with torch.no_grad():
    _, X_train_latent = dae(X_train_tensor)
    _, X_test_latent = dae(X_test_tensor)

X_train_latent_np = X_train_latent.cpu().numpy()
X_test_latent_np = X_test_latent.cpu().numpy()

# 使用 XGBoost 分类器
print("Training XGBoost classifier...")
xgb_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_latent_np, y_train)

# 评估分类器
y_pred = xgb_model.predict(X_test_latent_np)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nXGBoost Classifier Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")
