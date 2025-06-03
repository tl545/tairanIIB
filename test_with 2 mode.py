import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

time = datetime.now().strftime("%Y%m%d_%H%M%S")



# Model Parameter setting
input_size = 2  
hidden_sizes = [128, 64, 32, 16]  
num_epochs = 350
batch_size = 32

# Choose mode: "all_materials" (0–6) or "filtered_materials" (1–6)
mode = "all_materials"  # Change to "all_materials" to include material 0

# Load the training and testing data
training_data_path = '/home/tl545/桌面/iibtl545/datasets/training+testing/photocurrent_results_random_all_training.xlsx'
testing_data_path = '/home/tl545/桌面/iibtl545/datasets/training+testing/photocurrent_results_random_all_testing.xlsx'

# The output path is set at the end of the code (line 130)



training_data = pd.read_excel(training_data_path)
testing_data = pd.read_excel(testing_data_path)

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  
                layers.append(nn.ReLU())   # Add activation for hidden layers

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Filter data based on the selected mode
if mode == "filtered_materials":
    training_data = training_data[training_data['Material_Label'] != 0]  # Exclude material 0

# Extract features and labels
X_train = training_data[['red', 'blue']].values
y_train = training_data['Material_Label'].values

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train - (1 if mode == "filtered_materials" else 0), dtype=torch.long)

output_size = len(set(y_train)) 

# Instantiate the model
model = MLP(input_size, hidden_sizes, output_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(num_epochs):
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Filter data based on the selected mode
if mode == "filtered_materials":
    testing_data = testing_data[testing_data['Material_Label'] != 0]  # Exclude material 0

# Extract features for testing
X_test = testing_data[['red', 'blue']].values
X_test = scaler.transform(X_test)  # Use the same scaler as for training
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Inference
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted_labels = torch.argmax(test_outputs, dim=1).numpy()

# Adjust predicted labels for filtered mode
if mode == "filtered_materials":
    predicted_labels += 1

# Add predictions to the testing data
testing_data['Predicted_Label'] = predicted_labels

# Evaluation
y_test = testing_data['Material_Label'].values
accuracy = accuracy_score(y_test, predicted_labels)
print()
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))

# Save the predictions
output_file_path = f'/home/tl545/桌面/iibtl545/datasets/results/predictions_{mode}_{time}_accuracy{accuracy:.4f}.xlsx'
testing_data.to_excel(output_file_path, index=False)
print(f"Predictions saved to {output_file_path}")