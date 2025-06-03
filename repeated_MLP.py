import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from scipy.stats import mode  
import time

code_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Model Parameter setting
input_size = 4
hidden_sizes = [128, 64, 32, 16]
num_epochs = 350
batch_size = 32
num_runs = 5    

# Choose mode: "all_materials" (0–6) or "filtered_materials" (1–6)
MLP_mode = "all_materials"  # Change to "all_materials" to include material 0

# Load the training and testing data
training_data_path = '/home/tl545/桌面/iibtl545/datasets/training+testing/photocurrent_results_random_all_training.xlsx'
testing_data_path = '/home/tl545/桌面/iibtl545/datasets/training+testing/photocurrent_results_random_all_testing.xlsx'

training_data = pd.read_excel(training_data_path)
testing_data = pd.read_excel(testing_data_path)

print("Training and testing data loaded")

start_time = time.time()
print("Start analysing ... ")


# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  
                layers.append(nn.ReLU())  # Add activation for hidden layers

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Filter data based on the selected mode
if MLP_mode == "filtered_materials":
    training_data = training_data[training_data['Material_Label'] != 0]  # Exclude material 0

# Extract features and labels
X_train = training_data[['red', 'blue', 'green', 'black']].values
y_train = training_data['Material_Label'].values

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train - (1 if MLP_mode == "filtered_materials" else 0), dtype=torch.long)

output_size = len(set(y_train))

# Inference with 5 runs and mode calculation
model_predictions = []  # To store predictions for each run

for run in range(num_runs):
    # Instantiate and train the model for each run
    model = MLP(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Testing the model for the current run
    if MLP_mode == "filtered_materials":
        testing_data = testing_data[testing_data['Material_Label'] != 0]  # Exclude material 0

    X_test = testing_data[['red','blue', 'green', 'black']].values
    X_test = scaler.transform(X_test)  # Use the same scaler as for training
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predicted_labels = torch.argmax(test_outputs, dim=1).numpy()

    if MLP_mode == "filtered_materials":
        predicted_labels += 1

    model_predictions.append(predicted_labels)

# Convert model_predictions into a 2D array (rows = runs, columns = predictions per run)
model_predictions_array = np.array(model_predictions)

# Transpose the array so that each column corresponds to the predictions for one sample across all runs
transposed_predictions = model_predictions_array.T

# Calculate the mode for each sample
final_predictions = mode(transposed_predictions, axis=1).mode.flatten()

# Add predictions to the testing data
testing_data['Predicted_Label'] = final_predictions

end_time = time.time()
print("Finished.", f"Time used: {end_time - start_time:.2f} sec")

# Evaluation
y_test = testing_data['Material_Label'].values
accuracy = accuracy_score(y_test, final_predictions)
print()
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, final_predictions))

# Save the predictions
output_file_path = f'/home/tl545/桌面/iibtl545/datasets/results/predictions_{mode}_{code_time}_accuracy{accuracy:.4f}.xlsx'
testing_data.to_excel(output_file_path, index=False)
print(f"Predictions saved to {output_file_path}")
