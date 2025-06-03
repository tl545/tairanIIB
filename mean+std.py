import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score

file_paths = [
    '/Users/lit./Desktop/iibproject/CNN_report/CNN_20250601_164609.xlsx',
    '/Users/lit./Desktop/iibproject/CNN_report/CNN_20250601_162930.xlsx',
    '/Users/lit./Desktop/iibproject/CNN_report/CNN_20250601_161746.xlsx',
    '/Users/lit./Desktop/iibproject/CNN_report/CNN_20250601_160816.xlsx',
    '/Users/lit./Desktop/iibproject/CNN_report/CNN_20250601_155919.xlsx'
]

accuracies = []
macro_recalls = []

for file in file_paths:
    df = pd.read_excel(file)

    # Ensure both columns exist
    if 'Material' not in df.columns or 'Predicted_Label' not in df.columns:
        raise ValueError(f"Missing columns in file: {file}")

    y_true = df['Material'].values
    y_pred = df['Predicted_Label'].values

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')

    accuracies.append(acc)
    macro_recalls.append(recall)

    print(f"{file}: Accuracy = {acc:.4f}, Macro Recall = {recall:.4f}")

# Final summary
print("\n=== Summary Over 5 Runs ===")
print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Average Macro Recall: {np.mean(macro_recalls):.4f} ± {np.std(macro_recalls):.4f}")
