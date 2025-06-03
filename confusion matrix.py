import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import glob
import numpy as np

file_paths = [
    #'/Users/lit./Desktop/iibproject/AE_MLP_report/AE_20250601_111056_.csv',
    #'/Users/lit./Desktop/iibproject/AE_MLP_report/AE_20250601_114403_.csv',
    #'/Users/lit./Desktop/iibproject/AE_MLP_report/AE_20250601_115022_.csv',
    #'/Users/lit./Desktop/iibproject/AE_MLP_report/AE_20250601_120603_.csv',
    '/Users/lit./Desktop/predictions.csv'
]

combined_df = pd.concat([pd.read_csv(f) for f in file_paths], ignore_index=True)

y_true = combined_df['Material']
y_pred = combined_df['Predicted_Label']

labels = sorted(y_true.unique())   # optional: list(range(10)) for fixed classes

cm = confusion_matrix(y_true, y_pred, labels=labels)

# Normalize by column (predicted labels)
cm_percent = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :] * 100

# Handle divide-by-zero if any predicted class has no samples
cm_percent = np.nan_to_num(cm_percent)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format=".1f")
plt.title("Normalized Confusion Matrix (% per predicted class)")
plt.tight_layout()
plt.show()