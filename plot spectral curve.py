import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load Excel file
file_path = '/Users/lit./Desktop/iibproject/filtered_lowpass_preserve_label.csv'
df = pd.read_csv(file_path)

# Step 2: Select the 5135th row (remember: Python is 0-indexed, so index = 5134)
row_index = 60-2
row_data = df.iloc[row_index]

# Step 3: Drop the first and last elements
row_values = row_data.values[1:-1]  # Exclude first and last
print(row_values)

# Step 4: Plot
x_values = range(400, 400 + len(row_values))
plt.figure(figsize=(10, 4))
plt.plot(x_values, row_values, marker='o')
plt.title(f"Spectral data for sample {row_index + 1} (excluding first and last elements)")
plt.xlabel("wavelength")
plt.ylabel("value")
plt.grid(True)
plt.tight_layout()
plt.show()
