import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_path = '/Users/lit./Desktop/iibproject/filtered_lowpass_preserve_label.csv'
df = pd.read_csv(file_path)

# Get material labels and define selected wavelengths (every 50 nm from 400 to 1000)
materials = df['Material'].unique()
selected_wavelengths = list(range(450, 1000, 50))
selected_columns = [str(wl) for wl in selected_wavelengths]

# Randomly sample one representative sample per material
np.random.seed(42)  # For reproducibility
sampled_spectra = {}
for material in materials:
    subset = df[df['Material'] == material]
    random_sample = subset.sample(n=1)
    sampled_spectra[material] = random_sample.iloc[0, 1:-1]

# Create data matrix for selected wavelengths only
spectra_matrix = np.array([sampled_spectra[m][selected_columns].values.astype(float) for m in sorted(materials)])

# Compute maximum reflectance difference across materials at each selected wavelength
diff_across_materials = np.max(spectra_matrix, axis=0) - np.min(spectra_matrix, axis=0)

# Select top 10 wavelengths with highest material differences
top_indices = np.argsort(diff_across_materials)[-20:]
best_wavelengths = [selected_wavelengths[i] for i in sorted(top_indices)]

# Get top 10 indices sorted by descending difference
top_indices_sorted = np.argsort(diff_across_materials)[-20:][::-1]
best_wavelengths_ranked = [selected_wavelengths[i] for i in top_indices_sorted]
diff_values_ranked = [diff_across_materials[i] for i in top_indices_sorted]

# Combine wavelengths and their corresponding differences into a DataFrame
result_df = pd.DataFrame({
    "Wavelength (nm)": best_wavelengths_ranked,
    "Reflectance Difference": diff_values_ranked
})

# Plot reflectance of each material at selected wavelengths
plt.figure(figsize=(10, 6))
for i, material in enumerate(sorted(materials)):
    plt.plot(selected_wavelengths, spectra_matrix[i], marker='o', label=f'Material {material}')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Reflectance")
plt.title("Sampled Spectra per Material (Selected Wavelengths)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(result_df)
