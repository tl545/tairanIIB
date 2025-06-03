import pandas as pd

# Load the existing file
file_path = "/Users/lit./Desktop/iibproject/filtered_spectral_data.csv"
df = pd.read_csv(file_path)

# Define mapping from string to numeric labels
material_labels = {
    'PP': 2,
    'PS': 10,
    'PET': 4,
    'PET crystalline (PETc)': 8,
    'PET amorphus (PETa)': 9,
    'PVC': 5,
    'PE': 7
}

# Apply the mapping
df["Material"] = df["Material"].map(material_labels)

# Optional: print a few lines to check
print(df[["Sample Index", "Material"]].head())

# Save back to CSV
df.to_csv("filtered_spectral_data_labeled_PET_PS).csv", index=False)
print("✅ Material column updated and saved to 'filtered_spectral_data_labeled_PET_PS.csv'")
