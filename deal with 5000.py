import pandas as pd

# Load Excel file
file_path = "spectial_5000.xlsx"  # Adjust path if needed
xls = pd.ExcelFile(file_path)

print("Sheet names found:", xls.sheet_names)

# Load metadata
metadata_df = pd.read_excel(xls, sheet_name=1)
metadata_df.columns = metadata_df.columns.str.strip()

# Explicitly define column names
index_col = "ID-data sheet"                # This is column B
label_col = "Polymer commercial labeling"  # This is column G

# Define target material labels
target_labels = ["PP", "PS", "PET crystalline (PETc)", "PET amorphus (PETa)", "PET", "PE", "PVC"]

# Filter metadata
filtered_metadata = metadata_df[metadata_df[label_col].isin(target_labels)]

# Show metadata match
print(f"\nFiltered metadata for target labels (n={len(filtered_metadata)}):")
print(filtered_metadata[[index_col, label_col]].head())


# Load spectral data properly (data is column-wise per sample)
spectra_raw = pd.read_excel(xls, sheet_name=2, header=None)

# Drop the first two columns (NaNs or wavelength)
spectra_only = spectra_raw.iloc[:, 2:]

# Use first column (row 0) as the new column headers = data indices
spectra_only.columns = spectra_only.iloc[0].astype(str)
spectra_only = spectra_only[1:]  # Remove the header row

# Now spectra_only columns are data indices as strings
spectra_only.columns = spectra_only.columns.astype(str)

# Show a preview
print("\nCleaned spectral data (first 5 rows):")
print(spectra_only.iloc[:5, :5])

# Match with metadata
# Convert metadata index values to **string without decimal points**
data_indices = filtered_metadata[index_col].astype(int).astype(str).values

# Convert spectral data columns to **clean strings without decimals**
spectra_only.columns = spectra_only.columns.astype(float).astype(int).astype(str)

# Find matches again
matched_indices = list(set(data_indices) & set(spectra_only.columns))

print(f"\n Fixed Matched Data Indices (n={len(matched_indices)}): {matched_indices[:10]}")


# Filter spectral data and add material labels
selected_spectra = spectra_only.loc[:, matched_indices]

selected_spectra_T = selected_spectra.T
selected_spectra_T['Material'] = selected_spectra_T.index.map(
    dict(zip(filtered_metadata[index_col].astype(str), filtered_metadata[label_col]))
)

print("\nTransposed data with labels (first 5 rows):")
print(selected_spectra_T.head())

output_path = "filtered_spectral_data.csv"
wavelengths = spectra_raw.iloc[:, 1].values[1:]  # Skip header row
wavelengths = wavelengths.astype(int)  # Ensure clean column names
column_names = ["Sample Index"] + list(wavelengths) + ["Material"]

selected_spectra_T.insert(0, "Sample Index", selected_spectra_T.index)
selected_spectra_T.columns = column_names

selected_spectra_T.to_csv("filtered_spectral_data.csv", index=False)
print(f"\n Saved filtered spectral data to: {output_path}")