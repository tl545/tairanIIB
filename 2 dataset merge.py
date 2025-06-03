import pandas as pd

file1 = '/Users/lit./Desktop/iibproject/interpolated_spectra_clean.csv'
file2 = '/Users/lit./Desktop/iibproject/second_spectra_cleaned.csv'
output_path = '/Users/lit./Desktop/iibproject/combined_spectra_dataset.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

combined_df = pd.concat([df1, df2], ignore_index=True)
combined_df.to_csv(output_path, index=False)

print("File saved.")
