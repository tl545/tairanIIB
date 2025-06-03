import pandas as pd

input_path = '/Users/lit./Desktop/iibproject/filtered_lowpass_preserve_label.csv'
output_path = '/Users/lit./Desktop/iibproject/second_spectra_cleaned.csv'

df = pd.read_csv(input_path)

df_cleaned = df[df['Material'] != 9]
df_cleaned['Material'] = df_cleaned['Material'].replace(10, 9)


df_cleaned.to_csv(output_path, index=False)
print("File saved.")