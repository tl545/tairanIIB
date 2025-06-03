import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

file_path = '/Users/lit./Desktop/iibproject/filtered_spectral_data_labeled_PET_PS.csv'
df = pd.read_csv(file_path)


row_index = 4000 - 2
row_data = df.iloc[row_index]
row_values = row_data.values[51:-111]  # Exclude first and last
print("Original", len(row_values))

wavelengths = np.linspace(400, 1000, len(row_values))

def butter_filter(data, cutoff, fs, btype='low', order=5):
    nyq = 0.5 * fs                 
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = filtfilt(b, a, data)
    return y

fs = len(row_values) / (1000 - 400)  # 1/nm

df_filtered = df.copy()

for idx in df.index:
    spectral_data = df.iloc[idx, 51: -111].values.astype(float)
    filtered = butter_filter(spectral_data, 0.1, fs, btype='low', order=5)
    df_filtered.iloc[idx, 51: -111] = filtered        # Replace only spectral part

output_path = '/Users/lit./Desktop/iibproject/filtered_lowpass_preserve_label.csv'
df_filtered.to_csv(output_path, index=False)

#plt.figure(figsize=(12, 4))
#plt.plot(wavelengths, row_values, label='Original signal', alpha=0.5)
#plt.xlabel('Wavelength (nm)')
#plt.ylabel('Value')
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()
