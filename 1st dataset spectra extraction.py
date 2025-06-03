import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# folder path
folder_path = '/Users/lit./Desktop/iibproject/Dataset of spectral reflectances and hypercubes of submerged plastic litter, including COVID-19 medical waste, pristine plasti>_1_all/L0_SE_Raw_Cleaned_Datafiles'
metadata_path = '/Users/lit./Desktop/iibproject/Dataset of spectral reflectances and hypercubes of submerged plastic litter, including COVID-19 medical waste, pristine plasti>_1_all/L0_SE_Raw_Metadata.csv'


metadata = pd.read_csv(metadata_path)
material_labels = {
    'HDPE': 1,
    'PP': 2,
    'XPS': 3,
    'PET': 4,
    'PVC': 5,
    'PA6': 6,
    'Blank no Glass': 0,
    'Blank with Glass': 0,
    'Glass_Blank': 0,
    'BLANK': 0,
    'WATER': 0
}
metadata['Material'] = metadata['Material'].map(material_labels)


def get_material_label(filename):
    file_parts = filename.split('__')[0].split('_')
    file_day = str(int(file_parts[2]))
    file_month = file_parts[1]
    file_year = file_parts[0][2:]
    file_date = f"{file_day}-{file_month}-{file_year}"
    file_index = int(filename.split('__')[-1].split('.')[0])

    matching_date_metadata = metadata.loc[metadata['Date'] == file_date]
    if not matching_date_metadata.empty:
        matched_metadata = matching_date_metadata.loc[
            (matching_date_metadata['Start'] <= file_index) &
            (matching_date_metadata['End'] >= file_index)
        ]
        if not matched_metadata.empty:
            return matched_metadata['Material'].values[0]
    return None

# low pass filter
def butter_lowpass_filter(data, cutoff=0.1, fs=1.0, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

target_wavelengths = np.arange(400, 1001)


results = []
for filename in os.listdir(folder_path):
    if filename.endswith('.sed'):
        file_path = os.path.join(folder_path, filename)
        
        wavelengths = []
        reflectance = []
        
        with open(file_path, 'r') as file:
            data_section = False
            for line in file:
                if "Data:" in line:
                    data_section = True
                    continue
                if not data_section:
                    continue
                try:
                    parts = line.split()
                    if len(parts) == 2:
                        wl = float(parts[0])
                        refl = float(parts[1])
                        if 400 <= wl <= 1000:
                            wavelengths.append(wl)
                            reflectance.append(refl)
                except ValueError:
                    continue

        if not wavelengths:
            continue

        try:
            interpolated_refl = np.interp(target_wavelengths, wavelengths, reflectance)
            filtered_refl = butter_lowpass_filter(interpolated_refl, cutoff=0.1, fs=len(interpolated_refl)/600)
        except:
            continue

        label = get_material_label(filename)
        entry = {'Sample': filename}
        entry.update(dict(zip(target_wavelengths, filtered_refl)))
        entry['Material'] = label
        results.append(entry)


df = pd.DataFrame(results)

df = df.dropna(subset=['Material'])

df.to_csv('/Users/lit./Desktop/iibproject/interpolated_spectra_clean.csv', index=False)
print("File saved.")
