import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Folder containing all SED files
folder_path = '/Users/lit./Desktop/iibproject/Dataset of spectral reflectances and hypercubes of submerged plastic litter, including COVID-19 medical waste, pristine plasti>_1_all/L0_SE_Raw_Cleaned_Datafiles'

# Responsivity data paths
responsivity_paths = {
    #'red': '/Users/lit./Desktop/iibproject/photocurrent/red/extracted red.csv',
    #'blue': '/Users/lit./Desktop/iibproject/photocurrent/blue/extracted blue.csv',
    #'green': '/Users/lit./Desktop/iibproject/photocurrent/green/extracted green.csv',
    #'black': '/Users/lit./Desktop/iibproject/photocurrent/black/extracted black.csv'
    'red': '/Users/lit./Desktop/iibproject/filter_TiO2_501nm_try.csv',
    'blue': '/Users/lit./Desktop/iibproject/filter_TiO2_540nm_try.csv',
    'green': '/Users/lit./Desktop/iibproject/filter_TiO2_582nm_try.csv'

}
modified_paths = {
    #'red': '/Users/lit./Desktop/iibproject/photocurrent/red/modified_responsivity_red.csv',
    #'blue': '/Users/lit./Desktop/iibproject/photocurrent/blue/modified_responsivity_blue.csv',   
    #'green': '/Users/lit./Desktop/iibproject/photocurrent/green/modified_responsivity_green.csv',
    #'black': '/Users/lit./Desktop/iibproject/photocurrent/black/modified_responsivity_black.csv'
    'red': '/Users/lit./Desktop/iibproject/filter_TiO2_501nm_try.csv',
    'blue': '/Users/lit./Desktop/iibproject/filter_TiO2_540nm_try.csv',
    'green': '/Users/lit./Desktop/iibproject/filter_TiO2_582nm_try.csv'
}

# Metadata path
metadata_path = '/Users/lit./Desktop/iibproject/Dataset of spectral reflectances and hypercubes of submerged plastic litter, including COVID-19 medical waste, pristine plasti>_1_all/L0_SE_Raw_Metadata.csv'

# Load metadata and map material labels
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
metadata['Material_Label'] = metadata['Material'].map(material_labels)

# Function to load or create modified responsivity files
def load_responsivity(filter_name, original_path, modified_path):
    if not os.path.exists(modified_path):
        print(f"Creating modified responsivity file for {filter_name}...")
        res_data = pd.read_csv(original_path)
        res_data.columns = ['wavelength', 'responsivity']

        # Interpolate and modify responsivity
        interpolated_responsivity = np.interp(
            np.arange(400, 1001),  # Full wavelength range (400 to 1000 Hz)
            res_data['wavelength'],
            res_data['responsivity']
        )
        responsivity_df = pd.DataFrame({
            'wavelength': np.arange(400, 1001),
            'responsivity': interpolated_responsivity / 100 * np.arange(400, 1001) / 1240
        })
        responsivity_df.to_csv(modified_path, index=False)
        #print(f"Modified responsivity file saved to {modified_path}")
    #else:
        #print(f"Using existing modified responsivity file for {filter_name}: {modified_path}")
    return pd.read_csv(modified_path)

def butter_lowpass_filter(data, cutoff=0.1, fs=1.0, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Function to calculate photocurrent for a single SED file
def calculate_photocurrent(file_path, responsivity_data):
    wavelengths = []
    reflectance = []

    # Read the SED file and extract data
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
                    wavelengths.append(float(parts[0]))
                    reflectance.append(1 - float(parts[1]))
            except ValueError:
                continue

    # Filter data for the responsivity range
    filtered_wavelengths = []
    filtered_reflectance = []
    for wl, ref in zip(wavelengths, reflectance):
        if 400 <= wl <= 1000:
            filtered_wavelengths.append(wl)
            filtered_reflectance.append(ref)

    filtered_reflectance = butter_lowpass_filter(filtered_reflectance, cutoff=0.1, fs=len(filtered_reflectance)/600)

    # Ensure the responsivity aligns with the filtered wavelengths
    responsivity = np.interp(filtered_wavelengths, responsivity_data['wavelength'], responsivity_data['responsivity'])
    
    # Calculate delta wavelength
    delta_wavelength = np.diff(filtered_wavelengths, append=filtered_wavelengths[-1])

    # Calculate the photocurrent
    photocurrent = np.sum(np.array(filtered_reflectance) * responsivity * delta_wavelength)
    return photocurrent

# Function to get material label for a file
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
            return matched_metadata['Material_Label'].values[0]
    return None

# Calculate photocurrents for all filters and add material labels
results = []
for filename in os.listdir(folder_path):
    if filename.endswith('.sed'):
        file_path = os.path.join(folder_path, filename)
        photocurrents = {}
        for filter_name, res_path in responsivity_paths.items():
            modified_path = modified_paths[filter_name]
            responsivity_data = load_responsivity(filter_name, res_path, modified_path)
            photocurrents[filter_name] = calculate_photocurrent(file_path, responsivity_data)
        material_label = get_material_label(filename)
        results.append({'SED_File': filename, **photocurrents, 'Material_Label': material_label})

# Save results to CSV
results_df = pd.DataFrame(results)

results_df = results_df.dropna(subset=['Material_Label'])

results_df.to_csv('/Users/lit./Desktop/iibproject/photocurrent_new_filters_noiseless.csv', index=False)
print("Photocurrent calculation completed. Results saved.")