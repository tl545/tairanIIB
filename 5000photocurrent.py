import pandas as pd
import numpy as np

spectral_csv = "/Users/lit./Desktop/iibproject/filtered_lowpass_preserve_label.csv"  # use the updated file
responsivity_files = {
    #"red": "/Users/lit./Desktop/iibproject/photocurrent/red/modified_responsivity_red.csv",
    #"green": "/Users/lit./Desktop/iibproject/photocurrent/green/modified_responsivity_green.csv",
    #"blue": "/Users/lit./Desktop/iibproject/photocurrent/blue/modified_responsivity_blue.csv",
    #"black": "/Users/lit./Desktop/iibproject/photocurrent/black/modified_responsivity_black.csv"
    'red': '/Users/lit./Desktop/iibproject/filter_TiO2_501nm_try.csv',
    'blue': '/Users/lit./Desktop/iibproject/filter_TiO2_540nm_try.csv',
    'green': '/Users/lit./Desktop/iibproject/filter_TiO2_582nm_try.csv'
}
wavelength_range = np.arange(400, 1001)  # Wavelengths from 400 to 1000 nm

spec_df = pd.read_csv(spectral_csv)
material_labels = spec_df["Material"]  
sample_indices = spec_df["Sample"]
spectra = spec_df.drop(columns=["Sample", "Material"])

spectra.columns = spectra.columns.astype(int)
spectra = spectra[wavelength_range.astype(int)]

# Convert reflectance to absorptance
absorptance = 1 - spectra.values  # shape = (num_samples, num_wavelengths)


responsivities = {}
for name, filepath in responsivity_files.items():
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    interp_func = np.interp(
        wavelength_range, df['wavelength'], df['responsivity']
    )
    responsivities[name] = interp_func  # shape = (num_wavelengths,)

photocurrent_data = {}
delta_lambda = 1  # Integration step size in nm

for name, r_curve in responsivities.items():
    photocurrent = np.sum(absorptance * r_curve, axis=1) * delta_lambda
    photocurrent_data[name] = photocurrent

photocurrent_df = pd.DataFrame(photocurrent_data)
photocurrent_df["Material"] = material_labels.values
photocurrent_df["Sample"] = sample_indices.values

cols = ["Sample", "Material", "red", "green", "blue"]
photocurrent_df = photocurrent_df[cols]

photocurrent_df.to_csv("photocurrent_data_5000_try.csv", index=False)
print("Photocurrent data saved to 'photocurrent_data_5000_try.csv'")
