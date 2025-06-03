import matplotlib.pyplot as plt
import numpy as np

# Data
filename = '/Users/lit./Desktop/iibproject/Dataset of spectral reflectances and hypercubes of submerged plastic litter, including COVID-19 medical waste, pristine plasti>_1_all/L0_SE_Raw_Cleaned_Datafiles/2021_Sep_01_SR-3501_SN19387Q4__00371.sed'

def extract(file_path):
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
    return filtered_wavelengths, filtered_reflectance

wavelength = extract(filename)[0]
reflectance = extract(filename)[1]

# Plot
plt.figure(figsize=(8, 4))
plt.plot(wavelength, reflectance, color='black', linewidth=2)
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.show()
