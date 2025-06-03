import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random

#spectral_csv = "/Users/lit./Desktop/iibproject/combined_spectra_EXPANDED_500.csv"
spectral_csv = "expanded_train_RGB_mixup600.csv"
#spectral_csv = "/Users/lit./Desktop/iibproject/train_photocurrent.csv"

responsivity_files = {
    #"red": "/Users/lit./Desktop/iibproject/photocurrent/red/modified_responsivity_red.csv",
    #"green": "/Users/lit./Desktop/iibproject/photocurrent/green/modified_responsivity_green.csv",
    #"blue": "/Users/lit./Desktop/iibproject/photocurrent/blue/modified_responsivity_blue.csv",
    #"black": "/Users/lit./Desktop/iibproject/photocurrent/black/modified_responsivity_black.csv",
    #"N1": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/CMV4000_267.csv",
    #"N2": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/CMV4000_313.csv",
    #"N3": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/CMV4000_354.csv",
    #"N4": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/CMV4000_166.csv"
    #"S1": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/366_ag.csv",
    #"S2": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/343_ag.csv",
    #"S3": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/321_ag.csv",
    #"S4": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/298_ag.csv",
    #"S5": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/276_ag.csv",
    #"S6": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/253_ag.csv",
    #"S7": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/230_ag.csv",
    #"S8": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/207_ag.csv",
    #"S9": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/182_ag.csv",
    "E12": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/CMV4000_NIR_done.csv",
    "R": "/Users/lit./Desktop/iibproject/photocurrent/new/filter_r_final.csv",
    "G": "/Users/lit./Desktop/iibproject/photocurrent/new/filter_g_final.csv",
    "B": "/Users/lit./Desktop/iibproject/photocurrent/new/filter_b_final.csv",
    "G1": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/363_au.csv",
    "G2": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/340_au.csv",
    "G3": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/318_au.csv",
    "G4":"/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/295_au.csv",
    "G5": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/271_au.csv",
    "G6": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/248_au.csv"

}

wavelength_range = np.arange(400, 1001)  # nm
NOISE_STD = 0  # 1/100    # Photocurrent Gaussian noise
SCALE_RANGE = 0  # 1/100  # Scaling ±n%
SHIFT_RANGE = 0  # 0.5   # Wavelength shift ±n nm
delta_lambda = 1   


spec_df = pd.read_csv(spectral_csv)
material_labels = spec_df["Material"]
sample_indices = spec_df["Sample"]
spectra = spec_df.drop(columns=["Sample", "Material"])

spectra.columns = spectra.columns.astype(int)
spectra = spectra[wavelength_range.astype(int)]
absorptance = 1 - spectra.values  # shape = (num_samples, num_wavelengths)


baseline_responsivities = {}
distorted_responsivities = {}

for name, filepath in responsivity_files.items():
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    base_wl = df['wavelength'].values
    base_resp = df['responsivity'].values

    interp_func = interp1d(base_wl, base_resp, kind='cubic', fill_value="extrapolate")
    baseline_curve = interp_func(wavelength_range)
    baseline_responsivities[name] = baseline_curve

    scale = np.random.uniform(1 - SCALE_RANGE, 1 + SCALE_RANGE)
    shift = np.random.uniform(-SHIFT_RANGE, SHIFT_RANGE)
    shifted_wavelengths = wavelength_range + shift
    distorted_curve = interp_func(shifted_wavelengths) * scale
    distorted_responsivities[name] = distorted_curve

plt.figure(figsize=(12, 8))
for i, name in enumerate(responsivity_files):
    plt.plot(wavelength_range, baseline_responsivities[name], label=f"{name.capitalize()} - Original")
    plt.plot(wavelength_range, distorted_responsivities[name], linestyle='--', label=f"{name.capitalize()} - Distorted")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Responsivity (A/W)")
plt.title("Filter Responsivity Curves with Scaling and Shift")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


photocurrent_data = {key: [] for key in responsivity_files}
num_samples = absorptance.shape[0]

sample_indices_to_print = sorted(random.sample(range(num_samples), 5))
print_summary = {}

for i in range(num_samples):
    absorb = absorptance[i, :]
    if i in sample_indices_to_print:
        print_summary[i] = {"original": {}, "noisy": {}}

    for name in distorted_responsivities:
        response = distorted_responsivities[name]

        I = np.sum(absorb * response) * delta_lambda

        std_dev = NOISE_STD * I
        noise = np.random.normal(0.0, std_dev)
        I_noisy = I + noise

        photocurrent_data[name].append(I_noisy)

        if i in sample_indices_to_print:
            print_summary[i]["original"][name] = I
            print_summary[i]["noisy"][name] = I_noisy


print("\n=== Sampled Photocurrent Comparison (Before vs After Noise) ===")
for idx in sample_indices_to_print:
    print(f"\nSample {idx}:")
    for f in ["E12", "R", "G", "B", "G1", "G2", "G3", "G4", "G5", "G6"]:
        original = print_summary[idx]["original"][f]
        noisy = print_summary[idx]["noisy"][f]
        print(f"  {f.capitalize()} | Original: {original:.4f}  →  Noisy: {noisy:.4f}")


photocurrent_df = pd.DataFrame(photocurrent_data)
photocurrent_df["Material"] = material_labels.values
photocurrent_df["Sample"] = sample_indices.values

cols = ["Sample", "E12", "R", "G", "B", "G1", "G2", "G3", "G4", "G5", "G6", "Material"]
photocurrent_df = photocurrent_df[cols]

#output_file = "photocurrent_RGB_500_noisy.csv"
output_file = "train_10filter_600_gold.csv"
#output_file = "test_4filter_600_clean_NIR.csv"
#output_file = "train_clean.csv"

photocurrent_df.to_csv(output_file, index=False)
print("Finished.")

