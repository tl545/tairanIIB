import numpy as np
import matplotlib.pyplot as plt
from tmm import coh_tmm

target_wavelengths = [950,900,850,800,750,700]  # in nm
# there will be ~10 nm mismatch between the target wavelength and the actual peak

n_air = 1.0
n_substrate = 1.5   # assume to be glass
thickness_au = 30   # in nm
m = 1  # Mode number

wavelengths = np.linspace(400, 1000, 600)  # range 400 - 1000 nm

wavelengths_known = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]  

n_tio2_vals = [2.69, 2.56, 2.48, 2.44, 2.40, 2.38, 2.36, 2.35, 2.34, 2.33, 2.33, 2.32, 2.31]
n_vals = [1.60, 1.47, 0.91, 0.43, 0.28, 0.20, 0.18, 0.18, 0.19, 0.20, 0.22, 0.24, 0.26]    #Au
k_vals = [2.00, 1.97, 1.96, 2.59, 3.18, 3.72, 4.22, 4.68, 5.12, 5.53, 5.94, 6.35, 6.74]    #Au
#n_vals = [0.05, 0.04, 0.05, 0.06, 0.06, 0.05, 0.04, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04]    #Ag
#k_vals = [2.10, 2.65, 3.13, 3.60, 4.01, 4.41, 4.80, 5.19, 5.57, 5.97, 6.37, 6.74, 7.12]    #Ag

def n_k_tio2(wl):
    n = np.interp(wl, wavelengths_known, n_tio2_vals)
    k = 0
    return n + 1j * k

def n_k_au(wl):
    n = np.interp(wl, wavelengths_known, n_vals)
    k = np.interp(wl, wavelengths_known, k_vals)
    return n + 1j * k

# Phase shift at TiO2–Au interface
def phase_shift_tio2_au(wl, n1):
    n2 = n_k_au(wl)
    r = (n1 - n2) / (n1 + n2)
    return np.angle(r)  # in radians

corrected_thicknesses = []
for wl in target_wavelengths:
    n_tio2 = np.interp(wl, wavelengths_known, n_tio2_vals)
    phi = 2 * phase_shift_tio2_au(wl, n_tio2)   # symmetric mirror
    k0 = 2 * np.pi / wl
    d = (m * wl - phi / k0) / (2 * n_tio2)
    corrected_thicknesses.append(d)

corrected_thicknesses = [round(d, 1) for d in corrected_thicknesses]

print("Target λ (nm) | Corrected TiO2 thickness (nm)")
print("---------------------------------------------")
for wl, d in zip(target_wavelengths, corrected_thicknesses):
    print(f"{wl:14.1f} | {d:27.1f}")

#corrected_thicknesses = [100, 250]

plt.figure()
for d_tio2 in corrected_thicknesses:
    T = []
    for wl in wavelengths:
        n_list = [n_air, n_k_au(wl), n_k_tio2(wl), n_k_au(wl), n_substrate]
        d_list = [np.inf, thickness_au, d_tio2, thickness_au, np.inf]
        data = coh_tmm('s', n_list, d_list, 0, wl)
        T.append(data['T'])
    plt.plot(wavelengths, T, label=f'TiO2 = {d_tio2:.1f} nm')

plt.xlabel("Wavelength")
plt.ylabel("Transmission")
plt.title("Fabry-Perot Filter Spectrum")
plt.legend()
plt.grid(True)
plt.show()



# Export transmission spectra for each filter to CSV
wavelengths_fine = np.arange(400, 1001, 1)  # 1 nm resolution

for d_tio2 in corrected_thicknesses:
    T_fine = []
    for wl in wavelengths_fine:
        n_list = [n_air, n_k_au(wl), n_k_tio2(wl), n_k_au(wl), n_substrate]
        d_list = [np.inf, thickness_au, d_tio2, thickness_au, np.inf]
        data = coh_tmm('s', n_list, d_list, 0, wl)
        T_fine.append(data['T'])
    
    # Save to CSV
    import pandas as pd
    df_out = pd.DataFrame({
        "Wavelength": wavelengths_fine,
        "Responsivity": T_fine
    })
    filename = f"filter_TiO2_{int(d_tio2)}nm_Au.csv"
    df_out.to_csv(filename, index=False)
    print(f"Saved: {filename}")
