import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

tio2_data = pd.read_csv('/Users/lit./Desktop/iibproject/Sarkar.csv')
tio2_data.columns = tio2_data.columns.str.strip()  # Remove spaces
tio2_data = tio2_data.apply(pd.to_numeric, errors='coerce').dropna()

wavelengths_tio2 = tio2_data["wl"].values * 1000   #  in nm
n_tio2_values = tio2_data["n"].values

n_tio2_interp = interp1d(wavelengths_tio2, n_tio2_values, kind='linear', fill_value='extrapolate')

wavelengths_nm = np.linspace(400, 1000, 1000)  

plt.figure(figsize=(8, 5))
plt.plot(wavelengths_tio2, n_tio2_values, label='Original Data (CSV)', color='red', linestyle='dotted')
plt.plot(wavelengths_nm, n_tio2_interp(wavelengths_nm), label='Interpolated Function', color='blue')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Refractive Index (n)')
plt.title('Refractive Index of TiO2')
plt.legend()
plt.grid()
plt.show()

incident_spectrum = np.ones_like(wavelengths_nm)  

def gaussian_spectrum(wl, center=500, width=100):
    return np.exp(-((wl - center) ** 2) / (2 * width ** 2))

# incident_spectrum = gaussian_spectrum(wavelengths_nm)

plt.figure(figsize=(8, 5))
plt.plot(wavelengths_nm, incident_spectrum, label='Incident Spectrum', color='purple')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.title('Incident Light Spectrum')
plt.legend()
plt.grid()
plt.show()


d = 30  # in nm

def n_al2o3(wl):
    wl_micron = wl / 1000  # convert nm to µm
    return np.sqrt(1 + (1.4313493 * wl_micron**2) / (wl_micron**2 - 0.0726631**2) + 
                      (0.65054713 * wl_micron**2) / (wl_micron**2 - 0.1193242**2) + 
                      (5.3414021 * wl_micron**2) / (wl_micron**2 - 18.028251**2))

n_al2o3_values = [n_al2o3(wl) for wl in wavelengths_nm]

# Plot refractive index of Al2O3
plt.figure(figsize=(8, 5))
plt.plot(wavelengths_nm, n_al2o3_values, label='n(Al2O3)', color='green')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Refractive Index (n)')
plt.title('Refractive Index of Al₂O₃')
plt.legend()
plt.grid()
plt.show()

# single-film
transmission_single = []
for wl in wavelengths_nm:
    n0 = 1.0  
    ns = 1.0 
    n_film = n_tio2_interp(wl) 
    
    delta = (2 * np.pi / (wl)) * n_film * d
    
    B = np.cos(delta) + (1j / n_film) * np.sin(delta) * ns
    C = 1j * n_film * np.sin(delta) + np.cos(delta) * ns
    
    T = (4 * n0 * np.real(ns)) / (np.abs(n0 * B + C) ** 2)
    transmission_single.append(T)

transmission_single = np.array(transmission_single)

transmitted_spectrum_single = incident_spectrum * transmission_single

plt.figure(figsize=(8, 5))
plt.plot(wavelengths_nm, transmitted_spectrum_single, label='Single Film Transmission (TiO2)', color='purple')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.title('Transmitted Light Spectrum Through Single TiO2 Layer')
plt.legend()
plt.grid()
plt.show()

# for multilayer 
transmission_multi = []
for wl in wavelengths_nm:
    n_air = 1.0  
    n_sub = 1.0
    n_tio2_layer = n_tio2_interp(wl)  
    layers = [(n_tio2_layer, d), (n_al2o3(wl), d), (n_tio2_layer, d), (n_al2o3(wl), d)]
    
    M = np.array([[1, 0], [0, 1]], dtype=complex)
    
    for n_j, d_j in layers:
        delta_j = (2 * np.pi / (wl)) * n_j * d_j
        Mj = np.array([[np.cos(delta_j), 1j * np.sin(delta_j) / n_j],
                       [1j * n_j * np.sin(delta_j), np.cos(delta_j)]], dtype=complex)
        M = M @ Mj
    
    B = M[0, 0] * 1 + M[0, 1] * n_sub
    C = M[1, 0] * 1 + M[1, 1] * n_sub
    
    T = (4 * n_air * np.real(n_sub)) / (np.abs(n_air * B + C) ** 2)
    transmission_multi.append(T)


transmission_multi = np.array(transmission_multi)

transmitted_spectrum_multi = incident_spectrum * transmission_multi

plt.figure(figsize=(8, 5))
plt.plot(wavelengths_nm, transmitted_spectrum_multi, label='Transmitted Spectrum (Multilayer)', color='blue')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.title('Transmitted Light Spectrum After Multi-Layer Filter')
plt.legend()
plt.grid()
plt.show()
