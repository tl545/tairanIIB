import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "/Users/lit./Desktop/iibproject/photocurrent/black/modified_responsivity_black.csv"
df = pd.read_csv(file_path)

# Clean up column names
df.columns = df.columns.str.strip().str.lower()  # Ensure 'wavelength', 'responsivity'

wavelength = df['wavelength'].values
responsivity = df['responsivity'].values

# Apply scaling factor (e.g., simulate 1% decrease in responsivity)
scaling_factor = 0.98  
scaled_responsivity = responsivity * scaling_factor

shift_amount = 1  # nm
shifted_wavelength = wavelength + shift_amount

shifted_responsivity = np.interp(wavelength, shifted_wavelength, scaled_responsivity)

plt.figure(figsize=(10, 6))
plt.plot(wavelength, responsivity, label="Original Responsivity", linewidth=2)
plt.plot(wavelength, shifted_responsivity, label="Scaled + Shifted (↑2%, →1nm)", linestyle='--', linewidth=2)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Responsivity (A/W)")
plt.title("Filter Responsivity: 2% Scaling and 1nm Shift")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
