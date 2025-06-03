import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

df = pd.read_csv("/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/CMV4000_NIR.csv", header=None)
df.columns = ['Wavelength', 'Responsivity']  

interp_func = interp1d(df['Wavelength'], df['Responsivity'], kind='linear', bounds_error=False, fill_value=0)

new_wavelengths = np.arange(400, 1001, 1)  
new_responsivity = interp_func(new_wavelengths)

new_wavelengths = np.arange(400, 1001)  # λ in nm
responsivity_aw = new_responsivity * new_wavelengths / 1240  # A/W

new_df = pd.DataFrame({
    'Wavelength': new_wavelengths,
    'Responsivity': responsivity_aw
})

new_df.to_csv("/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/CMV4000_NIR_done.csv", index=False)
