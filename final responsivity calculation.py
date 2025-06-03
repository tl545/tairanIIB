import pandas as pd
import matplotlib.pyplot as plt

mono_df = pd.read_csv("/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/CMV4000_NIR_done.csv")
filter_df = pd.read_csv("/Users/lit./Desktop/iibproject/filter_TiO2_248nm_Au.csv")

assert all(mono_df["Wavelength"] == filter_df["Wavelength"]), "Wavelength Inconsistent"

result_df = pd.DataFrame({
    "Wavelength": mono_df["Wavelength"],
    "Responsivity": mono_df["Responsivity"] * filter_df["Responsivity"]
})

result_df.to_csv("/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/248_au.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(result_df["Wavelength"], result_df["Responsivity"], label="Effective Responsivity", linewidth=2)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Responsivity (A/W)")
plt.title("Effective Responsivity Curve (Mono × Filter)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
