import pandas as pd
import matplotlib.pyplot as plt

responsivity_files = {
    "S1": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/366_ag.csv",
    "S2": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/343_ag.csv",
    "S3": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/321_ag.csv",
    "S4": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/298_ag.csv",
    "S5": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/276_ag.csv",
    "S6": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/253_ag.csv",
    "S7": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/230_ag.csv",
    "S8": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/207_Ag.csv",
    "S9": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/182_ag.csv",
    #"Standard": "/Users/lit./Desktop/iibproject/photocurrent/black/modified_responsivity_black.csv",
    "E12": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/CMV4000_NIR_done.csv"
    #"Red": "/Users/lit./Desktop/iibproject/photocurrent/new/filter_r_final.csv",
    #"Green": "/Users/lit./Desktop/iibproject/photocurrent/new/filter_g_final.csv",
    #"Blue": "/Users/lit./Desktop/iibproject/photocurrent/new/filter_b_final.csv",
    #"G1": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/363_au.csv",
    #"G2": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/340_au.csv",
    #"G3": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/318_au.csv",
    #"G4":"/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/295_au.csv",
    #"G5": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/271_au.csv",
    #"G6": "/Users/lit./Desktop/iibproject/photocurrent/CMV4000_NIR/248_au.csv"

}

plt.figure(figsize=(10, 6))

for label, path in responsivity_files.items():
    try:
        df = pd.read_csv(path)
        # 自动识别列名
        if "Wavelength" in df.columns and "Responsivity" in df.columns:
            x = df["Wavelength"]
            y = df["Responsivity"]
        else:
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
        plt.plot(x, y, label=label)
    except Exception as e:
        print(f"Failed to load {label} from {path}: {e}")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Responsivity")
plt.title("CMV4000 Mono Responsivity Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
