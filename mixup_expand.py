import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

input_csv = "/Users/lit./Desktop/iibproject/combined_spectra_EXPANDED_500.csv"  
mixup_count = 100
mixup_sim_threshold = 0.98

df = pd.read_csv(input_csv)
feature_cols = [str(wl) for wl in range(400, 1001)]
all_classes = df["Material"].unique()

augmented_rows = []


for label in all_classes:
    class_df = df[df["Material"] == label]
    real_samples = class_df[feature_cols].values
    n_real = len(real_samples)

    mixup_added = 0
    attempts = 0
    max_attempts = 300  

    while mixup_added < mixup_count and attempts < max_attempts:
        idx1, idx2 = np.random.choice(n_real, size=2, replace=False)
        lam = np.random.beta(0.4, 0.4)
        mixed = lam * real_samples[idx1] + (1 - lam) * real_samples[idx2]
        
        sim = cosine_similarity(mixed.reshape(1, -1), real_samples).max()
        if sim >= mixup_sim_threshold:
            row = dict(zip(feature_cols, mixed))
            row["Material"] = label
            row["Sample"] = f"aug_mixup_{label}_{np.random.randint(1e6)}"
            augmented_rows.append(row)
            mixup_added += 1

        attempts += 1


mixup_df = pd.DataFrame(augmented_rows)
combined_df = pd.concat([df, mixup_df], ignore_index=True)
combined_df.to_csv("mixup_expanded_600.csv", index=False)
print("Finished.")
