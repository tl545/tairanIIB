import pandas as pd
import numpy as np
import random
import uuid

df = pd.read_csv('/Users/lit./Desktop/iibproject/combined_spectra_dataset.csv')

TARGET_COUNT = 700
FEATURE_COLUMNS = [str(wl) for wl in range(400, 1001)]
NOISE_SCALE = 0.02  


grouped = df.groupby('Material')
augmented_rows = []

for label, group in grouped:
    count = len(group)
    if count >= TARGET_COUNT:
        continue  

    num_to_generate = TARGET_COUNT - count
    print(f"Expand Material={label}; Now {count}; Will be expanded by {num_to_generate}")

    for _ in range(num_to_generate):
        sample = group.sample(n=1).iloc[0]
        spectrum = sample[FEATURE_COLUMNS].values.astype(float)

        noise = np.random.normal(0, NOISE_SCALE, size=spectrum.shape)
        noisy_spectrum = spectrum * (1 + noise)
        noisy_spectrum = np.clip(noisy_spectrum, 0, 1) 

        new_row = dict(zip(FEATURE_COLUMNS, noisy_spectrum))
        new_row['Material'] = label
        new_row['Sample'] = f"aug_{label}_{uuid.uuid4().hex[:8]}"  

        augmented_rows.append(new_row)

augmented_df = pd.DataFrame(augmented_rows)
final_df = pd.concat([df, augmented_df], ignore_index=True)

output_path = '/Users/lit./Desktop/iibproject/combined_spectra_EXPANDED_700.csv'
final_df.to_csv(output_path, index=False)

print(f"Finished.")
