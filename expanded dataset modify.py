import pandas as pd
from sklearn.model_selection import train_test_split

#input_path = "/Users/lit./Desktop/iibproject/photocurrent_RGB_500_noisy.csv"
input_path =  "/Users/lit./Desktop/iibproject/combined_spectra_dataset.csv"
df = pd.read_csv(input_path)

train_df, test_df = train_test_split(
    df,
    test_size=0.10,
    stratify=df["Material"],
    random_state=42
)


train_df.to_csv("train_photocurrent.csv", index=False)
test_df.to_csv("test_photocurrent.csv", index=False)

print("Files saved.")
