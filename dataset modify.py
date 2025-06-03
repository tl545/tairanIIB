import pandas as pd
from sklearn.model_selection import train_test_split

file1_path = "/Users/lit./Desktop/iibproject/photocurrent_data_5000_try.csv"         
file2_path = "/Users/lit./Desktop/iibproject/photocurrent_new_filters_noiseless.csv"  


df1 = pd.read_csv(file1_path)
df1 = df1[df1["Material"] != 9]                   # delete label = 9
df1["Material"] = df1["Material"].replace(10, 9)  # label 10 -> 9

train_df1, test_df1 = train_test_split(
    df1,
    test_size=0.10,
    stratify=df1["Material"],
    random_state=42
)


df2 = pd.read_csv(file2_path)

train_df2, test_df2 = train_test_split(
    df2,
    test_size=0.10,
    stratify=df2["Material"],
    random_state=42
)

combined_train = pd.concat([train_df1, train_df2], ignore_index=True)
combined_test = pd.concat([test_df1, test_df2], ignore_index=True)


combined_train.to_csv("combined_train_NEW.csv", index=False)
combined_test.to_csv("combined_test_NEW.csv", index=False)

print("Combined training and testing datasets saved.")

