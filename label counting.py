import pandas as pd

train_df = pd.read_csv("/Users/lit./Desktop/iibproject/combined_train_RGB.csv")
test_df = pd.read_csv("/Users/lit./Desktop/iibproject/combined_test_RGB.csv")

train_counts = train_df['Material'].value_counts().sort_index()
test_counts = test_df['Material'].value_counts().sort_index()

all_labels = list(range(10))
train_counts = train_counts.reindex(all_labels, fill_value=0)
test_counts = test_counts.reindex(all_labels, fill_value=0)


total_counts = train_counts + test_counts

print("Train:")
print(train_counts)
print("\nTest")
print(test_counts)
print("\nTotal:")
print(total_counts)
