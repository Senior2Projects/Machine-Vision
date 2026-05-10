import pandas as pd
from sklearn.model_selection import train_test_split
import os

# load annotations
df = pd.read_csv("data/annotations.csv")

# create output folder
os.makedirs("data/splits", exist_ok=True)

train_list = []
val_list = []
test_list = []

# split PER CLASS
for label in df["label"].unique():

    class_df = df[df["label"] == label]

    # use all 1000 images
    class_df = class_df.sample(n=1000, random_state=42)

    # train = 700
    train_df, temp_df = train_test_split(
        class_df,
        test_size=300,
        random_state=42,
        shuffle=True
    )

    # val = 150, test = 150
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    train_list.append(train_df)
    val_list.append(val_df)
    test_list.append(test_df)

# combine
train_df = pd.concat(train_list)
val_df = pd.concat(val_list)
test_df = pd.concat(test_list)

# shuffle
train_df = train_df.sample(frac=1, random_state=42)
val_df = val_df.sample(frac=1, random_state=42)
test_df = test_df.sample(frac=1, random_state=42)

# save
train_df.to_csv("data/splits/train.csv", index=False)
val_df.to_csv("data/splits/val.csv", index=False)
test_df.to_csv("data/splits/test.csv", index=False)

print("Splits created successfully!")

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")