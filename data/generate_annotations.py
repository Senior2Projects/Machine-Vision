import os
import csv

dataset_path = "data/raw"

output_csv = "data/annotations.csv"

rows = []

# loop through each class folder
for class_name in os.listdir(dataset_path):

    class_folder = os.path.join(dataset_path, class_name)

    # skip non-folders
    if not os.path.isdir(class_folder):
        continue

    # loop through images
    for image_name in os.listdir(class_folder):

        image_path = os.path.join(class_folder, image_name)

        rows.append([image_path, class_name])

# write CSV
with open(output_csv, mode="w", newline="") as file:

    writer = csv.writer(file)

    # header
    writer.writerow(["filepath", "label"])

    # rows
    writer.writerows(rows)

print(f"Saved {len(rows)} annotations to {output_csv}")