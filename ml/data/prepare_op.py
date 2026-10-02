"""Prepare a labeled, split dataset for osteoporosis severity classification.

WHAT THIS SCRIPT DOES
The dataset (mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset on
Kaggle) contains 1,947 knee X-ray images, sorted into three folders --
Normal, Osteopenia, and Osteoporosis -- matching the three real-world
severity categories this app is designed to report. Unlike the osteoarthritis
dataset, the folder names here are words, not numbers, so this script maps
each folder name to a numeric label explicitly (see LABEL_MAP) rather than
converting the folder name directly. This script:
  1. Walks the OS Collected Data/ folder and records every image's path,
     numeric label, and class name (taken from the folder it's in).
  2. Splits all 1,947 images into train (70%), validation (15%), and test
     (15%), stratified by label so each split keeps roughly the same
     per-class proportions as the full dataset.
  3. Saves each group as its own CSV (op_train.csv, op_val.csv, op_test.csv),
     with columns image_path, label, and class_name.

WHY THIS PARTICULAR DATASET, AND NOT THE FIRST TWO TRIED
Two smaller osteoporosis datasets were checked and rejected before this one:
  - waseemkathia/osteoporosis (Hugging Face): only 74 images total (33 train
    + 4 test per class) -- far too small to train anything meaningful.
  - stevepython/osteoporosis-knee-xray-dataset (Kaggle): a larger zip, but
    once inspected, the actual usable images were only 186 per class (372
    total) across its raw folders, with an even smaller pre-made train/test
    split (33/33 train, 4/4 test) bundled inside that didn't even use the
    full 372. Still too small, and binary (Normal/Osteoporosis) only -- no
    Osteopenia category.
This dataset was found instead via a 2026 research paper on opportunistic
bone-loss screening, which uses this same dataset (alongside a related one)
for real published work -- a reasonable signal it's a legitimate, usable
source, not just an arbitrary upload.

WHY THIS DATASET IS DOWNLOADED DIRECTLY
Kaggle datasets aren't loaded the way Hugging Face datasets are. Instead of a
load_dataset()-style call, this one is fetched as a zip file directly via
Kaggle's public API download endpoint:

    !curl -L -o osteoporosis_multiclass.zip \\
      "https://www.kaggle.com/api/v1/datasets/download/mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset"

This worked without needing a Kaggle API key/login for this particular
dataset. The downloaded zip is then extracted, producing a folder named
"OS Collected Data" containing the three class subfolders this script reads.

If this is run in Google Colab, the download and extraction must be repeated
in each new session unless the extracted data is copied to Google Drive,
because Colab storage is temporary.

CLASS BALANCE (as found by inspection, for reference)
Normal: 780 images | Osteopenia: 374 images | Osteoporosis: 793 images.
Total: 1,947. Osteopenia is underrepresented relative to the other two
(roughly 19% of the dataset, versus ~40% each for Normal and Osteoporosis),
though nowhere near as extreme as osteoarthritis's Grade 4 imbalance.
Stratification (see below) still matters here, and a weighted-loss approach
will likely still be worth using during training, though less critically
than for the fracture or osteoarthritis models.

WHY THE SPLIT IS STRATIFIED
A plain random split could, by chance, leave Osteopenia (the smallest class)
underrepresented in one of the three groups. Stratifying by label keeps each
class's proportion consistent across train/val/test, so all three severity
categories are properly represented in every group.

NO PROVIDED SPLIT EXISTED FOR THIS DATASET
This dataset ships as a single pool of 1,947 images with no train/val/test
split included. The full 70/15/15 split here was built entirely from
scratch, same approach as the osteoarthritis dataset.

OUTPUTS
op_train.csv, op_val.csv, op_test.csv -- each with columns image_path (full
path to the image), label (0 = Normal, 1 = Osteopenia, 2 = Osteoporosis, as
an integer), and class_name (the original folder name, kept for
readability). Not committed to version control (see .gitignore:
ml/data/*.csv) -- reproducible by re-running this script against a freshly
downloaded copy of the dataset, and random_state=42 ensures the exact same
split every time.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "osteoporosis_multiclass_data/OS Collected Data"

OUTPUT_TRAIN = "op_train.csv"
OUTPUT_VAL = "op_val.csv"
OUTPUT_TEST = "op_test.csv"

# Map each folder name to a numeric label for training later.
LABEL_MAP = {"Normal": 0, "Osteopenia": 1, "Osteoporosis": 2}

def main():
    rows = []

    # Walk each class folder and record every image's path and label.
    # The folder name is the class for all images inside it.
    for class_name in sorted(os.listdir(DATA_DIR)):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir) or class_name not in LABEL_MAP:
            continue
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            rows.append({
                "image_path": image_path,
                "label": LABEL_MAP[class_name],
                "class_name": class_name,
            })

    df = pd.DataFrame(rows)

    # Show the total number of images found and the number in each class.
    print(f"Total images found: {len(df)}")
    print(df["class_name"].value_counts())

    # First, keep 70% for training and place the remaining 30% in temp_df.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )

    # Split the temporary 30% equally into validation and test sets.
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv(OUTPUT_TRAIN, index=False)
    val_df.to_csv(OUTPUT_VAL, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("\nTrain class distribution:")
    print(train_df["class_name"].value_counts(normalize=True))
    print("\nVal class distribution:")
    print(val_df["class_name"].value_counts(normalize=True))
    print("\nTest class distribution:")
    print(test_df["class_name"].value_counts(normalize=True))


if __name__ == "__main__":
    main()