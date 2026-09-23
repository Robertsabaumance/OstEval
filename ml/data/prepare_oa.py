"""Prepare a labeled, split dataset for knee osteoarthritis severity grading.

WHAT THIS SCRIPT DOES
The dataset (SilpaCS/kneeosteoarthritis on Hugging Face) contains 8,260 knee
X-ray images, already sorted into folders named 0-4, one per Kellgren-Lawrence
(KL) grade -- the standard clinical scale for osteoarthritis severity, where 0
is normal and 4 is severe. The folder name itself is the label, same principle
as the pneumonia warm-up's NORMAL/PNEUMONIA folders. This script:
  1. Walks the data/ folder and records every image's path and its grade
     (taken from the folder it's in).
  2. Splits all 8,260 images into train (70%), validation (15%), and test (15%),
     stratified by grade so each split keeps roughly the same per-grade
     proportions as the full dataset.
  3. Saves each group as its own CSV (oa_train.csv, oa_val.csv, oa_test.csv),
     with columns image_path and kl_grade.

WHY THIS DATASET IS DOWNLOADED DIRECTLY
Normally, Hugging Face can load a dataset with
load_dataset("SilpaCS/kneeosteoarthritis"). However, this dataset contains
incorrect label information. During loading, Hugging Face finds a commit hash
where it expects a class label such as 0, 1, 2, 3, or 4. This causes a
DatasetGenerationError before the images can be used. The problem is in the
dataset's metadata, so it cannot be fixed by this preparation script.

To work around the broken metadata, the script downloads the original ZIP file
directly with hf_hub_download(). This skips Hugging Face's dataset-loading
logic and retrieves the image archive instead:

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="SilpaCS/kneeosteoarthritis",
        filename="data.zip",
        repo_type="dataset",
    )

The downloaded ZIP file is then extracted. It contains folders named data/0
through data/4, which this script reads to find the images and their labels.

If this is run in Google Colab, the download and extraction must be repeated in
each new session unless the extracted data is copied to Google Drive, because
Colab storage is temporary.

CLASS BALANCE (as found by inspection, for reference)
Grade 0: 3,253 images | Grade 1: 1,495 | Grade 2: 2,175 | Grade 3: 1,086 |
Grade 4: 251. This is imbalanced -- Grade 4 is roughly 3% of the dataset --
so stratification (see below) matters here just as much as it did for
FracAtlas's fracture/non-fracture imbalance, and the same weighted-loss
approach will be needed during training, not handled by this script.

WHY THE SPLIT IS STRATIFIED
A plain random split could, by chance, leave a rare class (like Grade 4, only
251 images total) underrepresented or even absent from one of the three
groups. Stratifying by kl_grade keeps each grade's proportion consistent
across train/val/test, so all five grades are properly represented in every
group.

NO PROVIDED SPLIT EXISTED FOR THIS DATASET
Unlike FracAtlas, this dataset has no train/val/test split shipped with it at
all -- just the single data/ folder of all 8,260 images. The full 70/15/15
split here was built entirely from scratch.

OUTPUTS
oa_train.csv, oa_val.csv, oa_test.csv -- each with columns image_path (full
path to the image) and kl_grade (0-4, as an integer). Not committed to
version control (see .gitignore: ml/data/*.csv) -- reproducible by re-running
this script against a freshly downloaded copy of the dataset, and
random_state=42 ensures the exact same split every time.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "oa_data/data"

OUTPUT_TRAIN = "oa_train.csv"
OUTPUT_VAL = "oa_val.csv"
OUTPUT_TEST = "oa_test.csv"


def main():
    rows = []

    # Walk each grade folder (0-4) and record every image's path and label.
    # The folder name is the KL severity grade for all images inside it.
    for grade in sorted(os.listdir(DATA_DIR)):
        grade_dir = os.path.join(DATA_DIR, grade)
        if not os.path.isdir(grade_dir):
            continue
        for filename in os.listdir(grade_dir):
            image_path = os.path.join(grade_dir, filename)
            rows.append({"image_path": image_path, "kl_grade": int(grade)})

    df = pd.DataFrame(rows)

    # Show the total number of images found and the number in each severity grade.
    print(f"Total images found: {len(df)}")
    print(df["kl_grade"].value_counts().sort_index())

    # First, keep 70% for training and place the remaining 30% in temp_df.
    # test_size=0.30 means 30% goes to the second output, temp_df.
    # stratify= keeps the grade distribution consistent in both groups.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["kl_grade"], random_state=42
    )

    # Split the temporary 30% equally into validation and test sets.
    # test_size=0.50 sends half of temp_df to test_df and the other half to val_df,
    # giving 15% validation and 15% test of the original dataset.
    # Stratifying again preserves the grade distribution in both sets.
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["kl_grade"], random_state=42
    )

    # Save each split as a CSV file. index=False prevents pandas from adding
    # an extra column containing the DataFrame row numbers.
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    val_df.to_csv(OUTPUT_VAL, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    # Report how many images are in each split.
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Print the percentage of images in each severity grade for the train,
    # validation, and test sets.
    # Example:
    # Train grade distribution:
    # 0    0.3938
    # 1    0.1810
    # 2    0.2633
    # 3    0.1315
    # 4    0.0304

    print("\nTrain grade distribution:")
    print(train_df["kl_grade"].value_counts(normalize=True).sort_index())
    print("\nVal grade distribution:")
    print(val_df["kl_grade"].value_counts(normalize=True).sort_index())
    print("\nTest grade distribution:")
    print(test_df["kl_grade"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
