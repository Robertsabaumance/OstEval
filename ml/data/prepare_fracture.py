"""Prepare a labeled, split dataset from FracAtlas for fracture classification.

WHAT THIS SCRIPT DOES
FracAtlas is a public dataset of 4,083 X-ray images covering the hand, leg,
hip, and shoulder. Every image has a known, correct label: whether it shows a
fracture or not. That label, along with body-part flags and a few other
details, is stored centrally in one file, dataset.csv -- not in the folder
structure alone. This script:
  1. Reads dataset.csv.
  2. Builds the full file path to each image, based on its filename (from
     dataset.csv) and which subfolder it lives in (Fractured/ or
     Non_fractured/, determined by the "fractured" column).
  3. Splits all 4,083 labeled images into three groups: 70% train, 15%
     validation, 15% test.
  4. Saves each group as its own CSV (fracture_train.csv, fracture_val.csv,
     fracture_test.csv), each with just two columns: image_path and fractured.

WHY THE SPLIT IS "STRATIFIED"
Only 719 of the 4,083 images (17.6%) actually show a fracture -- the rest are
normal. A plain random split could, by chance, put an unrepresentative share
of fractured images into one group over another (e.g. too few fractures in
the training set for the model to learn from properly). "Stratified" means
the split is done in a way that deliberately preserves that same ~17.6%
fracture rate in each of the three groups, so training, validation, and
testing are all working with a similarly balanced (or similarly imbalanced)
sample. This was confirmed after running the script: train, val, and test
all came out at approximately 17.6-17.65% fractured, matching the full
dataset almost exactly.

WHY THE DATASET'S OWN PROVIDED SPLIT WAS NOT USED
FracAtlas ships with its own train.csv / valid.csv / test.csv files, found
under Utilities/Fracture Split/. These were inspected before writing this
script and found to only include 719 rows combined -- exactly matching the
number of *fractured* images in the dataset, not the full 4,083. This strongly
suggests that split was built for a different task: fracture localization or
segmentation (drawing a bounding box or outline around the fracture itself),
which only makes sense for images that actually contain a fracture. Since
this script prepares data for a simpler task -- classifying an image as
fractured or not, across ALL images including normal ones -- that provided
split does not fit, and a new one covering all 4,083 images was built instead.

DATASET.CSV COLUMNS (for reference)
image_id, hand, leg, hip, shoulder, mixed, hardware, multiscan, fractured,
fracture_count, frontal, lateral, oblique
Only image_id and fractured are used by this script. The body-part columns
(hand/leg/hip/shoulder/mixed) are preserved in the DataFrame during
processing and could be used later for body-part-aware model routing, but
are dropped before the final CSVs are written, since this script's current
scope is fracture classification only.

HOW TO GET THE DATASET BEFORE RUNNING THIS SCRIPT
FracAtlas is hosted on Figshare (article ID 22363012). Hugging Face's mirror
of this dataset does not work as of this writing (its loading script uses a
mechanism Hugging Face's `datasets` library has since deprecated and blocked).
Figshare's bulk "download all files as zip" endpoint also does not work
directly with a simple request -- it responds with HTTP 202 (Accepted) and
processes the request asynchronously rather than returning the file. Instead,
use Figshare's API to get a direct link to the individual file:

    import requests
    api_url = "https://api.figshare.com/v2/articles/22363012"
    response = requests.get(api_url)
    data = response.json()
    for file in data["files"]:
        print(file["name"], file["download_url"])

As of this writing, this returns one file, FracAtlas.zip (~322MB), at
https://ndownloader.figshare.com/files/65518038 -- note this exact URL may
change if Figshare re-processes the file in future; re-running the API call
above will always return the current correct URL. Download and unzip that
file so the following paths exist relative to wherever this script is run:
  fracatlas_data/FracAtlas/dataset.csv
  fracatlas_data/FracAtlas/images/Fractured/...
  fracatlas_data/FracAtlas/images/Non_fractured/...
If run in Google Colab, note that Colab's storage is temporary: unless the
downloaded/unzipped data is also copied to Google Drive, it will need to be
re-downloaded in any new Colab session.

OUTPUTS
fracture_train.csv, fracture_val.csv, fracture_test.csv -- each with columns
image_path (full path to the image file) and fractured (0 or 1). These are
not committed to version control (see .gitignore: ml/data/*.csv) since they
are fully reproducible by re-running this script against a fresh copy of the
dataset, and random_state=42 ensures the exact same split every time.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_CSV = "fracatlas_data/FracAtlas/dataset.csv"
IMAGES_DIR = "fracatlas_data/FracAtlas/images"

OUTPUT_TRAIN = "fracture_train.csv"
OUTPUT_VAL = "fracture_val.csv"
OUTPUT_TEST = "fracture_test.csv"


def main():
    # Load the full labeled dataset. Columns include image_id, body part flags
    # (hand/leg/hip/shoulder/mixed), and "fractured" (0 or 1) -- the label we need.
    df = pd.read_csv(DATASET_CSV)

    # Build the full path to each image. Images are sorted into Fractured/ and
    # Non_fractured/ subfolders based on the "fractured" column.
    df["image_path"] = (
        IMAGES_DIR
        + "/"
        + df["fractured"].map({1: "Fractured", 0: "Non_fractured"})
        + "/"
        + df["image_id"]
    )

    # Keep just what we need for classification. Body part columns (hand, leg,
    # hip, shoulder, mixed) are still available in df if needed later for
    # body-part-aware routing in the app.
    clean_df = df[["image_path", "fractured"]]

    # First, keep 70% for training and place the remaining 30% in temp_df.
    # test_size=0.30 means 30% goes to the second output, temp_df.
    # stratify= keeps the fracture rate consistent in both groups.
    train_df, temp_df = train_test_split(
        clean_df, test_size=0.30, stratify=clean_df["fractured"], random_state=42
    )

    # Split the temporary 30% equally into validation and test sets.
    # test_size=0.50 sends half of temp_df to test_df and the other half to val_df,
    # giving 15% validation and 15% test of the original dataset.
    # Stratifying again preserves the ~17.6% fracture rate in both sets.
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["fractured"], random_state=42
    )

    # Save each split as a CSV file. index=False prevents pandas from adding
    # an extra column containing the DataFrame row numbers.
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    val_df.to_csv(OUTPUT_VAL, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    # Report how many images are in each split.
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # The mean of the 0/1 fractured column is the proportion of fractured images
    # in each split. For example, 0.176 means approximately 17.6% are fractured.
    print("Train fracture rate:", train_df["fractured"].mean())
    print("Val fracture rate:", val_df["fractured"].mean())
    print("Test fracture rate:", test_df["fractured"].mean())


if __name__ == "__main__":
    main()
