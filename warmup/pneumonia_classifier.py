"""Pneumonia chest X-ray classifier - practice warm-up before the main project.

Before starting work on my actual project, I decided to practice with a smaller
dataset first, to get comfortable with the process and learn more about machine
learning. It's an area I'm genuinely interested in - using Claude to help explain
the concepts to me along the way.

The dataset I chose was a chest X-ray pneumonia dataset from Kaggle. It's
relatively small, and chest X-rays are a well-researched, commonly used dataset
in machine learning, which made it a good, fast choice for practice.

I used Google Colab to do the training, since it provides a virtual GPU that's
much faster than running everything on my laptop's local CPU. I connected Colab
to my Google Drive, where the dataset folder was stored, so the training script
could read the images directly from there.

For this project, my approach wasn't to build an AI model from scratch - instead,
I used transfer learning, which is standard practice in the field of ML.

Transfer learning means starting from a model architecture that has already been
trained by researchers on a large, general dataset, so it already understands
basic visual patterns like edges, shapes, and textures, rather than starting from
nothing. In my case, I used ResNet18, an architecture that has already been
trained by Microsoft Research on over a million general images, with the
resulting pretrained model made freely available through PyTorch.

From there, I added my own additional training on top, using my specific
dataset - chest X-rays labelled either "pneumonia" or "normal." The process
works like a repeated feedback loop: the model looks at an image, makes a guess,
and the training code automatically compares that guess against the correct
label from the dataset - then the model slightly adjusts its internal settings
to be a little more accurate next time. This repeats thousands of times, across
the whole dataset, several times over.

The dataset I trained on had 5,219 images. I set the code to run 5 epochs with a
batch size of 32. An epoch is one full pass through the entire training dataset,
and batch size is how many images the model looks at together before making one
round of adjustments. I chose a batch size of 32 as a balance - small enough to
run efficiently, but large enough not to take an unreasonably long time.

These were the results:

Epoch 1/5 | Loss: 0.1194 | Validation accuracy: 93.75%
Epoch 2/5 | Loss: 0.0580 | Validation accuracy: 93.75%
Epoch 3/5 | Loss: 0.0498 | Validation accuracy: 100.00%
Epoch 4/5 | Loss: 0.0364 | Validation accuracy: 87.50%
Epoch 5/5 | Loss: 0.0315 | Validation accuracy: 75.00%

Looking at these results, the loss steadily decreased throughout training,
meaning the model kept improving on the training data. However, validation
accuracy - tested on a separate small batch of 19 images the model never
trained on - went up, then dropped toward the end (100% at epoch 3, down to 75%
by epoch 5). This is a common pattern called overfitting: by the later epochs,
the model was starting to memorize specifics of the training images rather than
learning generalizable patterns, which shows up as declining performance on new,
unseen data. With such a small validation set, each individual image also
carries a lot of weight in the percentage (roughly 5% per image), so some of
this drop is likely partly noise rather than a dramatic decline. This was a
useful thing to see firsthand, since overfitting is something I'll need to
watch for and address properly - with better data splits, more data, or early
stopping - once I move on to training the real disease detection models for my
main project.
"""

# This tells Python to keep type hints as text for now instead of trying to work them out immediately.
# E.g. we create a function whose type hint we don't know yet,
# We make the function a string so it can be found easier later when the tyoe hint is chosen.
from __future__ import annotations

# argparse lets us read options from the command line, like --epochs and --batch-size.
# An epoch is one full pass through the entire training dataset, and batch size is how
# many images the model looks at together before making one round of adjustments.
# e.g. python train.py --epochs 10 --batch-size 32
# This means the model will go through the entire training dataset 10 times, looking at
# 32 images at once each time it makes an adjustment.
import argparse

# Path is used to work with file and folder locations on disk.
from pathlib import Path

# torch is the main PyTorch library and contains tensors (grids of numbers), model
# tools, and the training helpers that run the guess -> check -> adjust loop for us.
import torch

# nn contains neural network layers like Linear and loss functions like CrossEntropyLoss.
# "Loss" is the single number that says how wrong a guess was compared to the real answer.
from torch import nn
# Adam is the optimizer: the part that actually nudges the model's internal settings
# a little bit each time, based on how wrong the last guess was.
from torch.optim import Adam
# DataLoader bundles images into batches so training is efficient.
from torch.utils.data import DataLoader
# torchvision gives us datasets, pretrained models, and image transforms.
from torchvision import datasets, models, transforms


# These numbers are the mean and standard deviation used for ImageNet-trained models.
# ResNet18 was originally trained by Microsoft Research on over a million general
# images, and these numbers describe the pixel color range it was trained to expect.
# Using the same numbers here keeps our X-ray pixels in the format the pretrained
# model already understands.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_dataloaders(data_root: Path, batch_size: int, num_workers: int):
    # This is the image-preprocessing pipeline: it turns each X-ray image into the
    # grid of pixel numbers the model actually works with.
    transform = transforms.Compose(
        [
            # Resize every image to the size expected by ResNet.
            transforms.Resize((224, 224)),
            # Turn each image into a grid of numbers (a tensor) with pixel values between 0 and 1.
            transforms.ToTensor(),
            # Shift and scale those pixel numbers so they match what the pretrained model expects.
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # ImageFolder reads folders like train/NORMAL and train/PNEUMONIA and uses the
    # folder name itself as the label -- the correct answer for every image inside it.
    # These labels are what the training loop checks each guess against.
    train_data = datasets.ImageFolder(data_root / "train", transform=transform)
    val_data = datasets.ImageFolder(data_root / "val", transform=transform)

    # A DataLoader groups examples into mini-batches (32 images at a time here) so
    # training is efficient instead of adjusting the model after every single image.
    # shuffle=True makes the training order random, which helps the model learn better.
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # Validation data should stay in the same order so we can compare results consistently.
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # Return the loaders plus the names of the classes so the model knows how many outputs to create.
    return train_loader, val_loader, train_data.classes


def build_model(num_classes: int) -> nn.Module:
    # Start with ResNet18 and its pretrained weights -- downloaded automatically from
    # PyTorch's official model hosting, originally trained by Microsoft Research on
    # over a million general photos. This is transfer learning: we reuse a model that
    # already understands basic visual patterns (edges, shapes, textures) instead of
    # starting from nothing.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace the last layer so it outputs one score for each class in this dataset.
    # For example, with 2 classes (NORMAL, PNEUMONIA), it will output 2 numbers.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    # Put the model in evaluation mode: we are only checking its guesses here, not
    # letting it learn from them, so nothing gets adjusted during this step.
    model.eval()
    correct = 0
    total = 0

    # no_grad tells PyTorch not to bother calculating how the model could be adjusted.
    # This saves memory and speed because we are only checking performance, not learning.
    with torch.no_grad():
        for images, labels in data_loader:
            # Move images and labels to the same device as the model, such as GPU or CPU.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # The model produces a score for each class; argmax picks the highest-scoring
            # class as its guess for that image.
            predictions = model(images).argmax(dim=1)

            # Compare each guess against the real label and count how many were correct.
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Return the fraction of correct guesses -- this is the validation accuracy.
    return correct / total if total else 0.0


def train(data_root: Path, epochs: int, batch_size: int, num_workers: int) -> None:
    # Confirm the dataset has the required train and val folders before starting.
    # If they are missing, stop early with a clear error message.
    if not (data_root / "train").is_dir() or not (data_root / "val").is_dir():
        raise FileNotFoundError(
            f"Expected train/ and val/ folders under dataset root: {data_root}"
        )

    # Pick the GPU if one is available; otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the data loaders and get the class names (the labels) from the folder structure.
    train_loader, val_loader, class_names = build_dataloaders(
        data_root, batch_size, num_workers
    )

    # Build the model, with its pretrained weights, and send it to the chosen device.
    model = build_model(len(class_names)).to(device)

    # CrossEntropyLoss is the "checking" step: it compares the model's guess to the
    # real label and produces a single number describing how wrong that guess was.
    loss_function = nn.CrossEntropyLoss()

    # Adam is the "adjusting" step: it uses that wrongness to nudge the model's
    # internal settings a little bit, so the next guess on a similar image is a
    # bit more accurate.
    optimizer = Adam(model.parameters(), lr=0.001)

    # Print a quick summary for the user.
    print(f"Device: {device}")
    print(f"Classes: {class_names}")

    # Train for the number of epochs chosen by the user (5, in my run). One epoch =
    # one full pass through every training image (5,219 images, in my run).
    for epoch in range(epochs):
        # Put the model in training mode so it is allowed to learn from this data.
        model.train()
        running_loss = 0.0

        # Loop through each mini-batch of training images (32 images per batch, in my
        # run) -- this is the repeated guess -> check -> adjust loop happening over
        # and over.
        for images, labels in train_loader:
            # Move the batch to the current device so the model can process it.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Clear old adjustment information from the previous batch before computing new ones.
            optimizer.zero_grad()

            # Guess: run this batch of images through the model.
            predictions = model(images)

            # Check: compare the guesses to the real labels and get a wrongness score (the loss).
            loss = loss_function(predictions, labels)

            # Work out exactly how the model should change to reduce that wrongness.
            loss.backward()

            # Adjust: actually update the model by that small calculated amount.
            optimizer.step()

            # Add this batch's loss to a running total so we can compute an average later.
            running_loss += loss.item() * images.size(0)

        # Average loss over the whole training set for this epoch. In my run this
        # dropped steadily each epoch (0.1194 -> 0.0580 -> 0.0498 -> 0.0364 -> 0.0315),
        # meaning the model kept improving on the training data it was seeing.
        average_loss = running_loss / len(train_loader.dataset)

        # Check the model on validation data (19 images it never trained on) after
        # this epoch, to see whether it learned a general pattern or just memorized
        # the training set. In my run this went 93.75% -> 93.75% -> 100% -> 87.5% ->
        # 75%: it went up, then dropped toward the end. That drop, even while loss
        # kept improving, is a classic sign of overfitting -- the model was starting
        # to memorize specifics of the training images rather than learning
        # generalizable patterns. With only 19 validation images, each one is worth
        # about 5% of the score, so part of that drop is likely just noise rather
        # than a dramatic decline. Either way, this is something to watch for and
        # address properly (better data splits, more data, or early stopping) once
        # I move on to training the real disease detection models for my main project.
        validation_accuracy = evaluate(model, val_loader, device)

        # Print the training progress for this epoch.
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.2%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pneumonia image classifier.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/content/chest_xray"),
        help="Folder containing train/, val/, and optionally test/.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    # Batch size of 32 was chosen as a balance: small enough to run efficiently,
    # but large enough not to take an unreasonably long time.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    # parse_known_args ignores extra arguments Colab/Jupyter injects behind the
    # scenes (like -f ...kernel.json), instead of crashing on them.
    arguments, _ = parser.parse_known_args()
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    train(
        data_root=arguments.data_root,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
    )