#!/usr/bin/env python3

import os
import random
import shutil
from pydub import AudioSegment

# ------------------------------------------
# Configuration
# ------------------------------------------

# Specify the folders where you have your snore and non-snore files.
# These should already exist with your original downloaded Kaggle files.
SNORE_DIR = "./Snoring Dataset/1"         # e.g., contains 1_0.wav to 1_499.wav
NON_SNORE_DIR = "./Snoring Dataset/0" # e.g., contains 0_0.wav to 0_499.wav

# The directory structure we will create
DATASET_DIR = "Dataset"
RAW_DIR = os.path.join(DATASET_DIR, "Raw")

TRAIN_DIR = os.path.join(RAW_DIR, "Train")
VAL_DIR   = os.path.join(RAW_DIR, "Val")
TEST_DIR  = os.path.join(RAW_DIR, "Test")

# Ratios for splitting data
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Check that the ratios sum to 1
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, \
    "Train/Val/Test split ratios must sum to 1."


# ------------------------------------------
# Helper Functions
# ------------------------------------------

def create_folder_structure():
    """
    Create the folder structure:
        Dataset/Raw
        ├─ Train
        │   ├─ original
        │   │   ├─ 1 (snore)
        │   │   └─ 0 (non-snore)
        │   └─ mixing
        │       ├─ noisy (snore+non-snore)
        │       └─ 0 (non-snore)
        ├─ Val
        └─ Test
    """
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(split_dir, exist_ok=True)

        # original/(1,0)
        original_dir = os.path.join(split_dir, "original")
        os.makedirs(os.path.join(original_dir, "1"), exist_ok=True)
        os.makedirs(os.path.join(original_dir, "0"), exist_ok=True)

        # mixing/(noisy, 0)
        mixing_dir = os.path.join(split_dir, "mixing")
        os.makedirs(os.path.join(mixing_dir, "noisy"), exist_ok=True)
        os.makedirs(os.path.join(mixing_dir, "0"), exist_ok=True)


def get_audio_file_paths(snore_dir, non_snore_dir):
    """
    Retrieve lists of audio file paths for snore and non-snore.
    """
    snore_files = [os.path.join(snore_dir, f)
                   for f in os.listdir(snore_dir)
                   if os.path.isfile(os.path.join(snore_dir, f))]
    
    non_snore_files = [os.path.join(non_snore_dir, f)
                       for f in os.listdir(non_snore_dir)
                       if os.path.isfile(os.path.join(non_snore_dir, f))]
    
    return snore_files, non_snore_files


def split_data(file_list, train_ratio, val_ratio, test_ratio):
    """
    Shuffle and split file_list into train, val, test sets.
    Returns: train_files, val_files, test_files
    """
    random.shuffle(file_list)
    n = len(file_list)
    train_end = int(train_ratio * n)
    val_end   = train_end + int(val_ratio * n)

    train_files = file_list[:train_end]
    val_files   = file_list[train_end:val_end]
    test_files  = file_list[val_end:]
    return train_files, val_files, test_files


def copy_files(file_paths, dest_dir):
    """
    Copy each file in file_paths to the destination directory dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)
    for fp in file_paths:
        shutil.copy(fp, dest_dir)


def mix_audio_files(snore_file, non_snore_file, output_path):
    """
    Overlay snore_file and non_snore_file using pydub (both ~1 second).
    Export the result to output_path as WAV.
    """
    snore = AudioSegment.from_file(snore_file)
    non_snore = AudioSegment.from_file(non_snore_file)

    # Overlay them
    mixed = snore.overlay(non_snore)
    mixed.export(output_path, format="wav")


def create_noisy_files(snore_list, non_snore_list, split_path):
    """
    For each snore file in the snore_list, pick a random non-snore file
    and create a noisy (mixed) version that retains the snore file's base name,
    appending '_noisy' before the file extension.

    Example:
      If snore file is '1_44.wav', output is '1_44_noisy.wav'.
    """
    noisy_dir = os.path.join(split_path, "mixing", "noisy")
    
    for s_file in snore_list:
        if not non_snore_list:
            print("Warning: No non-snore files available to mix!")
            break
        
        chosen_non_snore = random.choice(non_snore_list)
        
        base_name = os.path.basename(s_file)               # e.g. "1_44.wav"
        file_root, file_ext = os.path.splitext(base_name)  # ("1_44", ".wav")

        out_filename = f"{file_root}_noisy{file_ext}"      # "1_44_noisy.wav"
        output_path = os.path.join(noisy_dir, out_filename)

        # Mix and save
        mix_audio_files(s_file, chosen_non_snore, output_path)


# ------------------------------------------
# Main logic
# ------------------------------------------
def main():
    # 1. Create the top-level folder 'Dataset' (if it doesn't exist)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # 2. Create subfolders under Dataset/Raw
    create_folder_structure()

    # 3. Gather snore and non-snore files
    snore_files, non_snore_files = get_audio_file_paths(SNORE_DIR, NON_SNORE_DIR)

    # 4. Split into Train, Val, Test
    snore_train, snore_val, snore_test = split_data(
        snore_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    non_snore_train, non_snore_val, non_snore_test = split_data(
        non_snore_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    # 5. Copy snore files into 'original/1'
    copy_files(snore_train, os.path.join(TRAIN_DIR, "original", "1"))
    copy_files(snore_val, os.path.join(VAL_DIR, "original", "1"))
    copy_files(snore_test, os.path.join(TEST_DIR, "original", "1"))

    # 6. Copy non-snore files into 'original/0'
    copy_files(non_snore_train, os.path.join(TRAIN_DIR, "original", "0"))
    copy_files(non_snore_val, os.path.join(VAL_DIR, "original", "0"))
    copy_files(non_snore_test, os.path.join(TEST_DIR, "original", "0"))

    # 7. Copy non-snore files into 'mixing/0' as well
    copy_files(non_snore_train, os.path.join(TRAIN_DIR, "mixing", "0"))
    copy_files(non_snore_val, os.path.join(VAL_DIR, "mixing", "0"))
    copy_files(non_snore_test, os.path.join(TEST_DIR, "mixing", "0"))

    # 8. Create noisy data (snore+non-snore) in 'mixing/noisy'
    create_noisy_files(snore_train, non_snore_train, TRAIN_DIR)
    create_noisy_files(snore_val, non_snore_val, VAL_DIR)
    create_noisy_files(snore_test, non_snore_test, TEST_DIR)

    print("Dataset preparation complete!\nCheck 'Dataset/Raw' for the results.")


if __name__ == "__main__":
    main()