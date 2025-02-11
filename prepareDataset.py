#!/usr/bin/env python3

import os
import random
import shutil
import json
from pydub import AudioSegment

# ------------------------------------------
# Configuration
# ------------------------------------------

# Folders in your current dataset
SNORE_DIR     = "Snoring Dataset/1"   # snoring sounds (both clean & bg_noise)
NON_SNORE_DIR = "Snoring Dataset/0"   # non-snoring sounds

# Path to your JSON file (containing bg_noise/clean lists)
FILELIST_JSON = "filelist.json"

# Target folder structure we want to create
DATASET_DIR = "Dataset"
RAW_DIR     = os.path.join(DATASET_DIR, "Raw")

TRAIN_DIR = os.path.join(RAW_DIR, "Train")
VAL_DIR   = os.path.join(RAW_DIR, "Val")
TEST_DIR  = os.path.join(RAW_DIR, "Test")

# Split ratios (0.6 / 0.2 / 0.2)
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2
TEST_RATIO  = 0.2

# Check sum to 1
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, \
    "Train/Val/Test split ratios must sum to 1.0"

# ------------------------------------------
# Helper Functions
# ------------------------------------------

def create_folder_structure():
    """
    Creates the folder structure:
    
    Dataset/Raw/
        ├─ Train
        │   ├─ original
        │   │   ├─ 1
        │   │   └─ 0
        │   └─ mixing
        │       └─ noisy
        ├─ Val
        │   ├─ original
        │   │   ├─ 1
        │   │   └─ 0
        │   └─ mixing
        │       └─ noisy
        └─ Test
            ├─ original
            │   ├─ 1
            │   └─ 0
            ├─ mixing
            │   └─ noisy
            └─ real_mixing
    """
    # Train, Val, Test folders
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(split_dir, exist_ok=True)
        
        # original
        original_dir = os.path.join(split_dir, "original")
        os.makedirs(os.path.join(original_dir, "1"), exist_ok=True)
        os.makedirs(os.path.join(original_dir, "0"), exist_ok=True)

        # mixing
        mixing_dir = os.path.join(split_dir, "mixing")
        os.makedirs(os.path.join(mixing_dir, "noisy"), exist_ok=True)

    # Only the Test folder needs 'real_mixing'
    os.makedirs(os.path.join(TEST_DIR, "real_mixing"), exist_ok=True)


def get_audio_file_paths(folder_path):
    """
    Returns a list of audio file paths (WAV, etc.)
    from a given folder_path.
    (Not used for snore if we rely on filelist.json, 
     but still used for non-snore)
    """
    if not os.path.isdir(folder_path):
        return []
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]


def split_data(file_list, train_ratio, val_ratio, test_ratio):
    """
    Shuffle and split file_list into train/val/test sets.
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
    Overlay snore_file with non_snore_file using pydub and save to output_path.
    """
    snore     = AudioSegment.from_file(snore_file)
    non_snore = AudioSegment.from_file(non_snore_file)
    mixed     = snore.overlay(non_snore)
    mixed.export(output_path, format="wav")


def create_noisy_files(snore_list, non_snore_list, split_path):
    """
    For each snore file in snore_list, pick a random non-snore file
    and create a 'noisy_{snore_name_wo_ext}_{non_snore_name_wo_ext}.wav'
    in 'mixing/noisy'.
    """
    noisy_dir = os.path.join(split_path, "mixing", "noisy")
    os.makedirs(noisy_dir, exist_ok=True)

    if not snore_list:
        print("Warning: No snore files found!")
        return
    if not non_snore_list:
        print("Warning: No non-snore files found!")
        return

    for s_file in snore_list:
        chosen_non_snore = random.choice(non_snore_list)

        snore_filename     = os.path.splitext(os.path.basename(s_file))[0]
        non_snore_filename = os.path.splitext(os.path.basename(chosen_non_snore))[0]

        out_filename = f"noisy_{snore_filename}_{non_snore_filename}.wav"
        output_path  = os.path.join(noisy_dir, out_filename)

        mix_audio_files(s_file, chosen_non_snore, output_path)


# ------------------------------------------
# Main logic
# ------------------------------------------
def main():
    # 1. Create the top-level 'Dataset' folder
    os.makedirs(DATASET_DIR, exist_ok=True)

    # 2. Create the subfolder structure under Dataset/Raw
    create_folder_structure()

    # 3. Parse filelist.json to get the "bg_noise" vs "clean" from SNORE_DIR
    if not os.path.isfile(FILELIST_JSON):
        raise FileNotFoundError(f"JSON file '{FILELIST_JSON}' not found.")

    with open(FILELIST_JSON, "r") as f:
        json_data = json.load(f)

    # Ensure the keys 'bg_noise' and 'clean' exist
    bg_noise_fnames = json_data.get("bg_noise", [])
    clean_fnames    = json_data.get("clean", [])

    # Convert filenames to full paths
    bg_noise_files   = [os.path.join(SNORE_DIR, fn) for fn in bg_noise_fnames]
    clean_snore_files = [os.path.join(SNORE_DIR, fn) for fn in clean_fnames]

    # 4. Get all non-snore file paths
    non_snore_files = get_audio_file_paths(NON_SNORE_DIR)

    # 5. Move all bg_noise files to 'real_mixing' in Test (no splitting)
    #    We will do this at the end in step #9

    # 6. Split the clean snore files into Train/Val/Test
    snore_train, snore_val, snore_test = split_data(
        clean_snore_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    # 7. Split the non-snore files into Train/Val/Test
    non_snore_train, non_snore_val, non_snore_test = split_data(
        non_snore_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    # 8. Copy the clean snore files to their respective 'original/1' folders
    copy_files(snore_train, os.path.join(TRAIN_DIR, "original", "1"))
    copy_files(snore_val,   os.path.join(VAL_DIR,   "original", "1"))
    copy_files(snore_test,  os.path.join(TEST_DIR,  "original", "1"))

    # 9. Copy the non-snore files to 'original/0'
    copy_files(non_snore_train, os.path.join(TRAIN_DIR, "original", "0"))
    copy_files(non_snore_val,   os.path.join(VAL_DIR,   "original", "0"))
    copy_files(non_snore_test,  os.path.join(TEST_DIR,  "original", "0"))

    # 10. Create the noisy (mixed) files for Train, Val, Test using the splitted data
    create_noisy_files(snore_train, non_snore_train, TRAIN_DIR)
    create_noisy_files(snore_val,   non_snore_val,   VAL_DIR)
    create_noisy_files(snore_test,  non_snore_test,  TEST_DIR)

    # 11. Finally, copy all bg_noise (137 files) to Test/real_mixing
    copy_files(bg_noise_files, os.path.join(TEST_DIR, "real_mixing"))

    print("Dataset preparation complete!")
    print(f"Check '{DATASET_DIR}/Raw' for the results.")


if __name__ == "__main__":
    main()