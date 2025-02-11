#!/usr/bin/env python3

import os
import shutil
import numpy as np
import soundfile as sf
import librosa
import sys

###############################################################################
#                          Configuration & Globals
###############################################################################
DATASET_DIR       = "Dataset"
RAW_DIR           = os.path.join(DATASET_DIR, "Raw")
PREPROCESSED_DIR  = os.path.join(DATASET_DIR, "Preprocessed")

# We assume these splits exist under Dataset/Raw
SPLITS = ["Train", "Val", "Test"]

# Classes: "1" = snore, "0" = non-snore
CLASS_LABELS = ["1", "0"]

# For mixing amplitude factors
MIXING_SCALES = [0.5, 1.0, 1.5]

# Desired sample rate
TARGET_SR = 16000

###############################################################################
#                           Basic Audio Utilities
###############################################################################
def load_and_normalize(file_path, sr=TARGET_SR):
    """
    Load the audio as mono, at sample rate `sr`.
    Normalize to [-1, +1].
    Returns (audio_array, sr).
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    max_abs = np.max(np.abs(audio))
    if max_abs > 1e-8:
        audio = audio / max_abs
    return audio, sr


def save_wav(file_path, audio, sr=TARGET_SR):
    """
    Final normalization (just to be safe) and save as 16-bit WAV.
    """
    max_abs = np.max(np.abs(audio))
    if max_abs > 1e-8:
        audio = audio / max_abs
    sf.write(file_path, audio, sr, subtype='PCM_16')


def overlay_audio(snore_audio, non_snore_audio, scale=1.0):
    """
    Mix snore_audio + (scale * non_snore_audio).
    Trim to the minimum length if they differ.
    Normalize the result to [-1,1].
    """
    length = min(len(snore_audio), len(non_snore_audio))
    mixture = snore_audio[:length] + scale * non_snore_audio[:length]
    max_abs = np.max(np.abs(mixture))
    if max_abs > 1e-8:
        mixture /= max_abs
    return mixture


###############################################################################
#              1) Create Preprocessed Folder Structure
###############################################################################
def create_preprocessed_structure():
    """
    Under Dataset/Preprocessed:
      For each split (Train, Val, Test):
        - original/1
        - original/0
        - mixing/noisy_0.5
        - mixing/noisy_1.0
        - mixing/noisy_1.5
      And only Test has 'real_mixing'.
    """
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    for split in SPLITS:
        split_dir = os.path.join(PREPROCESSED_DIR, split)

        # original/<class>
        orig_dir = os.path.join(split_dir, "original")
        for c in CLASS_LABELS:
            os.makedirs(os.path.join(orig_dir, c), exist_ok=True)

        # mixing/noisy_<scale>
        mixing_dir = os.path.join(split_dir, "mixing")
        for scale in MIXING_SCALES:
            scale_dir = f"noisy_{scale}"
            os.makedirs(os.path.join(mixing_dir, scale_dir), exist_ok=True)

        # Only Test has 'real_mixing'
        if split == "Test":
            os.makedirs(os.path.join(split_dir, "real_mixing"), exist_ok=True)


###############################################################################
#              2) Preprocess "original" Folder (Downsample & Normalize)
###############################################################################
def preprocess_original_audio():
    """
    - For each split (Train/Val/Test), each class (snore=1, non-snore=0),
      load each file from Raw/<split>/original/<class>.
    - Downsample & normalize => no augmentation.
    - Save results to Preprocessed/<split>/original/<class>.
    """
    print("[INFO] Starting to preprocess 'original' folders...")
    for split in SPLITS:
        raw_split_dir = os.path.join(RAW_DIR, split, "original")
        prepro_split_dir = os.path.join(PREPROCESSED_DIR, split, "original")

        for c in CLASS_LABELS:
            raw_class_dir = os.path.join(raw_split_dir, c)
            prepro_class_dir = os.path.join(prepro_split_dir, c)

            if not os.path.isdir(raw_class_dir):
                continue

            file_list = [
                f for f in os.listdir(raw_class_dir)
                if f.lower().endswith(".wav")
            ]
            print(f"  -> Split: {split}, Class: {c}, Found {len(file_list)} files")

            for idx, fname in enumerate(file_list, start=1):
                in_path = os.path.join(raw_class_dir, fname)
                base_name, _ = os.path.splitext(fname)

                # Load & normalize
                audio, sr = load_and_normalize(in_path, TARGET_SR)

                # Just save it back (no augmentation)
                out_path = os.path.join(prepro_class_dir, f"{base_name}.wav")
                save_wav(out_path, audio, sr)

                if idx % 100 == 0:
                    print(f"    Processed {idx}/{len(file_list)} for {split}/{c}...", flush=True)

    print("[INFO] Finished preprocessing 'original' folders.\n")


###############################################################################
#          3) Preprocess "real_mixing" Folder (Test Only, No Augmentation)
###############################################################################
def preprocess_real_mixing():
    """
    Copy all files from Raw/Test/real_mixing => Preprocessed/Test/real_mixing,
    downsample+normalize as usual. No augmentation.
    """
    print("[INFO] Preprocessing 'real_mixing' folder (Test only)...")
    raw_real_dir = os.path.join(RAW_DIR, "Test", "real_mixing")
    prepro_real_dir = os.path.join(PREPROCESSED_DIR, "Test", "real_mixing")

    if not os.path.isdir(raw_real_dir):
        print("   -> No real_mixing folder found for Test.")
        return

    file_list = [
        f for f in os.listdir(raw_real_dir) if f.lower().endswith(".wav")
    ]
    print(f"   -> Found {len(file_list)} files in real_mixing")

    for idx, fname in enumerate(file_list, start=1):
        in_path = os.path.join(raw_real_dir, fname)
        out_path = os.path.join(prepro_real_dir, fname)

        audio, sr = load_and_normalize(in_path, TARGET_SR)
        save_wav(out_path, audio, sr)

        if idx % 100 == 0:
            print(f"     Processed {idx}/{len(file_list)} real_mixing files...", flush=True)

    print("[INFO] Finished preprocessing 'real_mixing' for Test.\n")


###############################################################################
#           4) Create the Mixed Audio (snore + scale * non_snore)
###############################################################################
def create_mixed_audio():
    """
    For each split (Train/Val/Test):
      - We look at Preprocessed/<split>/original/1 (snore) and
        Preprocessed/<split>/original/0 (non-snore).
      - For each snore file and each non-snore file, create
        3 versions in subfolders mixing/noisy_0.5, mixing/noisy_1.0, mixing/noisy_1.5.
    - The final name includes something like:
         noisy_0.5_snore-<snore_label>_non-<non_label>.wav
    """
    print("[INFO] Creating mixed audio (snore + scale * non_snore)...")

    for split in SPLITS:
        snore_dir = os.path.join(PREPROCESSED_DIR, split, "original", "1")
        non_snore_dir = os.path.join(PREPROCESSED_DIR, split, "original", "0")
        mixing_dir = os.path.join(PREPROCESSED_DIR, split, "mixing")

        if not (os.path.isdir(snore_dir) and os.path.isdir(non_snore_dir)):
            print(f"  -> Skipping {split} (no snore_dir or non_snore_dir).")
            continue

        # List .wav files
        snore_files = [
            f for f in os.listdir(snore_dir)
            if f.lower().endswith(".wav")
        ]
        non_snore_files = [
            f for f in os.listdir(non_snore_dir)
            if f.lower().endswith(".wav")
        ]
        print(f"  -> Split: {split}, #snore={len(snore_files)}, #non-snore={len(non_snore_files)}")

        # ------------------------------------------------------------------
        # A) Cache the audio in memory to avoid reloading for each mix
        # ------------------------------------------------------------------
        snore_cache = {}
        for s_f in snore_files:
            s_path = os.path.join(snore_dir, s_f)
            s_audio, _ = load_and_normalize(s_path, TARGET_SR)
            snore_cache[s_f] = s_audio

        non_snore_cache = {}
        for ns_f in non_snore_files:
            ns_path = os.path.join(non_snore_dir, ns_f)
            ns_audio, _ = load_and_normalize(ns_path, TARGET_SR)
            non_snore_cache[ns_f] = ns_audio

        # ------------------------------------------------------------------
        # B) Generate mixtures
        # ------------------------------------------------------------------
        total_combinations = len(snore_files) * len(non_snore_files) * len(MIXING_SCALES)
        count = 0

        for s_f in snore_files:
            snore_label = os.path.splitext(s_f)[0]
            snore_audio = snore_cache[s_f]

            for ns_f in non_snore_files:
                non_label = os.path.splitext(ns_f)[0]
                non_audio = non_snore_cache[ns_f]

                for scale in MIXING_SCALES:
                    mixed = overlay_audio(snore_audio, non_audio, scale=scale)
                    out_fname = f"noisy_{scale}_snore-{snore_label}_non-{non_label}.wav"
                    out_dir = os.path.join(mixing_dir, f"noisy_{scale}")
                    out_path = os.path.join(out_dir, out_fname)

                    save_wav(out_path, mixed, TARGET_SR)

                    count += 1
                    if count % 1000 == 0:
                        print(f"    [Progress] {count}/{total_combinations} mixtures for {split}...", flush=True)

        print(f"  -> Done creating mixtures for {split}: {count} total.\n")

    print("[INFO] Finished creating mixed audio.\n")


###############################################################################
#                               MAIN
###############################################################################
def main():
    # 1) Create folder structure under Dataset/Preprocessed
    create_preprocessed_structure()

    # 2) Preprocess the "original" folders (Train/Val/Test)
    preprocess_original_audio()

    # 3) Preprocess "real_mixing" folder in Test (if present)
    preprocess_real_mixing()

    # 4) Create the 3-scale mixing (0.5, 1.0, 1.5) for each snore+non-snore
    create_mixed_audio()

    print("All preprocessing steps complete!")
    print(f"Check '{PREPROCESSED_DIR}' for the results.")


if __name__ == "__main__":
    main()