#!/usr/bin/env python3

import os
import shutil
import numpy as np
import soundfile as sf
import librosa

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

# For mixing amplitude factors (only for non-snore audio)
MIXING_SCALES = [0.5, 1.0, 1.5]

# Desired sample rate
TARGET_SR = 16000

# For static data augmentation
# (Only used on non-snore in Train & Val)
# Adjust these as you see fit to produce 3 distinct transformations
def create_augmented_versions(audio, sr):
    """
    Return a list of 3 'static' augmented versions of the input audio.
    You can modify these to be any transformations you like,
    as long as they are deterministic (for "static" augmentation).
    """
    out_versions = []

    # Version 1: Slight pitch shift up by +2 semitones
    ver1 = librosa.effects.pitch_shift(audio, sr, n_steps=2)
    out_versions.append(ver1)

    # Version 2: Slight pitch shift down by -2 semitones
    ver2 = librosa.effects.pitch_shift(audio, sr, n_steps=-2)
    out_versions.append(ver2)

    # Version 3: Add mild Gaussian noise
    noise = np.random.RandomState(42).randn(len(audio)) * 0.01  # fixed seed => "static"
    ver3 = audio + noise
    out_versions.append(ver3)

    return out_versions

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
#              2) Preprocess "original" Folder (Downsample, Norm, Augment)
###############################################################################
def preprocess_original_audio():
    """
    - For each split (Train/Val/Test), each class (snore=1, non-snore=0),
      load each file from Raw/<split>/original/<class>.
    - Downsample & normalize => then optionally augment (if Train/Val & non-snore).
    - Save results to Preprocessed/<split>/original/<class>.
    """
    for split in SPLITS:
        raw_split_dir = os.path.join(RAW_DIR, split, "original")
        prepro_split_dir = os.path.join(PREPROCESSED_DIR, split, "original")

        for c in CLASS_LABELS:
            raw_class_dir = os.path.join(raw_split_dir, c)
            prepro_class_dir = os.path.join(prepro_split_dir, c)

            if not os.path.isdir(raw_class_dir):
                continue

            for fname in os.listdir(raw_class_dir):
                if not fname.lower().endswith(".wav"):
                    continue

                in_path = os.path.join(raw_class_dir, fname)
                base_name, ext = os.path.splitext(fname)  # e.g. ("0_123", ".wav")

                # Load & normalize
                audio, sr = load_and_normalize(in_path, TARGET_SR)

                # ----------------------------------------------------------------------------
                #  If it's non-snore (c="0") AND in Train or Val => create 3 augmented copies
                # ----------------------------------------------------------------------------
                if c == "0" and split in ["Train", "Val"]:
                    # Save the *original* version
                    out_original = os.path.join(prepro_class_dir, f"{base_name}.wav")
                    save_wav(out_original, audio, sr)

                    # Create 3 "static" augmented versions
                    augmented_list = create_augmented_versions(audio, sr)
                    for i, aug_audio in enumerate(augmented_list, start=1):
                        out_aug = os.path.join(prepro_class_dir,
                                               f"{base_name}_aug{i}.wav")
                        save_wav(out_aug, aug_audio, sr)

                else:
                    # c="1" (snore) in any split, or c="0" in Test => no augmentation, just save
                    out_original = os.path.join(prepro_class_dir, f"{base_name}.wav")
                    save_wav(out_original, audio, sr)


###############################################################################
#          3) Preprocess "real_mixing" Folder (Test Only, No Augmentation)
###############################################################################
def preprocess_real_mixing():
    """
    Copy all files from Raw/Test/real_mixing => Preprocessed/Test/real_mixing,
    but also do the usual downsample+normalize.  No augmentation.
    """
    raw_real_dir = os.path.join(RAW_DIR, "Test", "real_mixing")
    prepro_real_dir = os.path.join(PREPROCESSED_DIR, "Test", "real_mixing")

    if not os.path.isdir(raw_real_dir):
        return

    for fname in os.listdir(raw_real_dir):
        if not fname.lower().endswith(".wav"):
            continue
        in_path = os.path.join(raw_real_dir, fname)
        out_path = os.path.join(prepro_real_dir, fname)

        audio, sr = load_and_normalize(in_path, TARGET_SR)
        save_wav(out_path, audio, sr)


###############################################################################
#           4) Create the Mixed Audio (snore + scale * non_snore)
###############################################################################
def create_mixed_audio():
    """
    For each split (Train/Val/Test):
      - We look at Preprocessed/<split>/original/1 (snore) and
        Preprocessed/<split>/original/0 (non-snore).
      - For each snore file and each non-snore file (including augmented ones),
        create 3 versions in the subfolders mixing/noisy_0.5, mixing/noisy_1.0, mixing/noisy_1.5.
        The final name includes both snore and non-snore labels, e.g.:
            noisy_0.5_snore-1_005_non-0_123_aug2.wav
    - Note: If you want to limit the combinatorial explosion for Train, you could sample
      fewer non-snore files. But this code does the full cross product.
    """
    for split in SPLITS:
        # Directories
        snore_dir = os.path.join(PREPROCESSED_DIR, split, "original", "1")
        non_snore_dir = os.path.join(PREPROCESSED_DIR, split, "original", "0")
        mixing_dir = os.path.join(PREPROCESSED_DIR, split, "mixing")

        if not os.path.isdir(snore_dir) or not os.path.isdir(non_snore_dir):
            continue

        snore_files = [f for f in os.listdir(snore_dir) if f.lower().endswith(".wav")]
        non_snore_files = [f for f in os.listdir(non_snore_dir) if f.lower().endswith(".wav")]

        for snore_f in snore_files:
            snore_label = os.path.splitext(snore_f)[0]  # e.g. "1_005"
            snore_path = os.path.join(snore_dir, snore_f)
            snore_audio, _ = load_and_normalize(snore_path, TARGET_SR)

            for non_f in non_snore_files:
                non_label = os.path.splitext(non_f)[0]  # e.g. "0_123_aug2"
                non_path = os.path.join(non_snore_dir, non_f)
                non_audio, _ = load_and_normalize(non_path, TARGET_SR)

                # For each scale, create mixture
                for scale in MIXING_SCALES:
                    mixed = overlay_audio(snore_audio, non_audio, scale=scale)

                    # e.g. "noisy_0.5_snore-1_005_non-0_123_aug2.wav"
                    out_fname = f"noisy_{scale}_snore-{snore_label}_non-{non_label}.wav"
                    out_dir = os.path.join(mixing_dir, f"noisy_{scale}")
                    out_path = os.path.join(out_dir, out_fname)

                    save_wav(out_path, mixed, TARGET_SR)


###############################################################################
#                               MAIN
###############################################################################
def main():
    # 1) Create folder structure under Dataset/Preprocessed
    create_preprocessed_structure()

    # 2) Preprocess the "original" folders in Train/Val/Test
    #    - Downsample & normalize everything
    #    - For non-snore in Train/Val, produce 3 augmented copies
    preprocess_original_audio()

    # 3) Preprocess "real_mixing" folder in Test
    preprocess_real_mixing()

    # 4) Create the 3-scale mixing (0.5, 1.0, 1.5) for each snore + non-snore
    create_mixed_audio()

    print("All preprocessing steps complete!")
    print(f"Check '{PREPROCESSED_DIR}' for the results.")


if __name__ == "__main__":
    main()