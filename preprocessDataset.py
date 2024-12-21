#!/usr/bin/env python3

import os
import random
import numpy as np
import librosa  # type: ignore
import soundfile as sf

# -------------------------------------------
# Configuration
# -------------------------------------------

DATASET_DIR       = "Dataset"
RAW_DIR           = os.path.join(DATASET_DIR, "Raw")
PREPROCESSED_DIR  = os.path.join(DATASET_DIR, "Preprocessed")

# Desired sample rate
TARGET_SR         = 16000

# Range for random amplitude scaling (SNR variation):
# We'll multiply the non-snore by a factor in [0.3, 1.5] to simulate different SNRs.
SCALE_MIN = 0.3
SCALE_MAX = 1.5

# Larger pitch shift range (in semitones) for more variety:
PITCH_SHIFT_RANGE = (-4, 4)

# Max time shift proportion
TIME_SHIFT_PROP   = 0.15  # up to 15% shift

# Noise amplitude factor
NOISE_FACTOR      = 0.02


# -------------------------------------------
# 1) Basic Downsampling & Normalizing
# -------------------------------------------

def load_and_preprocess(in_file, sr=TARGET_SR):
    """
    Load audio at sample rate sr, mono, then normalize to [-1,1].
    Return (audio_array, sr).
    """
    audio, _ = librosa.load(in_file, sr=sr, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 1e-8:
        audio /= max_val
    return audio, sr


def save_wav(out_file, audio, sr=TARGET_SR):
    """
    Normalize again just in case, then save as 16-bit WAV.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 1e-8:
        audio = audio / max_val
    sf.write(out_file, audio, sr, subtype='PCM_16')


# -------------------------------------------
# 2) Augmentations
# -------------------------------------------

def augment_time_shift(audio):
    """
    Randomly shift the audio by up to TIME_SHIFT_PROP of its length.
    """
    n_samples = len(audio)
    max_shift = int(n_samples * TIME_SHIFT_PROP)
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return audio
    if shift > 0:
        shifted = np.concatenate([audio[shift:], np.zeros(shift, dtype=audio.dtype)])
    else:
        shift = abs(shift)
        shifted = np.concatenate([np.zeros(shift, dtype=audio.dtype), audio[:-shift]])
    return shifted


def augment_pitch_shift(audio, sr):
    """
    Randomly pitch shift within PITCH_SHIFT_RANGE semitones.
    """
    semitones = random.uniform(*PITCH_SHIFT_RANGE)
    shifted = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=semitones)
    return shifted


def augment_add_noise(audio):
    """
    Add Gaussian noise at NOISE_FACTOR amplitude.
    """
    noise = np.random.randn(len(audio)) * NOISE_FACTOR
    return audio + noise


def scale_amplitude(audio):
    """
    Randomly scale amplitude by a factor in [SCALE_MIN, SCALE_MAX].
    """
    scale = random.uniform(SCALE_MIN, SCALE_MAX)
    return audio * scale


def random_augmentation_pipeline(audio, sr):
    """
    Apply a random combination of augmentations (time shift, pitch shift,
    add noise, and amplitude scaling) in random order.
    
    We'll pick from these augmentations with some probability,
    but you can do it in a more controlled manner if you wish.
    """
    # Start with random amplitude scaling (SNR variation)
    audio = scale_amplitude(audio)

    # We'll randomly pick how many augmentations to apply (1 to 3).
    num_augs = random.randint(1, 3)
    
    # Possible augmentations we can apply
    augment_ops = [augment_time_shift, augment_pitch_shift, augment_add_noise]
    random.shuffle(augment_ops)  # shuffle them for random order

    # Apply the first `num_augs` operations in the shuffled list
    for i in range(num_augs):
        aug_fn = augment_ops[i]
        if aug_fn == augment_pitch_shift:
            audio = aug_fn(audio, sr)
        else:
            audio = aug_fn(audio)

    return audio


def overlay_audio(snore_audio, non_snore_audio):
    """
    Overlay two signals by simple summation, normalizing afterwards.
    If lengths differ, trim to the min length.
    """
    length = min(len(snore_audio), len(non_snore_audio))
    mixture = snore_audio[:length] + non_snore_audio[:length]
    max_val = np.max(np.abs(mixture))
    if max_val > 1e-8:
        mixture /= max_val
    return mixture


# -------------------------------------------
# 3) Create Preprocessed Folder Structure
# -------------------------------------------

def create_preprocessed_structure():
    """
    Mirrors the structure of Dataset/Raw into Dataset/Preprocessed:
      - Train, Val, Test
        -> original (1, 0)
        -> mixing  (noisy, 0)
    """
    for split in ["Train", "Val", "Test"]:
        split_path = os.path.join(PREPROCESSED_DIR, split)

        orig_dir = os.path.join(split_path, "original")
        os.makedirs(os.path.join(orig_dir, "1"), exist_ok=True)
        os.makedirs(os.path.join(orig_dir, "0"), exist_ok=True)

        mix_dir = os.path.join(split_path, "mixing")
        os.makedirs(os.path.join(mix_dir, "noisy"), exist_ok=True)
        os.makedirs(os.path.join(mix_dir, "0"), exist_ok=True)


# -------------------------------------------
# 4) Phase 1: Basic Preprocessing
# -------------------------------------------

def preprocess_split_folder(split, subfolder, class_subfolder):
    """
    Downsample + normalize all audio in Raw/{split}/{subfolder}/{class_subfolder},
    save to Preprocessed/{split}/{subfolder}/{class_subfolder}.
    """
    raw_path = os.path.join(RAW_DIR, split, subfolder, class_subfolder)
    preproc_path = os.path.join(PREPROCESSED_DIR, split, subfolder, class_subfolder)
    os.makedirs(preproc_path, exist_ok=True)

    if not os.path.isdir(raw_path):
        return

    for fname in os.listdir(raw_path):
        if not fname.lower().endswith((".wav", ".mp3")):
            continue

        in_file = os.path.join(raw_path, fname)
        out_file = os.path.join(preproc_path, fname)

        audio, sr = load_and_preprocess(in_file, TARGET_SR)
        save_wav(out_file, audio, sr)


def phase1_basic_preprocessing():
    """
    Copies snore and non-snore from:
      - original/1
      - original/0
      - mixing/0
    ignoring mixing/noisy (we'll rebuild it).
    """
    for split in ["Train", "Val", "Test"]:
        # original/1 => snore
        preprocess_split_folder(split, "original", "1")
        # original/0 => non-snore
        preprocess_split_folder(split, "original", "0")
        # mixing/0 => non-snore
        preprocess_split_folder(split, "mixing", "0")
        # skip mixing/noisy in Raw


# -------------------------------------------
# 5) Phase 2: Rebuild Noisy with More Variety
# -------------------------------------------

def rebuild_noisy_data(num_versions=3):
    """
    For each snore file in Preprocessed/original/1,
    randomly pick a non-snore from Preprocessed/mixing/0.
    For each (snore, non_snore) pair, create `num_versions` new
    noisy files with different random augmentations on the non-snore.

    We then overlay snore + augmented non-snore and save as:
    e.g. "noisy_{snore_root}_{i}.wav"
    """
    for split in ["Train", "Val", "Test"]:
        snore_dir = os.path.join(PREPROCESSED_DIR, split, "original", "1")
        non_snore_dir = os.path.join(PREPROCESSED_DIR, split, "mixing", "0")
        noisy_dir = os.path.join(PREPROCESSED_DIR, split, "mixing", "noisy")

        if not (os.path.isdir(snore_dir) and os.path.isdir(non_snore_dir)):
            continue

        # Gather file lists
        snore_files = [f for f in os.listdir(snore_dir)
                       if f.lower().endswith((".wav", ".mp3"))]
        non_snore_files = [f for f in os.listdir(non_snore_dir)
                           if f.lower().endswith((".wav", ".mp3"))]
        if not snore_files or not non_snore_files:
            continue

        for snore_fname in snore_files:
            # Load the snore
            snore_path = os.path.join(snore_dir, snore_fname)
            snore_audio, sr_snore = load_and_preprocess(snore_path, TARGET_SR)
            base_snore_root, _ = os.path.splitext(snore_fname)

            # Generate multiple versions:
            for i in range(1, num_versions + 1):
                # PICK A DIFFERENT NON-SNORE EACH TIME (moved inside the loop)
                chosen_non_snore_fname = random.choice(non_snore_files)
                chosen_non_snore_path = os.path.join(non_snore_dir, chosen_non_snore_fname)
                non_snore_audio, sr_ns = load_and_preprocess(chosen_non_snore_path, TARGET_SR)

                # Make a copy & augment
                non_snore_aug = non_snore_audio.copy()
                non_snore_aug = random_augmentation_pipeline(non_snore_aug, sr_ns)

                # Mix
                mixture = overlay_audio(snore_audio, non_snore_aug)

                # Save
                out_fname = f"noisy_{base_snore_root}_{i}.wav"
                out_path = os.path.join(noisy_dir, out_fname)
                save_wav(out_path, mixture, TARGET_SR)


# -------------------------------------------
# Main
# -------------------------------------------
def main():
    # 1. Create the folder structure in "Preprocessed"
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    create_preprocessed_structure()

    # 2. Basic preprocessing (downsample + normalize) for snore & non-snore
    phase1_basic_preprocessing()

    # 3. Rebuild the "noisy" data with more variety
    #    Each snore file => pick random non-snore => 3 random augmentations.
    rebuild_noisy_data(num_versions=3)

    print("Preprocessing complete! Check Dataset/Preprocessed folder for results.")


if __name__ == "__main__":
    main()