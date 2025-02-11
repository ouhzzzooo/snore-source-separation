#!/usr/bin/env python3

import os
import subprocess

def get_latest_checkpoint(model_name, noise_level):
    """
    Looks under exps/noise_level_{noise_level}/{model_name}/
    for subfolders named like YYYY-MM-DD-HH-MM-SS,
    picks the lexically last one, and returns:
      exps/noise_level_X/model_name/<LATEST_TIMESTAMP>/weights_model_name.pth
    If none found, returns None.
    """
    base_dir = f"exps/noise_level_{noise_level}/{model_name}"
    if not os.path.isdir(base_dir):
        return None

    # List subfolders (e.g. 2025-02-07-12-00-00)
    subfolders = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            subfolders.append(entry)

    if not subfolders:
        return None

    # Sort lexicographically, pick the last => "latest"
    subfolders.sort()
    latest_folder = subfolders[-1]  # e.g. "2025-02-07-12-00-00"
    # checkpoint name
    ckpt = f"weights_{model_name}.pth"
    ckpt_path = os.path.join(base_dir, latest_folder, ckpt)

    if not os.path.isfile(ckpt_path):
        return None

    return ckpt_path


def main():
    # 1) Configure which models and noise levels you want
    MODEL_LIST = [
        "UNet1D",
        "AdvancedCNNAutoencoder",
        "AttentionUNet1D",
        "WaveUNet1D",
        "ResUNet1D"
    ]
    NOISE_LEVELS = ["0.5", "1.0", "1.5"]

    # We'll run both 'synthetic' and 'non_synthetic' modes
    MODES = ["synthetic", "non_synthetic"]

    # 2) Optionally configure device
    device = "cuda"

    for model_name in MODEL_LIST:
        for noise_level in NOISE_LEVELS:
            # Find the "latest" checkpoint
            ckpt_path = get_latest_checkpoint(model_name, noise_level)
            if ckpt_path is None:
                print(f"[WARNING] No checkpoint found for {model_name} / noise_level={noise_level}")
                continue

            print("\n=============================================================")
            print(f"[INFO] Found latest checkpoint => {ckpt_path}")
            print("=============================================================")

            # 3) For each mode, run inference.py
            for mode in MODES:
                cmd = [
                    "python", "inference.py",
                    "--model_name", model_name,
                    "--noise_level", noise_level,
                    "--ckpt_path", ckpt_path,
                    "--device", device,
                    "--mode", mode
                ]
                print("\n-------------------------------------------------------------")
                print(f"[RUN] {' '.join(cmd)}")
                print("-------------------------------------------------------------")
                subprocess.run(cmd, check=True)

    print("\n[ALL DONE] Inference complete for all models / noise levels!")


if __name__ == "__main__":
    main()