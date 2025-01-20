#!/usr/bin/env python3

import os
import sys
import argparse
import glob
import csv
import shutil
import datetime
import numpy as np

import torch
import librosa
import soundfile as sf

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1) Model imports (same as in your trainer.py)
# ----------------------------------------------------------------------------
from src.models.UNet1D import UNet1D
from src.models.AdvancedCNNAutoencoder import AdvancedCNNAutoencoder
from src.models.AttentionUNet1D import AttentionUNet1D
from src.models.WaveUNet1D import WaveUNet1D
from src.models.ResUNet1D import ResUNet1D

def get_model(model_name: str):
    """Factory returning an instance of the requested model."""
    if model_name == 'UNet1D':
        return UNet1D()
    elif model_name == 'AdvancedCNNAutoencoder':
        return AdvancedCNNAutoencoder()
    elif model_name == 'AttentionUNet1D':
        return AttentionUNet1D()
    elif model_name == 'WaveUNet1D':
        return WaveUNet1D()
    elif model_name == 'ResUNet1D':
        return ResUNet1D()
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ----------------------------------------------------------------------------
# 2) SingleNoisyDataset: loads WAVs from e.g. "noisy_1.0_snore-1_005_non-0_123.wav"
# ----------------------------------------------------------------------------
class SingleNoisyDataset(Dataset):
    """
    Loads any .wav from the given directory. The naming filter is optional,
    or you can allow all .wav if you want. For demonstration, we load everything.
    Returns (raw_wave, filename) in the original amplitude domain (no normalization).
    """
    def __init__(self, noisy_dir, sr=16000):
        super().__init__()
        self.sr = sr
        # Collect all .wav files in that directory
        self.noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))

        print(f"[SingleNoisyDataset] Found {len(self.noisy_paths)} WAV files in '{noisy_dir}'.")

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        path = self.noisy_paths[idx]
        raw_wave, _ = librosa.load(path, sr=self.sr, mono=True)
        fname = os.path.basename(path)
        return raw_wave, fname

# ----------------------------------------------------------------------------
# 3) Cosine similarity utility
# ----------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    dot(a,b)/(||a||*||b||), trimming if lengths differ.
    """
    length = min(len(a), len(b))
    a = a[:length]
    b = b[:length]
    dot = np.sum(a * b)
    norm_a = np.sqrt(np.sum(a*a))
    norm_b = np.sqrt(np.sum(b*b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)

# ----------------------------------------------------------------------------
# 4) Parse the test filenames
#    e.g. "noisy_1.5_snore-1_005_non-0_123_aug2.wav"
# ----------------------------------------------------------------------------
def parse_test_filename(fname: str):
    """
    We assume format: "noisy_{scale}_snore-1_005_non-0_123_aug2.wav"
    Extract:
      snore_part = "1_005"
      nonSnore_part = "0_123_aug2"
    Return (snore_part, nonSnore_part).
    If not matching, return (None, None).
    """
    base = os.path.splitext(fname)[0]  # "noisy_1.5_snore-1_005_non-0_123_aug2"
    if "snore-" not in base or "_non-" not in base:
        return None, None

    # everything after "snore-"
    after_snore = base.split("snore-")[1]  # e.g. "1_005_non-0_123_aug2"
    # the snore ID portion is everything up to "_non-"
    snore_part = after_snore.split("_non-")[0]  # e.g. "1_005"
    # the non-snore portion is everything after "_non-"
    after_non = after_snore.split("_non-")[1]   # e.g. "0_123_aug2"
    nonSnore_part = after_non

    return snore_part, nonSnore_part

# ----------------------------------------------------------------------------
# 5) The main inference + evaluation routine
# ----------------------------------------------------------------------------
def infer_and_evaluate(model, device, test_dir, out_dir, sr=16000):
    """
    - test_dir: path to "Dataset/Preprocessed/Test/mixing/noisy_{0.5 or 1.0 or 1.5}"
    - out_dir: e.g. "Dataset/Preprocessed/Reconstructed/noise_level_1.0/ResUNet1D/"
    - We'll load raw snore from "Dataset/Raw/Test/original/1/{snore_part}.wav" to get amplitude.
      e.g. if snore_part="1_005", we find "Dataset/Raw/Test/original/1/1_005.wav"
    - Save "reconstructed_{snore_part}_{nonSnore_part}.wav"
    - measure optional cos sim with the snore wave (normalized).
    """

    os.makedirs(out_dir, exist_ok=True)
    dataset = SingleNoisyDataset(test_dir, sr=sr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # raw snore directory
    raw_snore_root = os.path.join("Dataset", "Raw", "Test", "original", "1")

    results = []
    model.eval()

    with torch.no_grad():
        for (raw_noisy_wave, noisy_fname) in tqdm(loader, desc=f"Inference on: {test_dir}"):
            if isinstance(raw_noisy_wave, torch.Tensor):
                raw_noisy_wave = raw_noisy_wave.detach().cpu().numpy()

            raw_noisy_wave = np.array(raw_noisy_wave, dtype=np.float32).reshape(-1)
            max_abs_noisy = np.max(np.abs(raw_noisy_wave)) if raw_noisy_wave.size > 0 else 0.0

            # Normalize input to [-1,1]
            if max_abs_noisy > 1e-9:
                normalized_noisy = raw_noisy_wave / max_abs_noisy
            else:
                normalized_noisy = raw_noisy_wave

            # shape => [1, 1, length]
            noisy_tensor = torch.tensor(
                normalized_noisy, dtype=torch.float32, device=device
            ).unsqueeze(0).unsqueeze(0)

            # Parse snore and non-snore IDs from filename
            snore_part, nonSnore_part = parse_test_filename(noisy_fname)
            if snore_part is None or nonSnore_part is None:
                # We skip if the filename format is unexpected
                continue

            # Forward pass
            recon_tensor = model(noisy_tensor)
            recon_np = recon_tensor.squeeze().cpu().numpy()  # normalized output

            # Load raw snore file => amplitude
            raw_snore_path = os.path.join(raw_snore_root, f"{snore_part}.wav")
            snore_amp = 0.0
            snore_wave = None
            if os.path.exists(raw_snore_path):
                snore_wave, _ = librosa.load(raw_snore_path, sr=sr, mono=True)
                if len(snore_wave) > 0:
                    snore_amp = np.max(np.abs(snore_wave))

            # Multiply model output by snore_amp
            raw_reconstructed = recon_np * snore_amp

            # Save reconstructed
            recon_fname = f"reconstructed_{snore_part}_{nonSnore_part}.wav"
            out_path = os.path.join(out_dir, recon_fname)
            sf.write(out_path, raw_reconstructed, sr)

            # (Optional) measure cos sim with normalized snore
            cos_sim = 0.0
            if snore_wave is not None and len(snore_wave) > 0:
                max_snore = np.max(np.abs(snore_wave))
                if max_snore > 1e-9:
                    snore_norm = snore_wave / max_snore
                    cos_sim = cosine_similarity(recon_np, snore_norm)

            # Store info for summary
            results.append({
                "noisy_file": noisy_fname,
                "reconstructed_file": recon_fname,
                "snore_file": f"{snore_part}.wav",
                "cosine_similarity": cos_sim,

                # Keep raw waveforms if you want to plot
                "raw_noisy_wave": raw_noisy_wave,
                "snore_wave": snore_wave,
                "reconstructed_wave": raw_reconstructed
            })

    return results

# ----------------------------------------------------------------------------
# 6) Plotting utility: top/bottom examples
# ----------------------------------------------------------------------------
def plot_waveforms(snore_wave, noisy_wave, recon_wave, cos_sim, out_file):
    """
    Basic multi-plot: top=snore, mid=noisy, bottom=reconstructed
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(f"CosSim = {cos_sim:.4f}")

    # Snore wave
    if snore_wave is None:
        snore_wave = np.zeros_like(noisy_wave)
    axes[0].plot(snore_wave, color='g')
    axes[0].set_title("Raw Snore")

    # Noisy
    axes[1].plot(noisy_wave, color='r')
    axes[1].set_title("Noisy")

    # Reconstructed
    axes[2].plot(recon_wave, color='b')
    axes[2].set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)


def compute_and_save_results(results, model_name, noise_level):
    """
    Sort by cos sim => top/bottom-5 => CSV => plot
    """
    if len(results) == 0:
        print("No results => skip.")
        return

    cos_sims = [r["cosine_similarity"] for r in results]
    avg_cos = sum(cos_sims)/len(cos_sims)
    max_cos = max(cos_sims)
    min_cos = min(cos_sims)

    best_rec = max(results, key=lambda x: x["cosine_similarity"])
    worst_rec= min(results, key=lambda x: x["cosine_similarity"])

    print(f"\n==== Cosine Similarity Summary ====")
    print(f"Count: {len(results)}")
    print(f"Average: {avg_cos:.4f}")
    print(f"Max: {max_cos:.4f} => {best_rec['reconstructed_file']}")
    print(f"Min: {min_cos:.4f} => {worst_rec['reconstructed_file']}")

    # Prepare output folders (CSV + plots)
    now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    csv_dir  = os.path.join("results", f"{model_name}", f"noise_level_{noise_level}", "csv")
    plot_dir = os.path.join("results", f"{model_name}", f"noise_level_{noise_level}", "plots_{now_str}")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Write CSV
    csv_path = os.path.join(csv_dir, f"{now_str}.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["noisy_file","reconstructed_file","snore_file","cosine_similarity"])
        for r in results:
            writer.writerow([
                r["noisy_file"],
                r["reconstructed_file"],
                r["snore_file"],
                f"{r['cosine_similarity']:.6f}"
            ])
        writer.writerow([])
        writer.writerow(["METRICS","COUNT","AVERAGE","MAX","MIN"])
        writer.writerow([
            "COS_SIM",
            len(results),
            f"{avg_cos:.6f}",
            f"{max_cos:.6f} ({best_rec['reconstructed_file']})",
            f"{min_cos:.6f} ({worst_rec['reconstructed_file']})"
        ])
    print(f"CSV results => {csv_path}")

    # Plot top5 & bottom5
    sorted_res = sorted(results, key=lambda x: x["cosine_similarity"])
    bottom_5 = sorted_res[:5]
    top_5    = sorted_res[-5:]

    def do_plot(r, prefix):
        out_file = os.path.join(plot_dir, f"{prefix}_{r['reconstructed_file'].replace('.wav','')}.png")
        plot_waveforms(
            r["snore_wave"],
            r["raw_noisy_wave"],
            r["reconstructed_wave"],
            r["cosine_similarity"],
            out_file
        )

    for r in top_5:
        do_plot(r, "best")
    for r in bottom_5:
        do_plot(r, "worst")

    print("Plotted top-5 & bottom-5 =>", plot_dir)


# ----------------------------------------------------------------------------
# 7) Main
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for snore denoising, scaled by raw snore amplitude."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Which model: UNet1D, ResUNet1D, etc.")
    parser.add_argument("--noise_level", type=str, default="1.0",
                        choices=["0.5","1.0","1.5"],
                        help="Which noise-level subfolder to use for inference.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the .pth checkpoint (weights).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load model and weights
    model = get_model(args.model_name)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2) We'll do inference on: "Dataset/Preprocessed/Test/mixing/noisy_{noise_level}"
    test_noisy_dir = os.path.join(
        "Dataset", "Preprocessed", "Test", "mixing",
        f"noisy_{args.noise_level}"
    )
    if not os.path.exists(test_noisy_dir):
        print(f"[ERROR] Not found: {test_noisy_dir}")
        sys.exit(1)

    # 3) Where to store reconstructed files? => "Dataset/Preprocessed/Reconstructed/noise_level_{noise_level}/{model_name}"
    recon_dir = os.path.join(
        "Dataset", "Preprocessed", "Reconstructed",
        f"noise_level_{args.noise_level}",
        args.model_name
    )
    if os.path.exists(recon_dir):
        print(f"Clearing out old reconstructed folder: {recon_dir}")
        shutil.rmtree(recon_dir)
    os.makedirs(recon_dir, exist_ok=True)

    # 4) Run inference
    results = infer_and_evaluate(
        model=model,
        device=device,
        test_dir=test_noisy_dir,
        out_dir=recon_dir,
        sr=16000
    )

    # 5) Summarize & Plot
    compute_and_save_results(results, args.model_name, args.noise_level)
    print(f"Inference complete! Reconstructed files in => {recon_dir}")

if __name__ == "__main__":
    main()