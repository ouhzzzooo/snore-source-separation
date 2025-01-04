#!/usr/bin/env python3

import os
import sys
import argparse
import glob
import csv
import time
import datetime
import numpy as np
import torch
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1) Model imports or a factory function that provides your models
# ----------------------------------------------------------------------------
from src.models.UNet1D import UNet1D
from src.models.AdvancedCNNAutoencoder import AdvancedCNNAutoencoder
from src.models.AttentionUNet1D import AttentionUNet1D
from src.models.WaveUNet1D import WaveUNet1D
from src.models.ResUNet1D import ResUNet1D

def get_model(model_name: str):
    """Factory function returning an instance of the requested model."""
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
        raise ValueError(f"Unknown model name: {model_name}")

# ----------------------------------------------------------------------------
# 2) A Dataset that picks only files ending with "_1.wav"
#    e.g. "noisy_1_195_1.wav", so we produce ONE reconstruction per snore
# ----------------------------------------------------------------------------
class SingleNoisyDataset(Dataset):
    """
    Loads .wav files from a directory, but only those ending with '_1.wav'.
    We'll reconstruct these and compare to 'original/1/1_<middle>.wav'.
    Potentially parse the non-snore if we had a stable naming. 
    """
    def __init__(self, noisy_dir):
        super().__init__()
        self.noisy_paths = []
        all_wavs = glob.glob(os.path.join(noisy_dir, "*.wav"))
        for path in all_wavs:
            fname = os.path.basename(path)
            if fname.endswith("_1.wav"):
                self.noisy_paths.append(path)

        print(f"[SingleNoisyDataset] Found {len(all_wavs)} total in '{noisy_dir}'")
        print(f"                   Using {len(self.noisy_paths)} that match '*_1.wav'")

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        path = self.noisy_paths[idx]
        wav, _ = librosa.load(path, sr=16000, mono=True)
        maxval = np.max(np.abs(wav)) if len(wav) else 0
        if maxval > 0:
            wav = wav / maxval  # normalize to [-1,1]
        fname = os.path.basename(path)
        # Return the wave + base filename
        return torch.tensor(wav, dtype=torch.float32), fname

# ----------------------------------------------------------------------------
# 3) Cosine Similarity function for waveforms
# ----------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    Cosine similarity = dot(a,b) / (||a|| * ||b||).
    We'll pad or trim if lengths differ.
    """
    length = min(len(a), len(b))
    a = a[:length]
    b = b[:length]

    dot = np.sum(a * b)
    norm_a = np.sqrt(np.sum(a * a))
    norm_b = np.sqrt(np.sum(b * b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)

# ----------------------------------------------------------------------------
# 4) Inference & Cosine Similarity
# ----------------------------------------------------------------------------
def infer_and_evaluate(model, device, test_dir, out_dir, original_dir):
    """
    - test_dir: e.g. "Dataset/Preprocessed/Test/mixing/noisy"
    - out_dir:  e.g. "Dataset/Preprocessed/Test/denoising"
    - original_dir: "Dataset/Preprocessed/Test/original/1"

    Returns: list of dict with fields:
      {
        "noisy_file": ...,
        "reconstructed_file": ...,
        "original_file": ...,
        "cosine_similarity": float
      }
    """
    os.makedirs(out_dir, exist_ok=True)
    dataset = SingleNoisyDataset(test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    results = []
    model.eval()

    with torch.no_grad():
        for noisy_wav, noisy_fname in tqdm(loader, desc=f"Inference on {test_dir}"):
            # shape: [1, waveform_length]
            noisy_wav = noisy_wav.to(device)
            # model expects [batch, channels, length] => [1,1,length]
            noisy_wav = noisy_wav.unsqueeze(1)

            # e.g. "noisy_1_195_1.wav" => parse "195"
            name_parts = os.path.splitext(noisy_fname[0])[0].split("_")
            if len(name_parts) < 4:
                continue
            middle_id = name_parts[2]  # "195"

            reconstructed_fname = f"reconstructed_{middle_id}.wav"
            out_path = os.path.join(out_dir, reconstructed_fname)

            # Forward pass
            reconstructed = model(noisy_wav)
            reconstructed_np = reconstructed.squeeze().cpu().numpy()

            # Re-normalize
            max_val = np.max(np.abs(reconstructed_np))
            if max_val > 0:
                reconstructed_np = reconstructed_np / max_val

            sf.write(out_path, reconstructed_np, 16000)

            # Compare with e.g. "1_{middle_id}.wav"
            original_snore_fname = f"1_{middle_id}.wav"
            original_snore_path = os.path.join(original_dir, original_snore_fname)
            if not os.path.exists(original_snore_path):
                cos_sim = 0.0
            else:
                orig_wav, _ = librosa.load(original_snore_path, sr=16000, mono=True)
                mv2 = np.max(np.abs(orig_wav)) if len(orig_wav) else 0
                if mv2 > 0:
                    orig_wav = orig_wav / mv2
                cos_sim = cosine_similarity(reconstructed_np, orig_wav)

            results.append({
                "noisy_file": noisy_fname[0],
                "reconstructed_file": reconstructed_fname,
                "original_file": original_snore_fname,
                "cosine_similarity": cos_sim,
                # We'll store these waveforms in memory for potential plotting
                "noisy_wave": noisy_wav.squeeze().cpu().numpy(),
                "reconstructed_wave": reconstructed_np,
                "original_snore_wave": orig_wav if os.path.exists(original_snore_path) else None
            })
    return results

# ----------------------------------------------------------------------------
# 5) Plotting Function for top/bottom 5
# ----------------------------------------------------------------------------

def plot_waveforms(original_ns, original_snore, noisy_wave, reconstructed_wave,
                   title_str, out_file):
    """
    Create a 4-subplot figure:
      1) original non-snoring (if we have it, else zeros)
      2) original snoring
      3) noisy
      4) reconstructed
    Save to out_file (PNG or JPG).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(title_str)

    # 1) original non-snoring
    if original_ns is None:
        # fallback to zeros
        original_ns = np.zeros_like(noisy_wave)  
    axes[0].plot(original_ns, color='gray')
    axes[0].set_title("Original Non-Snore")

    # 2) original snoring
    if original_snore is None:
        original_snore = np.zeros_like(noisy_wave)
    axes[1].plot(original_snore, color='g')
    axes[1].set_title("Original Snore")

    # 3) noisy
    axes[2].plot(noisy_wave, color='r')
    axes[2].set_title("Noisy")

    # 4) reconstructed
    axes[3].plot(reconstructed_wave, color='b')
    axes[3].set_title("Reconstructed")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_file)
    plt.close(fig)


def attempt_find_nonsnore(file_name):
    """
    If you truly want the original non-snore used in the mixture, you must have
    stored that info. We'll do a naive approach: none.
    Return None.
    """
    return None


# ----------------------------------------------------------------------------
# 6) Summarize + Save CSV + Plot
# ----------------------------------------------------------------------------
def compute_and_save_results(results, model_name):
    """
    1) Print average, min, max
    2) Save a CSV into results/{model_name}/Preprocessed/csv/<timestamp>.csv
    3) Create top-5 and bottom-5 wave plots => results/{model_name}/Preprocessed/plot/<timestamp>/...
    """
    if len(results) == 0:
        print("\nNo results => skipping save/plots.")
        return

    cos_sims = [r["cosine_similarity"] for r in results]
    avg_cos = sum(cos_sims) / len(cos_sims)
    max_cos = max(cos_sims)
    min_cos = min(cos_sims)

    max_record = max(results, key=lambda x: x["cosine_similarity"])
    min_record = min(results, key=lambda x: x["cosine_similarity"])

    print(f"\n===== Cosine Similarity Summary =====")
    print(f"Count: {len(cos_sims)}")
    print(f"Average: {avg_cos:.4f}")
    print(f"Max: {max_cos:.4f} (file={max_record['reconstructed_file']})")
    print(f"Min: {min_cos:.4f} (file={min_record['reconstructed_file']})")

    # Create subfolders with date/time
    now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    csv_dir = os.path.join("results", model_name, "Preprocessed", "csv")
    plot_dir = os.path.join("results", model_name, "Preprocessed", "plot", now_str)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 1) Save CSV
    csv_path = os.path.join(csv_dir, f"{now_str}.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["noisy_file","reconstructed_file","original_file","cosine_similarity"])
        for r in results:
            writer.writerow([
                r["noisy_file"],
                r["reconstructed_file"],
                r["original_file"],
                f"{r['cosine_similarity']:.6f}"
            ])
        # summary row
        writer.writerow([])
        writer.writerow(["METRICS","COUNT","AVERAGE","MAX","MIN"])
        writer.writerow([
            "COS_SIM",
            len(cos_sims),
            f"{avg_cos:.6f}",
            f"{max_cos:.6f} ({max_record['reconstructed_file']})",
            f"{min_cos:.6f} ({min_record['reconstructed_file']})"
        ])
    print(f"Saved CSV => {csv_path}")

    # 2) Plot Top-5 & Bottom-5
    sorted_results = sorted(results, key=lambda x: x["cosine_similarity"])
    bottom_5 = sorted_results[:5]
    top_5 = sorted_results[-5:]

    # For each example, we plot 4 waveforms: original non-snore, snore, noisy, reconstructed
    # We'll skip actual non-snore unless we have logic to parse it.

    def plot_example(r, prefix):
        # attempt to find the original non-snoring wave
        # (We do not have a real mapping in your code, but let's call a helper)
        original_ns = attempt_find_nonsnore(r["noisy_file"])

        original_snore = r["original_snore_wave"]  # might be None
        noisy_wave = r["noisy_wave"]
        reconstructed_wave = r["reconstructed_wave"]
        sim_val = r["cosine_similarity"]

        out_file = os.path.join(plot_dir, f"{prefix}_{r['reconstructed_file'].replace('.wav','')}.png")
        title_str = f"{prefix.upper()} cos_sim={sim_val:.4f}"
        plot_waveforms(original_ns, original_snore, noisy_wave, reconstructed_wave,
                       title_str, out_file)

    print(f"Creating plots in: {plot_dir}")

    for r in top_5:
        plot_example(r, prefix="best")

    for r in bottom_5:
        plot_example(r, prefix="worst")

    print("Plots created for top 5 and bottom 5 similarity.")

# ----------------------------------------------------------------------------
# 7) Main
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Inference for snore denoising, with plotting.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="UNet1D, AdvancedCNNAutoencoder, etc.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load model
    model = get_model(args.model_name)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # 2) Do inference
    pre_noisy_dir = "Dataset/Preprocessed/Test/mixing/noisy"
    pre_denoise_dir = "Dataset/Preprocessed/Test/denoising"
    pre_original_dir = "Dataset/Preprocessed/Test/original/1"

    if not os.path.exists(pre_noisy_dir):
        print("[WARNING] No test dir found:", pre_noisy_dir)
        return

    results = infer_and_evaluate(model, device, pre_noisy_dir, pre_denoise_dir, pre_original_dir)

    # 3) Summarize + Save + Plot
    compute_and_save_results(results, args.model_name)
    print("Inference complete!")

if __name__ == "__main__":
    main()