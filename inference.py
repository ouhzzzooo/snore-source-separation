#!/usr/bin/env python3

import os
import sys
import argparse
import glob
import csv
import datetime
import numpy as np
import torch
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1) Model imports (same factory approach)
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
# 2) Dataset for test files named: "noisy_1_{snoreID}_0_{nonSnoreID}.wav"
#    but we store the raw wave (unscaled) + do the normalization separately.
# ----------------------------------------------------------------------------
class SingleNoisyDataset(Dataset):
    """
    Loads .wav files from 'noisy_1_*_0_*.wav' in 'noisy_dir'.
    Returns a tuple: (raw_noisy_wave, noisy_fname)
    No normalization is done here, so we keep original amplitude for plotting.
    """
    def __init__(self, noisy_dir, sr=16000):
        super().__init__()
        self.sr = sr
        self.noisy_paths = []
        all_wavs = glob.glob(os.path.join(noisy_dir, "*.wav"))
        for p in all_wavs:
            fname = os.path.basename(p)
            if fname.lower().startswith("noisy_1_") and "_0_" in fname:
                self.noisy_paths.append(p)

        print(f"[SingleNoisyDataset] Found {len(all_wavs)} .wav in '{noisy_dir}'")
        print(f"                   Using {len(self.noisy_paths)} that match 'noisy_1_*_0_*.wav'")

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        path = self.noisy_paths[idx]
        # Load raw wave (no normalization)
        raw_wave, _ = librosa.load(path, sr=self.sr, mono=True)
        fname = os.path.basename(path)
        return raw_wave, fname


# ----------------------------------------------------------------------------
# 3) Cosine Similarity
# ----------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray):
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
# 4) Parsing "noisy_1_{snoreID}_0_{nonSnoreID}.wav"
# ----------------------------------------------------------------------------
def parse_test_filename(fname: str):
    """
    e.g. "noisy_1_94_0_33.wav" => snoreID=94, nonSnoreID=33
    Return (snoreID, nonSnoreID).
    If parse fails => (None, None).
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    # e.g. parts=['noisy','1','94','0','33']
    if len(parts) < 5:
        return None, None
    snoreID = parts[2]   # "94"
    nonSnoreID = parts[4]# "33"
    return snoreID, nonSnoreID


# ----------------------------------------------------------------------------
# 5) Inference + Evaluate
# ----------------------------------------------------------------------------
def infer_and_evaluate(model, device, test_dir, out_dir, snore_dir, nonsnore_dir, sr=16000):
    """
    - test_dir: e.g. 'Dataset/Preprocessed/Test/mixing/noisy'
      has files "noisy_1_{snoreID}_0_{nonSnoreID}.wav" in original amplitude.
    - out_dir: e.g. 'Dataset/Preprocessed/Test/denoising'
    - snore_dir: e.g. 'Dataset/Preprocessed/Test/original/1'
    - nonsnore_dir: e.g. 'Dataset/Preprocessed/Test/original/0'

    Steps:
      1) Load raw noisy => (raw_noisy, no normalization)
      2) For model input => normalized_noisy = raw_noisy / max_abs_noisy
      3) Model => normalized_recon
      4) Re-scale => raw_reconstructed = normalized_recon * max_abs_noisy
      5) Load raw snore => snore
      6) Load raw non-snore => ns
      7) Cos sim => compare normalized_recon vs normalized_snore
         or raw vs raw, but typically we do it in normalized domain.
    """
    os.makedirs(out_dir, exist_ok=True)
    dataset = SingleNoisyDataset(test_dir, sr=sr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    results = []
    model.eval()

    with torch.no_grad():
        for (raw_noisy_wave, noisy_fname) in tqdm(loader, desc=f"Inference on {test_dir}"):
            raw_noisy_wave = raw_noisy_wave.numpy().squeeze()  # shape [length,]
            # The max abs for scaling in/out the model
            max_abs_noisy = np.max(np.abs(raw_noisy_wave)) if len(raw_noisy_wave) else 0.0

            # Prepare model input
            if max_abs_noisy > 1e-9:
                normalized_noisy = raw_noisy_wave / max_abs_noisy
            else:
                normalized_noisy = raw_noisy_wave  # all zeros?

            # Convert to torch [batch=1, channels=1, length]
            noisy_tensor = torch.tensor(normalized_noisy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            # parse snore & non-snore IDs
            snoreID, nonSnoreID = parse_test_filename(noisy_fname[0])
            if snoreID is None or nonSnoreID is None:
                continue

            # forward pass
            recon_tensor = model(noisy_tensor)
            recon_np = recon_tensor.squeeze().cpu().numpy()  # normalized [-1,1]

            # re-scale back
            raw_reconstructed = recon_np * max_abs_noisy

            # Save as "reconstructed_{snoreID}_{nonSnoreID}.wav" in original amplitude
            recon_fname = f"reconstructed_{snoreID}_{nonSnoreID}.wav"
            out_path = os.path.join(out_dir, recon_fname)
            sf.write(out_path, raw_reconstructed, sr)

            # Load raw snore => "1_{snoreID}.wav"
            snore_wave = None
            snore_path = os.path.join(snore_dir, f"1_{snoreID}.wav")
            if os.path.exists(snore_path):
                snore_wave, _ = librosa.load(snore_path, sr=sr, mono=True)
                # no normalization for plot

            # Load raw non-snore => "0_{nonSnoreID}.wav"
            nonsnore_wave = None
            nonsnore_path = os.path.join(nonsnore_dir, f"0_{nonSnoreID}.wav")
            if os.path.exists(nonsnore_path):
                nonsnore_wave, _ = librosa.load(nonsnore_path, sr=sr, mono=True)

            # Cosine sim: let's do it in normalized domain => recon_np vs. snore_normalized
            cos_sim = 0.0
            if snore_wave is not None:
                # also normalize the snore wave to compare
                max_abs_snore = np.max(np.abs(snore_wave)) if len(snore_wave) else 0
                if max_abs_snore > 1e-9:
                    snore_norm = snore_wave / max_abs_snore
                    cos_sim = cosine_similarity(recon_np, snore_norm)
                else:
                    cos_sim = 0.0

            results.append({
                "noisy_file": noisy_fname[0],
                "reconstructed_file": recon_fname,
                "original_file": f"1_{snoreID}.wav",
                "cosine_similarity": cos_sim,
                # store waveforms for final plotting
                "original_non_snore_wave": nonsnore_wave,       # raw amplitude
                "original_snore_wave": snore_wave,             # raw amplitude
                "noisy_wave": raw_noisy_wave,                  # raw amplitude
                "reconstructed_wave": raw_reconstructed        # raw amplitude
            })

    return results


# ----------------------------------------------------------------------------
# 6) Plotting in original amplitude
# ----------------------------------------------------------------------------
def plot_waveforms(orig_ns, orig_snore, raw_noisy, raw_recon, cos_sim, out_file):
    """
    4-subplot figure: 1) original non-snoring
                      2) original snoring
                      3) raw noisy
                      4) raw reconstructed
    """
    fig, axes = plt.subplots(4, 1, figsize=(10,8), sharex=False)
    fig.suptitle(f"CosSim = {cos_sim:.4f}")

    if orig_ns is None:
        orig_ns = np.zeros_like(raw_noisy)
    axes[0].plot(orig_ns, color='gray')
    axes[0].set_title("Original Non-Snore (raw amplitude)")

    if orig_snore is None:
        orig_snore = np.zeros_like(raw_noisy)
    axes[1].plot(orig_snore, color='g')
    axes[1].set_title("Original Snore (raw amplitude)")

    axes[2].plot(raw_noisy, color='r')
    axes[2].set_title("Noisy (raw amplitude)")

    axes[3].plot(raw_recon, color='b')
    axes[3].set_title("Reconstructed (raw amplitude)")

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)


def compute_and_save_results(results, model_name):
    """
    Sort by cos sim => top5/bottom5, write CSV, plot with original amplitude
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

    now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    csv_dir  = os.path.join("results", model_name, "Preprocessed", "csv")
    plot_dir = os.path.join("results", model_name, "Preprocessed", "plot", now_str)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Write CSV
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
        writer.writerow([])
        writer.writerow(["METRICS","COUNT","AVERAGE","MAX","MIN"])
        writer.writerow([
            "COS_SIM",
            len(results),
            f"{avg_cos:.6f}",
            f"{max_cos:.6f} ({best_rec['reconstructed_file']})",
            f"{min_cos:.6f} ({worst_rec['reconstructed_file']})"
        ])
    print(f"CSV => {csv_path}")

    # Plot top5 & bottom5 in raw amplitude
    sorted_res = sorted(results, key=lambda x: x["cosine_similarity"])
    bottom_5 = sorted_res[:5]
    top_5    = sorted_res[-5:]

    def do_plot(r, prefix):
        out_file = os.path.join(plot_dir, f"{prefix}_{r['reconstructed_file'].replace('.wav','')}.png")
        plot_waveforms(
            r["original_non_snore_wave"],
            r["original_snore_wave"],
            r["noisy_wave"],
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
    parser = argparse.ArgumentParser(description="Inference for snore denoising with 4-wave plots in original amplitude.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Which model: UNet1D, ResUNet1D, etc.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to .pth checkpoint.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'.")
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

    # 2) Inference
    test_noisy_dir  = "Dataset/Preprocessed/Test/mixing/noisy"
    test_denoise_dir= "Dataset/Preprocessed/Test/denoising"
    test_snore_dir  = "Dataset/Preprocessed/Test/original/1"
    test_ns_dir     = "Dataset/Preprocessed/Test/original/0"

    if not os.path.exists(test_noisy_dir):
        print(f"[ERROR] Not found: {test_noisy_dir}")
        sys.exit(0)

    results = infer_and_evaluate(
        model, device,
        test_dir = test_noisy_dir,
        out_dir  = test_denoise_dir,
        snore_dir= test_snore_dir,
        nonsnore_dir = test_ns_dir,
        sr=16000
    )

    # 3) Summarize & Plot
    compute_and_save_results(results, args.model_name)
    print("Inference complete.")

if __name__ == "__main__":
    main()