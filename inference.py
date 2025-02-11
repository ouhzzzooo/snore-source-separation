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
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# (Optional) If you still see "Too many open files" with num_workers>0,
# you can do:
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy("file_system")

# ----------------------------------------------------------------------------
# 1) Model imports
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
# 2) Dataset classes
# ----------------------------------------------------------------------------
class SyntheticNoisyDataset(Dataset):
    """
    Loads .wav from:
      Dataset/Preprocessed/Test/mixing/noisy_{noise_level}/
    Filenames typically: "noisy_1.0_snore-1_005_non-0_123.wav".
    """
    def __init__(self, noisy_dir, sr=16000):
        super().__init__()
        self.sr = sr
        self.noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
        print(f"[SyntheticNoisyDataset] Found {len(self.noisy_paths)} WAV files in '{noisy_dir}'.")

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        path = self.noisy_paths[idx]
        raw_wave, _ = librosa.load(path, sr=self.sr, mono=True)
        wave_tensor = torch.tensor(raw_wave, dtype=torch.float32)
        fname = os.path.basename(path)
        return wave_tensor, fname


class NonSyntheticDataset(Dataset):
    """
    Loads .wav from:
      Dataset/Preprocessed/Test/real_mixing/
    Will try to match each file with `Dataset/Raw/Test/original/1/{same_filename}`
    for amplitude reference & cos sim.
    """
    def __init__(self, real_mixing_dir, sr=16000):
        super().__init__()
        self.sr = sr
        self.noisy_paths = sorted(glob.glob(os.path.join(real_mixing_dir, "*.wav")))
        print(f"[NonSyntheticDataset] Found {len(self.noisy_paths)} WAV files in '{real_mixing_dir}'.")

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        path = self.noisy_paths[idx]
        raw_wave, _ = librosa.load(path, sr=self.sr, mono=True)
        wave_tensor = torch.tensor(raw_wave, dtype=torch.float32)
        fname = os.path.basename(path)
        return wave_tensor, fname


# ----------------------------------------------------------------------------
# 3) Cosine similarity
# ----------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    dot(a,b)/(||a||*||b||), trimming if lengths differ
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
# 4) Parse synthetic filenames => snore_part, nonSnore_part
# ----------------------------------------------------------------------------
def parse_test_filename(fname: str):
    """
    Example: "noisy_1.0_snore-1_005_non-0_123.wav"
      => (snore_part="1_005", nonSnore_part="0_123")
    Return (None, None) if not matching.
    """
    base = os.path.splitext(fname)[0]
    if "snore-" not in base or "_non-" not in base:
        return None, None
    after_snore = base.split("snore-")[1]  # e.g. "1_005_non-0_123"
    snore_part = after_snore.split("_non-")[0]
    nonSnore_part = after_snore.split("_non-")[1]
    return snore_part, nonSnore_part


# ----------------------------------------------------------------------------
# 5) Inference routines
# ----------------------------------------------------------------------------
def infer_synthetic(model, device, noise_level, sr=16000):
    """
    1) Loads from Dataset/Preprocessed/Test/mixing/noisy_{noise_level}
    2) Parse snore => load amplitude from Dataset/Raw/Test/original/1/{snore_part}.wav
    3) Output cos sim, store reconstructed_{snore_part}_{nonSnore_part}.wav
       noise_reconstructed_{snore_part}_{nonSnore_part}.wav
    """
    test_dir = os.path.join("Dataset", "Preprocessed", "Test", "mixing", f"noisy_{noise_level}")
    dataset = SyntheticNoisyDataset(test_dir, sr=sr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    raw_snore_root = os.path.join("Dataset", "Raw", "Test", "original", "1")
    results = []
    model.eval()

    for batch in tqdm(loader, desc=f"Inference synthetic => noisy_{noise_level}"):
        noisy_wave_t, fname_t = batch
        noisy_wave  = noisy_wave_t[0].cpu().numpy()
        noisy_fname = fname_t[0]

        # Normalize to [-1,1]
        max_abs_noisy = np.max(np.abs(noisy_wave)) if noisy_wave.size > 0 else 0.0
        normalized_noisy = noisy_wave / max_abs_noisy if max_abs_noisy > 1e-9 else noisy_wave

        snore_part, nonSnore_part = parse_test_filename(noisy_fname)
        if not snore_part or not nonSnore_part:
            # skip if naming doesn't match
            continue

        # forward pass
        input_tensor = torch.tensor(normalized_noisy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            recon_tensor = model(input_tensor)
        recon_np = recon_tensor.squeeze().cpu().numpy()  # normalized output

        # load raw snore => amplitude
        raw_snore_path = os.path.join(raw_snore_root, f"{snore_part}.wav")
        snore_amp = 0.0
        snore_wave = None
        if os.path.exists(raw_snore_path):
            s_wave, _ = librosa.load(raw_snore_path, sr=sr, mono=True)
            if len(s_wave) > 0:
                snore_amp = np.max(np.abs(s_wave))
                snore_wave = s_wave

        # scale reconstruction up
        raw_reconstructed = recon_np * snore_amp

        # cos sim
        cos_sim = 0.0
        if snore_wave is not None and len(snore_wave) > 0:
            max_snore = np.max(np.abs(snore_wave))
            if max_snore > 1e-9:
                snore_norm = snore_wave / max_snore
                cos_sim = cosine_similarity(recon_np, snore_norm)

        # noise = noisy - reconstructed
        noise_file_name = f"noise_reconstructed_{snore_part}_{nonSnore_part}.wav"
        noise_only = noisy_wave - raw_reconstructed

        results.append({
            "noisy_file": noisy_fname,
            "snore_file": f"{snore_part}.wav",
            "reconstructed_file": f"reconstructed_{snore_part}_{nonSnore_part}.wav",
            "noise_reconstructed_file": noise_file_name,
            "cosine_similarity": cos_sim,

            "raw_noisy_wave": noisy_wave,
            "snore_wave": snore_wave,
            "reconstructed_wave": raw_reconstructed,
            "noise_wave": noise_only
        })

    return results


def infer_non_synthetic(model, device, noise_level, sr=16000):
    """
    1) Loads from Dataset/Preprocessed/Test/real_mixing
    2) Tries to match each "realNoisy.wav" with "Dataset/Raw/Test/original/1/realNoisy.wav"
       for amplitude referencing & cos sim.
    3) Output => reconstructed_{realNoisy.wav}, noise_reconstructed_{realNoisy.wav}
    """
    real_dir = os.path.join("Dataset", "Preprocessed", "Test", "real_mixing")
    dataset = NonSyntheticDataset(real_dir, sr=sr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # We'll also look for a matching snore in:
    # Dataset/Raw/Test/original/1/{same filename}
    raw_snore_root = os.path.join("Dataset", "Raw", "Test", "original", "1")

    results = []
    model.eval()

    for batch in tqdm(loader, desc="Inference non_synthetic => real_mixing"):
        noisy_wave_t, fname_t = batch
        noisy_wave  = noisy_wave_t[0].cpu().numpy()
        noisy_fname = fname_t[0]

        # normalize
        max_abs_noisy = np.max(np.abs(noisy_wave)) if noisy_wave.size > 0 else 0.0
        normalized_noisy = noisy_wave / max_abs_noisy if max_abs_noisy > 1e-9 else noisy_wave

        input_tensor = torch.tensor(normalized_noisy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            recon_tensor = model(input_tensor)
        recon_np = recon_tensor.squeeze().cpu().numpy()

        # Now let's see if there's a raw snore with the same filename in Raw
        raw_snore_path = os.path.join(raw_snore_root, noisy_fname)
        snore_amp = 0.0
        snore_wave = None

        if os.path.exists(raw_snore_path):
            s_wave, _ = librosa.load(raw_snore_path, sr=sr, mono=True)
            if len(s_wave) > 0:
                snore_amp = np.max(np.abs(s_wave))
                snore_wave = s_wave

        # scale up with snore_amp if found, else use input amplitude
        if snore_wave is not None:
            raw_reconstructed = recon_np * snore_amp
        else:
            # fallback: scale by input amplitude (like before)
            raw_reconstructed = recon_np * max_abs_noisy

        # cos sim
        cos_sim = 0.0
        if snore_wave is not None and len(snore_wave) > 0:
            max_snore = np.max(np.abs(snore_wave))
            if max_snore > 1e-9:
                snore_norm = snore_wave / max_snore
                # but note recon_np is still "normalized" output
                # if you want shape-based comparison, do cos sim with recon_np vs. snore_norm
                cos_sim = cosine_similarity(recon_np, snore_norm)

        # noise wave
        noise_file_name = f"noise_reconstructed_{noisy_fname}"
        noise_only = noisy_wave - raw_reconstructed

        results.append({
            "noisy_file": noisy_fname,
            "snore_file": noisy_fname if snore_wave is not None else None,
            "reconstructed_file": f"reconstructed_{noisy_fname}",
            "noise_reconstructed_file": noise_file_name,
            "cosine_similarity": cos_sim,

            "raw_noisy_wave": noisy_wave,
            "snore_wave": snore_wave,  # might be None
            "reconstructed_wave": raw_reconstructed,
            "noise_wave": noise_only
        })

    return results


# ----------------------------------------------------------------------------
# 6) Plotting & result saving
# ----------------------------------------------------------------------------
def plot_waveforms(snore_wave, noisy_wave, recon_wave, cos_sim, out_file):
    """
    top=snore, mid=noisy, bottom=reconstructed
    If snore_wave=None => plot zeros
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(f"CosSim = {cos_sim if cos_sim else 0.0:.4f}")

    if snore_wave is None:
        snore_wave = np.zeros_like(noisy_wave)

    axes[0].plot(snore_wave, color='g')
    axes[0].set_title("Raw Snore (or zero if no reference)")

    axes[1].plot(noisy_wave, color='r')
    axes[1].set_title("Noisy Input")

    axes[2].plot(recon_wave, color='b')
    axes[2].set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)


def compute_and_save_results(results, model_name, noise_level, mode):
    """
    - Reconstructed => results/{model_name}/noise_level_{noise_level}/{mode}/reconstructed/
    - Noise => results/{model_name}/noise_level_{noise_level}/{mode}/noise_reconstructed/
    - CSV => inside 'csv/', plots => 'plots_{timestamp}/'
    - If we have cos_sims > 0 for any item => do top/bottom-5 plots
    """
    if len(results) == 0:
        print("No results => skip.")
        return

    base_dir = os.path.join("results", model_name, f"noise_level_{noise_level}", mode)
    csv_dir = os.path.join(base_dir, "csv")
    now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    plot_dir = os.path.join(base_dir, f"plots_{now_str}")
    recon_dir = os.path.join(base_dir, "reconstructed")
    noise_recon_dir = os.path.join(base_dir, "noise_reconstructed")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(noise_recon_dir, exist_ok=True)

    # Save WAVs
    for item in results:
        # reconstructed
        wav_out_name = item["reconstructed_file"]
        out_path = os.path.join(recon_dir, wav_out_name)
        sf.write(out_path, item["reconstructed_wave"], 16000)

        # noise_reconstructed
        noise_out_name = item["noise_reconstructed_file"]
        noise_path = os.path.join(noise_recon_dir, noise_out_name)
        sf.write(noise_path, item["noise_wave"], 16000)

    # Write CSV
    csv_path = os.path.join(csv_dir, f"{now_str}.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "noisy_file", 
            "reconstructed_file",
            "noise_reconstructed_file", 
            "snore_file", 
            "cosine_similarity"
        ])
        for r in results:
            cos_str = f"{r['cosine_similarity']:.6f}"
            writer.writerow([
                r["noisy_file"],
                r["reconstructed_file"],
                r["noise_reconstructed_file"],
                r["snore_file"] if r["snore_file"] else "",
                cos_str
            ])

    # Gather all cos sims
    cos_sims = [r["cosine_similarity"] for r in results if r["cosine_similarity"] is not None]
    # If none or all zeros, you might see no variation, but let's do top/bottom anyway
    if len(cos_sims) == 0:
        print("\n[INFO] No cos sim data found => no top/bottom plots.")
        return

    avg_cos = sum(cos_sims)/len(cos_sims)
    max_cos = max(cos_sims)
    min_cos = min(cos_sims)
    best_rec = max(results, key=lambda x: x["cosine_similarity"])
    worst_rec= min(results, key=lambda x: x["cosine_similarity"])

    print(f"\n==== Cosine Similarity Summary ====")
    print(f"Count: {len(cos_sims)}")
    print(f"Average: {avg_cos:.4f}")
    print(f"Max: {max_cos:.4f} => {best_rec['reconstructed_file']}")
    print(f"Min: {min_cos:.4f} => {worst_rec['reconstructed_file']}")

    # Append summary to CSV
    with open(csv_path, "a", newline="") as f2:
        writer2 = csv.writer(f2)
        writer2.writerow([])
        writer2.writerow(["METRICS","COUNT","AVERAGE","MAX","MIN"])
        writer2.writerow([
            "COS_SIM",
            len(cos_sims),
            f"{avg_cos:.6f}",
            f"{max_cos:.6f} ({best_rec['reconstructed_file']})",
            f"{min_cos:.6f} ({worst_rec['reconstructed_file']})"
        ])

    # Plot top-5 & bottom-5
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

    print(f"Plots => {plot_dir}")


# ----------------------------------------------------------------------------
# 7) Main
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for snore denoising with synthetic & non-synthetic modes. "
                    "Saves both 'reconstructed' and 'noise_reconstructed' WAVs. "
                    "Computes cos sim + top/bottom plots for ALL modes."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Which model: UNet1D, ResUNet1D, etc.")
    parser.add_argument("--noise_level", type=str, default="1.0",
                        choices=["0.5","1.0","1.5"],
                        help="Noise-level subfolder for synthetic => 'noisy_0.5', 'noisy_1.0', etc.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the .pth checkpoint (weights).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'.")
    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["synthetic","non_synthetic"],
                        help="Choose 'synthetic' => read from mixing/noisy_{noise_level}, "
                             "'non_synthetic' => read from real_mixing.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # 1) Load model
    model = get_model(args.model_name)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2) Run inference
    if args.mode == "synthetic":
        results = infer_synthetic(model, device, args.noise_level, sr=16000)
    else:
        results = infer_non_synthetic(model, device, args.noise_level, sr=16000)

    # 3) Save + analyze results
    compute_and_save_results(results, args.model_name, args.noise_level, args.mode)

    print("\n[INFO] Inference complete!")


if __name__ == "__main__":
    main()