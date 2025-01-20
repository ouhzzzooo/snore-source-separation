import argparse
import sys
from pathlib import Path
import torch

# Adjust to your project structure
# e.g. your code can live in "src" or alongside
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from trainer import Trainer  # or from src.trainers.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a snore denoising model.")

    # Which model to train
    parser.add_argument("--model_name", type=str, default="UNet1D",
                        choices=["UNet1D", "AdvancedCNNAutoencoder", "AttentionUNet1D", 
                                 "WaveUNet1D", "ResUNet1D"],
                        help="Model architecture.")
    
    # Noise-level choice
    parser.add_argument("--noise_level", type=str, default="1.0",
                        choices=["0.5", "1.0", "1.5"],
                        help="Which noise-level folder to use: noisy_0.5, noisy_1.0, or noisy_1.5.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Based on the chosen noise_level, set the Train/Val "noisy" paths
    # ----------------------------------------------------------------
    # The "clean" folder is always the snore folder in /original/1
    # We'll rely on the noise_level subfolders in mixing/
    #
    # e.g. for noise_level=0.5 => Dataset/Preprocessed/Train/mixing/noisy_0.5
    #
    # If you wish, you can override these via the CLI in a more flexible approach,
    # but for simplicity, we fix them based on noise_level here.
    # ----------------------------------------------------------------
    base_preprocessed = "Dataset/Preprocessed"

    train_noisy_dir = f"{base_preprocessed}/Train/mixing/noisy_{args.noise_level}"
    val_noisy_dir   = f"{base_preprocessed}/Val/mixing/noisy_{args.noise_level}"

    # Clean data (snore) is always under /original/1
    train_clean_dir = f"{base_preprocessed}/Train/original/1"
    val_clean_dir   = f"{base_preprocessed}/Val/original/1"

    # We attach these to args for the Trainer
    args.train_noisy_dir = train_noisy_dir
    args.val_noisy_dir   = val_noisy_dir
    args.train_clean_dir = train_clean_dir
    args.val_clean_dir   = val_clean_dir

    return args

def main():
    torch.cuda.empty_cache()
    args = parse_args()

    print("----- TRAINING CONFIGURATION -----")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("----------------------------------")

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()