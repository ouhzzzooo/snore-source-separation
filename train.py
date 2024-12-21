import argparse
import sys
from pathlib import Path
import torch

from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a snore denoising model.")

    # Basic arguments for demonstration
    parser.add_argument("--model_name", type=str, default="UNet1D",
                        help="Which model architecture to use: UNet1D, AdvancedCNNAutoencoder, AttentionUNet1D, WaveUNet1D, ResUNet1D")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--patience", type=int, default=5, help="EarlyStopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # Paths to your preprocessed dataset
    parser.add_argument("--train_noisy_dir", type=str, default="Dataset/Preprocessed/Train/mixing/noisy",
                        help="Path to the directory containing noisy (combined) training .wav files")
    parser.add_argument("--train_clean_dir", type=str, default="Dataset/Preprocessed/Train/original/1",
                        help="Path to the directory containing clean (snore) training .wav files")
    parser.add_argument("--val_noisy_dir", type=str, default="Dataset/Preprocessed/Val/mixing/noisy",
                        help="Path to the directory containing noisy (combined) validation .wav files")
    parser.add_argument("--val_clean_dir", type=str, default="Dataset/Preprocessed/Val/original/1",
                        help="Path to the directory containing clean (snore) validation .wav files")
    
    # Where to save the best model
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save the best model .pth file")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("----- TRAINING CONFIGURATION -----")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("----------------------------------")

    # Create trainer and run
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()