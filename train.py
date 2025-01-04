import argparse
import sys
from pathlib import Path
import torch

# Adjust to your project structure
sys.path.append(str(Path(__file__).resolve().parent / 'src'))
from src.trainers.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a snore denoising model.")

    parser.add_argument("--model_name", type=str, default="UNet1D",
                        help="Model architecture: UNet1D, AdvancedCNNAutoencoder, AttentionUNet1D, WaveUNet1D, ResUNet1D")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # Preprocessed dataset paths (Train & Val)
    parser.add_argument("--train_noisy_dir", type=str,
                        default="Dataset/Preprocessed/Train/mixing/noisy",
                        help="Directory with noisy training WAV files")
    parser.add_argument("--train_clean_dir", type=str,
                        default="Dataset/Preprocessed/Train/original/1",
                        help="Directory with clean (snore) training WAV files")
    parser.add_argument("--val_noisy_dir", type=str,
                        default="Dataset/Preprocessed/Val/mixing/noisy",
                        help="Directory with noisy validation WAV files")
    parser.add_argument("--val_clean_dir", type=str,
                        default="Dataset/Preprocessed/Val/original/1",
                        help="Directory with clean (snore) validation WAV files")

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