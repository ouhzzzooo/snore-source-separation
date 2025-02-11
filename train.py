import argparse
import sys
from pathlib import Path
import torch

# Make sure Python can import 'trainer' (which is in the same folder).
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a snore denoising model.")

    # Model architecture choice
    parser.add_argument("--model_name", type=str, default="UNet1D",
                        choices=["UNet1D", "AdvancedCNNAutoencoder", "AttentionUNet1D", 
                                 "WaveUNet1D", "ResUNet1D"],
                        help="Which model architecture to use.")

    # Optional noise_level (used only for naming or quick defaults)
    parser.add_argument("--noise_level", type=str, default="1.0",
                        choices=["0.5", "1.0", "1.5"],
                        help="Noise-level folder name: 'noisy_0.5', 'noisy_1.0', or 'noisy_1.5'.")

    # Dataset paths (you can override these from CLI)
    parser.add_argument("--train_noisy_dir", type=str,
                        default="Dataset/Preprocessed/Train/mixing/noisy_1.0",
                        help="Path to the TRAIN noisy audio folder.")
    parser.add_argument("--val_noisy_dir", type=str,
                        default="Dataset/Preprocessed/Val/mixing/noisy_1.0",
                        help="Path to the VAL noisy audio folder.")
    parser.add_argument("--train_clean_dir", type=str,
                        default="Dataset/Preprocessed/Train/original/1",
                        help="Path to the TRAIN clean snore folder.")
    parser.add_argument("--val_clean_dir", type=str,
                        default="Dataset/Preprocessed/Val/original/1",
                        help="Path to the VAL clean snore folder.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")

    # New hyperparameters for training improvements:
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--init_features", type=int, default=64, help="Initial number of features in the model.")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate used in the model.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 for the Adam optimizer.")
    parser.add_argument("--lr_warmup_epochs", type=int, default=5, help="Number of epochs for learning rate warmup.")
    parser.add_argument("--use_cosine", action="store_true", help="Use cosine annealing after warmup.")
    parser.add_argument("--scheduler_patience", type=int, default=3, help="Patience for the ReduceLROnPlateau scheduler.")
    parser.add_argument("--min_delta", type=float, default=1e-3, help="Minimum delta for EarlyStopping improvement.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping.")

    args = parser.parse_args()

    # Convenience: update dataset folders if noise_level is changed and defaults are in place.
    desired_noisy_subfolder = f"noisy_{args.noise_level}"
    if args.train_noisy_dir == "Dataset/Preprocessed/Train/mixing/noisy_1.0" and args.noise_level != "1.0":
        args.train_noisy_dir = f"Dataset/Preprocessed/Train/mixing/{desired_noisy_subfolder}"
    if args.val_noisy_dir == "Dataset/Preprocessed/Val/mixing/noisy_1.0" and args.noise_level != "1.0":
        args.val_noisy_dir   = f"Dataset/Preprocessed/Val/mixing/{desired_noisy_subfolder}"

    return args

def main():
    # Clear any leftover GPU memory
    torch.cuda.empty_cache()

    # Parse command-line arguments
    args = parse_args()

    print("----- TRAINING CONFIGURATION -----")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("----------------------------------")

    # Create the trainer and launch training
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
