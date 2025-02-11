#!/usr/bin/env python3

import subprocess

def main():
    # List all model names you want to train
    model_list = [
        "UNet1D",
        "AdvancedCNNAutoencoder",
        "AttentionUNet1D",
        "WaveUNet1D",
        "ResUNet1D"
    ]
    
    # List the noise levels you want to train
    noise_levels = ["0.5", "1.0", "1.5"]

    # Common training hyperparameters
    epochs = "100"
    batch_size = "40"
    patience = "10"
    lr = "1e-4"

    # New hyperparameters
    num_workers = "4"
    init_features = "64"
    dropout_rate = "0.25"
    weight_decay = "1e-4"
    beta1 = "0.9"
    beta2 = "0.99"
    lr_warmup_epochs = "5"
    scheduler_patience = "3"
    min_delta = "1e-3"
    max_grad_norm = "1.0"
    
    # Boolean flags (store_true type): add them if you want True.
    # Here, we assume you want to enable both augmentation and cosine annealing.
    augment_flag = "--augment"
    use_cosine_flag = "--use_cosine"

    for model_name in model_list:
        for noise_level in noise_levels:
            # Build the command to run `train.py` with the desired arguments
            cmd = [
                "python", "train.py",
                "--model_name", model_name,
                "--noise_level", noise_level,
                "--epochs", epochs,
                "--batch_size", batch_size,
                "--lr", lr,
                "--patience", patience,
                augment_flag,
                "--num_workers", num_workers,
                "--init_features", init_features,
                "--dropout_rate", dropout_rate,
                "--weight_decay", weight_decay,
                "--beta1", beta1,
                "--beta2", beta2,
                "--lr_warmup_epochs", lr_warmup_epochs,
                use_cosine_flag,
                "--scheduler_patience", scheduler_patience,
                "--min_delta", min_delta,
                "--max_grad_norm", max_grad_norm
            ]

            print("\n========================================================")
            print(f"Training Model: {model_name} | Noise Level: {noise_level}")
            print("========================================================")

            # Run the command; check=True will raise an error if train.py fails
            subprocess.run(cmd, check=True)

    print("\n[INFO] All training runs completed successfully!")

if __name__ == "__main__":
    main()
