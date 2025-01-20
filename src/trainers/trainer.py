import os
import glob
import datetime
import librosa
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------
# 1) Model Factory + Losses + EarlyStopping
# ----------------------------------------------------------

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

def stft_loss(pred, target):
    """
    STFT-based difference in time-frequency domain.
    """
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    b, c, length = pred.shape
    pred = pred.view(b*c, length)
    target = target.view(b*c, length)

    window = torch.hann_window(256, device=pred.device)
    pred_stft = torch.stft(pred, n_fft=256, hop_length=128, win_length=256,
                           window=window, return_complex=True)
    target_stft = torch.stft(target, n_fft=256, hop_length=128, win_length=256,
                             window=window, return_complex=True)

    return torch.mean((pred_stft - target_stft).abs())

def combined_loss(pred, target):
    """
    70% MSE + 30% STFT difference
    """
    mse = nn.MSELoss()(pred, target)
    stft = stft_loss(pred, target)
    return 0.7 * mse + 0.3 * stft

class EarlyStopping:
    """
    If val_loss doesn't improve by min_delta for patience epochs, stop.
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ----------------------------------------------------------
# 2) SnoreDataset
# ----------------------------------------------------------
class SnoreDataset(Dataset):
    """
    Loads 'noisy_<scale>_snore-1_XXX_non-0_YYY.wav' => find the matching '1_XXX.wav'
    from the "clean" directory. The clean snore is the training target.
    """
    def __init__(self, noisy_dir, clean_dir):
        super().__init__()
        self.noisy_files = glob.glob(os.path.join(noisy_dir, "*.wav"))

        # Build a dictionary for clean files: { "1_005.wav": "/path/1_005.wav" }
        self.clean_map = {}
        for fpath in glob.glob(os.path.join(clean_dir, "*.wav")):
            fname = os.path.basename(fpath)
            self.clean_map[fname] = fpath

    def __len__(self):
        return len(self.noisy_files)

    def parse_noisy_to_clean(self, noisy_name):
        """
        Given something like:
            'noisy_1.0_snore-1_005_non-0_123_aug2.wav'
        we want to find the substring '1_005' from between 'snore-' and '_non-'
        and then add '.wav' => '1_005.wav' which should be in clean_map.
        """
        base_no_ext = os.path.splitext(noisy_name)[0]
        # e.g. "noisy_1.0_snore-1_005_non-0_123_aug2"
        if ("snore-" in base_no_ext) and ("_non-" in base_no_ext):
            # Extract the snore filename portion
            after_snore = base_no_ext.split("snore-")[1]  # "1_005_non-0_123_aug2"
            snore_part = after_snore.split("_non-")[0]    # "1_005"
            candidate = snore_part + ".wav"               # "1_005.wav"
            return candidate
        return None

    def __getitem__(self, idx):
        # 1) Load the noisy file
        noisy_path = self.noisy_files[idx]
        noisy_name = os.path.basename(noisy_path)

        noisy_audio, _ = librosa.load(noisy_path, sr=16000, mono=True)
        max_abs_noisy = np.max(np.abs(noisy_audio)) if len(noisy_audio) else 0
        if max_abs_noisy > 1e-8:
            noisy_audio /= max_abs_noisy

        # 2) Find the matching snore (clean) file
        clean_candidate = self.parse_noisy_to_clean(noisy_name)
        if clean_candidate and (clean_candidate in self.clean_map):
            clean_path = self.clean_map[clean_candidate]
            clean_audio, _ = librosa.load(clean_path, sr=16000, mono=True)
            max_abs_clean = np.max(np.abs(clean_audio)) if len(clean_audio) else 0
            if max_abs_clean > 1e-8:
                clean_audio /= max_abs_clean
        else:
            # fallback to zeros if no match
            clean_audio = np.zeros_like(noisy_audio)

        # Return as torch tensors [1, length]
        noisy_tensor = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0)
        clean_tensor = torch.tensor(clean_audio, dtype=torch.float32).unsqueeze(0)

        return (noisy_tensor, clean_tensor)


# ----------------------------------------------------------
# 3) Trainer
# ----------------------------------------------------------
class Trainer:
    def __init__(self, args):
        self.args = args

        # 1) Create dataset & DataLoader
        self.train_dataset = SnoreDataset(args.train_noisy_dir, args.train_clean_dir)
        self.val_dataset   = SnoreDataset(args.val_noisy_dir, args.val_clean_dir)

        if len(self.train_dataset) == 0:
            raise ValueError(f"No training data in {args.train_noisy_dir}!")
        if len(self.val_dataset) == 0:
            raise ValueError(f"No validation data in {args.val_noisy_dir}!")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 2) Create model
        self.model = get_model(args.model_name)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 3) Optimizer & LR Scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # 4) Loss & EarlyStopping
        self.criterion = combined_loss
        self.early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-3)

        # 5) AMP GradScaler
        self.scaler = GradScaler()

        # 6) Setup experiment folder
        #    exps/noise_level_{0.5}/ResUNet1D/2025-01-21-12-00-00/weights_ResUNet1D.pth
        now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.exp_dir = os.path.join(
            "exps",
            f"noise_level_{args.noise_level}",
            args.model_name,
            now_str
        )
        os.makedirs(self.exp_dir, exist_ok=True)

        self.model_path = os.path.join(
            self.exp_dir, f"weights_{args.model_name}.pth"
        )

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(self.args.epochs):
            print(f"\n----- EPOCH {epoch+1}/{self.args.epochs} -----")

            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Save if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"[SAVE] Best => {self.model_path} (val_loss={val_loss:.6f})")

            # Early stop
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        print(f"Best model at: {self.model_path}")

    def _train_one_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0.0

        for (noisy, clean) in tqdm(self.train_loader, desc=f"Train epoch {epoch_idx+1}"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                output = self.model(noisy)
                loss = self.criterion(output, clean)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        epoch_loss = total_loss / len(self.train_loader)
        print(f"TRAIN EPOCH {epoch_idx+1} - Loss: {epoch_loss:.6f}")
        return epoch_loss

    def _validate(self, epoch_idx):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for (noisy, clean) in tqdm(self.val_loader, desc=f"Val epoch {epoch_idx+1}"):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                with autocast():
                    output = self.model(noisy)
                    loss = self.criterion(output, clean)

                total_loss += loss.item()

        val_loss = total_loss / len(self.val_loader)
        print(f"VAL EPOCH {epoch_idx+1} - Loss: {val_loss:.6f}")
        return val_loss