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

# -----------------------------------------------------------------
# Ensure we can import from src.models if 'src' is in the same dir.
# -----------------------------------------------------------------
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))         # so we can do `import trainer` if needed
sys.path.append(str(CURRENT_DIR / "src")) # so we can do `from models...` if your src is here

# ----------------------------------------------------------
# 1) Model Factory + Losses + EarlyStopping
# ----------------------------------------------------------
from src.models.UNet1D import UNet1D
from src.models.AdvancedCNNAutoencoder import AdvancedCNNAutoencoder
from src.models.AttentionUNet1D import AttentionUNet1D
from src.models.WaveUNet1D import WaveUNet1D
from src.models.ResUNet1D import ResUNet1D

def get_model(model_name: str, init_features=64, dropout_rate=0.15):
    """Factory function returning an instance of the requested model."""
    if model_name == 'UNet1D':
        return UNet1D(init_features=init_features, dropout_rate=dropout_rate)
    elif model_name == 'AdvancedCNNAutoencoder':
        return AdvancedCNNAutoencoder(dropout_rate = dropout_rate)
    elif model_name == 'AttentionUNet1D':
        return AttentionUNet1D()
    elif model_name == 'WaveUNet1D':
        return WaveUNet1D(dropout_rate = dropout_rate)
    elif model_name == 'ResUNet1D':
        return ResUNet1D(dropout_rate = dropout_rate)
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
    If val_loss doesn't improve by min_delta for 'patience' epochs, stop.
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
            # improved
            self.best_loss = val_loss
            self.counter = 0
        else:
            # not improved
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ----------------------------------------------------------
# 2) SnoreDataset
# ----------------------------------------------------------
class SnoreDataset(Dataset):
    """
    Each item is (noisy_audio, clean_audio).
    The 'noisy' path is e.g. /Train/mixing/noisy_0.5/noisy_0.5_snore-1_005_non-0_123.wav
    We find the snore portion '1_005' and load /Train/original/1/1_005.wav as the clean target.
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
        E.g. 'noisy_1.0_snore-1_005_non-0_123.wav' -> find substring '1_005'
        and add '.wav' => '1_005.wav' which is in self.clean_map.
        """
        base_no_ext = os.path.splitext(noisy_name)[0]
        if ("snore-" in base_no_ext) and ("_non-" in base_no_ext):
            after_snore = base_no_ext.split("snore-")[1]
            snore_part = after_snore.split("_non-")[0]
            candidate = snore_part + ".wav"
            return candidate
        return None

    def __getitem__(self, idx):
        noisy_path = self.noisy_files[idx]
        noisy_name = os.path.basename(noisy_path)

        # load & normalize
        noisy_audio, _ = librosa.load(noisy_path, sr=16000, mono=True)
        max_abs_noisy = np.max(np.abs(noisy_audio)) if len(noisy_audio) else 0
        if max_abs_noisy > 1e-8:
            noisy_audio /= max_abs_noisy

        clean_candidate = self.parse_noisy_to_clean(noisy_name)
        if clean_candidate and (clean_candidate in self.clean_map):
            clean_path = self.clean_map[clean_candidate]
            clean_audio, _ = librosa.load(clean_path, sr=16000, mono=True)
            max_abs_clean = np.max(np.abs(clean_audio)) if len(clean_audio) else 0
            if max_abs_clean > 1e-8:
                clean_audio /= max_abs_clean
        else:
            clean_audio = np.zeros_like(noisy_audio)

        noisy_tensor = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0)
        clean_tensor = torch.tensor(clean_audio, dtype=torch.float32).unsqueeze(0)
        return (noisy_tensor, clean_tensor)

# ----------------------------------------------------------
# 3) Trainer
# ----------------------------------------------------------
class Trainer:
    def __init__(self, args):
        self.args = args

        # Create dataset & DataLoader
        self.train_dataset = SnoreDataset(args.train_noisy_dir, args.train_clean_dir)
        self.val_dataset   = SnoreDataset(args.val_noisy_dir, args.val_clean_dir)

        if len(self.train_dataset) == 0:
            raise ValueError(f"No training data found in {args.train_noisy_dir}!")
        if len(self.val_dataset) == 0:
            raise ValueError(f"No validation data found in {args.val_noisy_dir}!")

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

        # Create model
        self.model = get_model(args.model_name, init_features=args.init_features, dropout_rate=args.dropout_rate)
        if torch.cuda.device_count() > 1:
            print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel!")
            self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer & LR Scheduler (removed verbose to avoid warning)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # Loss & EarlyStopping
        self.criterion = combined_loss
        self.early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-3)

        # AMP GradScaler
        self.scaler = GradScaler()

        # Setup experiment folder
        now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.exp_dir = os.path.join(
            "exps",
            f"noise_level_{args.noise_level}",
            args.model_name,
            now_str
        )
        os.makedirs(self.exp_dir, exist_ok=True)

        self.model_path = os.path.join(self.exp_dir, f"weights_{args.model_name}.pth")

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(self.args.epochs):
            print(f"\n----- EPOCH {epoch+1}/{self.args.epochs} -----")

            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)

            # Scheduler step
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.6e}")

            # Check if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"[SAVE] Best => {self.model_path} (val_loss={val_loss:.6f})")

            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        print(f"Training done. Best validation loss: {best_val_loss:.6f}")
        print(f"Best model saved at: {self.model_path}")

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
        print(f"  [TRAIN] EPOCH {epoch_idx+1} - Loss: {epoch_loss:.6f}")
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
        print(f"  [VAL] EPOCH {epoch_idx+1} - Loss: {val_loss:.6f}")
        return val_loss
