"""
Step 5: CNN Autoencoder Training (Combined Dataset)
==================================================
Train a CNN Autoencoder to learn compositional embeddings from ALL markers 
(ITS, LSU, SSU) in an UNSUPERVISED manner.

This regenerated file is a true autoencoder:
 - It uses an Encoder-Decoder architecture.
 - It is UNSUPERVISED (does not load or use any 'taxID' labels).
 - It uses Mean Squared Error (MSE) reconstruction loss.
 - It saves only the ENCODER weights for downstream embedding tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from typing import Tuple, Optional, Sequence, List
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import gc
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CGRMemmapDataset(Dataset):
    """
    UNSUPERVISED Memmap-backed dataset.
    It only loads the CGR images and does not require any labels.
    
    Args:
      - cgr_npy_path: path to combined_cgr.npy
      - indices: list of global indices (into memmap) included in this dataset
      - augment: whether to apply simple augmentations
    """
    def __init__(self,
                 cgr_npy_path: str,
                 indices: Sequence[int],
                 augment: bool = False):
        self.cgr_path = cgr_npy_path
        self.cgr = np.load(cgr_npy_path, mmap_mode='r')
        self.indices = list(indices)
        self._needs_unsqueeze = (self.cgr.ndim == 3)
        self.augment = augment

        logger.info(f"Dataset created: {len(self)} sequences")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        arr = self.cgr[orig_idx]  # memmap view

        if self._needs_unsqueeze:
            arr = np.expand_dims(arr, axis=0)

        # Make a tiny writable copy for PyTorch
        arr_writable = np.array(arr, dtype=np.float32, copy=True)
        image = torch.from_numpy(arr_writable).float()

        if self.augment:
            # Add a small amount of noise as augmentation
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(image) * 0.01
                image = image + noise
                image = torch.clamp(image, 0.0, 1.0)
        
        # For an autoencoder, the input is also the target
        return image, image


class CGREncoder(nn.Module):
    """CNN Encoder for CGR images"""
    def __init__(self, embedding_dim: int = 128, input_size: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 1x128x128
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 32x64x64
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 64x32x32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128x16x16
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 256x8x8
        )

        self.gap = nn.AdaptiveAvgPool2d(1) # 256x1x1
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding


class CGRDecoder(nn.Module):
    """CNN Decoder for CGR images. Mirrors the Encoder."""
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        self.fc = nn.Linear(embedding_dim, 256 * 8 * 8)
        
        # Start from 256x8x8
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0), # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0), # 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0), # 32x64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0), # 1x128x128
            nn.Sigmoid() # To scale output between 0 and 1, matching input
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8) # Reshape to 256x8x8
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x


class Autoencoder(nn.Module):
    """Combines the Encoder and Decoder."""
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.encoder = CGREncoder(embedding_dim)
        self.decoder = CGRDecoder(embedding_dim)
        
    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return reconstructed


class AutoencoderTrainer:
    """Train CNN Autoencoder on combined dataset"""
    def __init__(self,
                 cgr_dir: str = "dataset/cgr",
                 output_dir: str = "models",
                 embedding_dim: int = 128,
                 batch_size: int = 64,
                 num_epochs: int = 50,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda'):
        self.cgr_dir = Path(cgr_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {self.device}")

    def load_combined_data(self) -> Tuple[Dataset, Dataset]:
        """
        Load combined memmap and split indices for unsupervised training.
        This function DOES NOT load or use any labels.
        """
        logger.info("Loading COMBINED dataset (ITS + LSU + SSU)...")

        cgr_file = self.cgr_dir / "combined_cgr.npy"
        if not cgr_file.exists():
            raise FileNotFoundError(f"Combined CGR not found: {cgr_file}")

        cgr_mem = np.load(cgr_file, mmap_mode='r')
        cgr_shape = cgr_mem.shape
        total_sequences = cgr_shape[0]
        
        logger.info(f"  CGR shape: {cgr_shape}")
        logger.info(f"  CGR size: {cgr_file.stat().st_size / (1024**3):.2f} GB")
        logger.info(f"  Total sequences: {total_sequences:,}")

        # Split the *indices* (0 to N-1) into train/val
        logger.info("\nSplitting train/validation (80/20) on all sequences...")
        all_indices = np.arange(total_sequences)
        train_indices, val_indices = train_test_split(
            all_indices, test_size=0.2, random_state=42, shuffle=True
        )

        logger.info(f"  Train indices: {len(train_indices):,}")
        logger.info(f"  Val indices: {len(val_indices):,}")

        logger.info("\nCreating training dataset (memmap-backed)...")
        train_dataset = CGRMemmapDataset(
            str(cgr_file),
            indices=train_indices,
            augment=True # Apply augmentation to training data
        )

        logger.info("\nCreating validation dataset (memmap-backed)...")
        val_dataset = CGRMemmapDataset(
            str(cgr_file),
            indices=val_indices,
            augment=False # No augmentation on validation data
        )

        logger.info(f"\nFinal datasets:")
        logger.info(f"  Train: {len(train_dataset):,} sequences")
        logger.info(f"  Val: {len(val_dataset):,} sequences")

        return train_dataset, val_dataset

    def train(self):
        """Train autoencoder on combined dataset"""
        logger.info(f"\n{'='*60}")
        logger.info("Training CNN Autoencoder on COMBINED Dataset (Unsupervised)")
        logger.info(f"{'='*60}\n")

        # Load combined data
        train_dataset, val_dataset = self.load_combined_data()

        effective_batch = min(self.batch_size, 64)
        num_workers = min(os.cpu_count() // 2, 4)
        logger.info(f"Using {num_workers} workers for DataLoader")

        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        # Initialize Autoencoder model
        model = Autoencoder(embedding_dim=self.embedding_dim).to(self.device)

        # Loss function (Reconstruction Loss)
        criterion = nn.MSELoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )

        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)
            # Dataset returns (image, image)
            for images, targets in pbar:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Get reconstructions
                reconstructed_images = model(images)

                # Calculate loss
                loss = criterion(reconstructed_images, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss /= max(1, len(train_loader))

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    reconstructed_images = model(images)
                    loss = criterion(reconstructed_images, targets)
                    val_loss += loss.item()

            val_loss /= max(1, len(val_loader))

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            scheduler.step()

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                        f"Val Loss={val_loss:.4f}")

            # Save best model (based on lowest reconstruction loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the ENCODER part
                self.save_model(model.encoder, 'best')
                logger.info(f"  ✓ Best encoder saved (val_loss={val_loss:.4f})")

        # Save final model
        self.save_model(model.encoder, 'final')

        # Save training history
        history_file = self.output_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"\n✓ Training complete!")
        logger.info(f"  Best val reconstruction loss: {best_val_loss:.4f}")
        logger.info(f"  Encoder models saved to: {self.output_dir}")

        return model, history

    def save_model(self, encoder_model, suffix: str):
        """Save ENCODER model checkpoint"""
        checkpoint = {
            'model_state_dict': encoder_model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'dataset': 'combined_ITS_LSU_SSU'
        }
        save_path = self.output_dir / f"cgr_autoencoder_encoder_{suffix}.pth"
        
        # Only log final save to avoid clutter
        if suffix == 'final':
            torch.save(checkpoint, save_path)
            logger.info(f"    Saved final encoder: {save_path}")
        elif suffix == 'best':
            torch.save(checkpoint, save_path)
            # logger.info(f"    Saved best encoder: {save_path}") # Covered by msg in train loop


def main():
    """Main execution"""
    trainer = AutoencoderTrainer(
        cgr_dir="dataset/cgr",
        output_dir="models",
        embedding_dim=128,
        batch_size=64, 
        num_epochs=50,
        learning_rate=1e-3,
        device='cuda'
    )

    try:
        model, history = trainer.train()
        return 0
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())