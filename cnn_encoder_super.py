"""
Step 5: CNN Encoder Training (Supervised)
=========================================
Train a SINGLE CNN Encoder to learn compositional embeddings from ALL markers 
(ITS, LSU, SSU) using SUPERVISED contrastive loss and a classification head.

This script reads 'dataset/metadata.csv' to get taxID labels for training.

This script includes:
 - Loading data using the 'dataset/metadata.csv' file for labels.
 - A global LabelEncoder to handle all taxIDs.
 - A check to ensure there is more than 1 class to train on.
 - A combined loss (Supervised Contrastive + Cross Entropy).
 - The fix for the 'non-writable numpy array' UserWarning.
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
from sklearn.preprocessing import LabelEncoder
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
    SUPERVISED Memmap-backed dataset.
    It loads CGR images and their corresponding integer labels.
    
    Args:
      - cgr_npy_path: path to combined_cgr.npy
      - valid_global_idx: list of global indices (into memmap) included in this dataset
      - labels_int: 1D numpy array of int labels aligned with valid_global_idx (same order)
      - label_encoder: sklearn LabelEncoder fitted on all valid tax IDs
      - augment: whether to apply simple augmentations
    """
    def __init__(self,
                 cgr_npy_path: str,
                 valid_global_idx: Sequence[int],
                 labels_int: np.ndarray,
                 label_encoder: LabelEncoder,
                 augment: bool = False):
        self.cgr_path = cgr_npy_path
        self.cgr = np.load(cgr_npy_path, mmap_mode='r')
        self.valid_global_idx = list(valid_global_idx)
        assert len(self.valid_global_idx) == len(labels_int), "indices and labels must match length"
        self.labels = np.array(labels_int, dtype=np.int64)
        self.label_encoder = label_encoder
        self.num_classes = len(label_encoder.classes_) if label_encoder is not None else int(self.labels.max() + 1)
        # detect if memmap lacks channel dim
        self._needs_unsqueeze = (self.cgr.ndim == 3)
        self.augment = augment

        logger.info(f"Dataset created: {len(self)} sequences, {self.num_classes} taxa")

    def __len__(self):
        return len(self.valid_global_idx)

    def __getitem__(self, idx):
        orig_idx = self.valid_global_idx[idx]
        arr = self.cgr[orig_idx]  # memmap view (may be non-writable)

        # Ensure shape (1, H, W)
        if self._needs_unsqueeze:
            arr = np.expand_dims(arr, axis=0)

        # --- FIX FOR USERWARNING ---
        # Make a tiny writable copy to avoid PyTorch UserWarning
        # about non-writable numpy memmap arrays.
        arr_writable = np.array(arr, dtype=np.float32, copy=True)

        image = torch.from_numpy(arr_writable).float()
        label = int(self.labels[idx])

        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[2])
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[1])
            # small gaussian noise
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(image) * 0.01
                image = image + noise
                image = torch.clamp(image, 0.0, 1.0)

        return image, label


class CGREncoder(nn.Module):
    """CNN Encoder for CGR images (matches autoencoder)"""
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


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss with safety for zero-positive rows"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask, device=device) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        max_vals, _ = similarity_matrix.max(dim=1, keepdim=True)
        stable_logits = similarity_matrix - max_vals.detach()
        exp_logits = torch.exp(stable_logits) * logits_mask

        denom = exp_logits.sum(1, keepdim=True) + 1e-12
        log_prob = stable_logits - torch.log(denom)

        positive_count = mask.sum(1)
        positive_count_safe = positive_count.clone()
        positive_count_safe[positive_count_safe == 0] = 1.0

        mean_log_prob_pos = (mask * log_prob).sum(1) / positive_count_safe
        mean_log_prob_pos = mean_log_prob_pos * (positive_count > 0).float()

        num_nonzero = (positive_count > 0).sum().float()
        if num_nonzero.item() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = - mean_log_prob_pos.sum() / num_nonzero
        return loss


class CNNTrainer:
    """Train CNN encoder on combined dataset (Supervised)"""
    def __init__(self,
                 cgr_dir: str = "dataset/cgr",
                 metadata_file: str = "dataset/metadata.csv",
                 output_dir: str = "models",
                 embedding_dim: int = 128,
                 batch_size: int = 64,
                 num_epochs: int = 50,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda'):
        self.cgr_dir = Path(cgr_dir)
        self.metadata_file = Path(metadata_file)
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
        Load combined memmap and build global LabelEncoder over only sequences that have taxIDs.
        Then split VALID indices (those with taxID) into train/val and return memmap-backed datasets.
        """
        logger.info("Loading COMBINED dataset (ITS + LSU + SSU)...")

        cgr_file = self.cgr_dir / "combined_cgr.npy"
        if not cgr_file.exists():
            raise FileNotFoundError(f"Combined CGR not found: {cgr_file}")

        cgr_mem = np.load(cgr_file, mmap_mode='r')
        logger.info(f"  CGR shape: {cgr_mem.shape}")
        logger.info(f"  CGR size: {cgr_file.stat().st_size / (1024**3):.2f} GB")

        metadata_pkl = self.cgr_dir / "combined_metadata.pkl"
        if not metadata_pkl.exists():
            raise FileNotFoundError(f"Combined metadata not found: {metadata_pkl}")
        with open(metadata_pkl, 'rb') as f:
            combined_metadata = pickle.load(f)
        seq_ids = combined_metadata['seq_ids']
        logger.info(f"  Total sequences: {len(seq_ids):,}")

        # taxonomy metadata
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Taxonomy metadata CSV not found: {self.metadata_file}")
        try:
            metadata_df = pd.read_csv(self.metadata_file, dtype={'taxID': str})
        except Exception as e:
            logger.error(f"Error reading {self.metadata_file}: {e}")
            logger.error("Please ensure the file exists and is a valid CSV.")
            raise

        # seqID -> taxID mapping
        metadata_dict = dict(zip(metadata_df['seqID'], metadata_df['taxID']))

        # Build list of tax_ids, mark NA for missing
        tax_ids_all: List[str] = [str(metadata_dict.get(sid, 'NA')) for sid in seq_ids]

        # Only keep indices where taxID exists (not 'NA')
        valid_global_idx = [i for i, t in enumerate(tax_ids_all) if t != 'NA']
        if len(valid_global_idx) == 0:
            raise ValueError(f"No sequences with taxID found. Check {self.metadata_file}.")

        # labels (strings) aligned with valid_global_idx
        valid_tax_ids = [tax_ids_all[i] for i in valid_global_idx]

        # Fit a global LabelEncoder on all valid tax IDs
        label_encoder = LabelEncoder()
        labels_int = label_encoder.fit_transform(valid_tax_ids)  # ints 0..(C-1)
        num_classes = len(label_encoder.classes_)

        logger.info(f"  Valid sequences with taxID: {len(valid_global_idx):,}")
        logger.info(f"  Unique taxa (global): {num_classes:,}")
        
        # --- CRITICAL CHECK ---
        # This check will now catch the *real* problem if your new metadata
        # file is still empty or has only one class.
        if num_classes < 2:
            logger.error(f"Found only {num_classes} unique taxa in the dataset.")
            logger.error("Training a classifier or contrastive loss model requires at least 2 classes.")
            logger.error(f"Please check your '{self.metadata_file}' (created by Step 3)")
            logger.error("and ensure the 'taxID' column has multiple different values.")
            if num_classes == 1:
                logger.error(f"The only class found was: {label_encoder.classes_[0]}")
            raise ValueError(f"Insufficient number of classes ({num_classes}) for training.")
        # --- END CHECK ---

        logger.info("\nSplitting train/validation (80/20) on valid sequences...")
        local_indices = np.arange(len(valid_global_idx))
        
        # Stratify ensures train/val splits have proportional classes
        # This is crucial for imbalanced datasets
        try:
            train_local, val_local = train_test_split(
                local_indices,
                test_size=0.2,
                random_state=42,
                shuffle=True,
                stratify=labels_int  
            )
        except ValueError as e:
            logger.error(f"Stratified split failed: {e}")
            logger.error("This can happen if some classes have only 1 sample.")
            logger.info("Falling back to a non-stratified split.")
            train_local, val_local = train_test_split(
                local_indices,
                test_size=0.2,
                random_state=42,
                shuffle=True
            )
        
        # Map back to global indices
        train_global_idx = [valid_global_idx[i] for i in train_local]
        val_global_idx = [valid_global_idx[i] for i in val_local]

        logger.info(f"  Train indices: {len(train_global_idx):,}")
        logger.info(f"  Val indices: {len(val_global_idx):,}")

        # Get labels for each split
        train_labels_int = labels_int[train_local]
        val_labels_int = labels_int[val_local]


        logger.info("\nCreating training dataset (memmap-backed)...")
        train_dataset = CGRMemmapDataset(
            str(cgr_file),
            valid_global_idx=train_global_idx,
            labels_int=train_labels_int,
            label_encoder=label_encoder,
            augment=True
        )

        logger.info("\nCreating validation dataset (memmap-backed)...")
        val_dataset = CGRMemmapDataset(
            str(cgr_file),
            valid_global_idx=val_global_idx,
            labels_int=val_labels_int,
            label_encoder=label_encoder,
            augment=False
        )

        logger.info(f"\nFinal datasets:")
        logger.info(f"  Train: {len(train_dataset):,} sequences")
        logger.info(f"  Val: {len(val_dataset):,} sequences")
        logger.info(f"  Total unique taxa (global): {num_classes:,}")

        return train_dataset, val_dataset

    def train(self):
        """Train encoder on combined dataset"""
        logger.info(f"\n{'='*60}")
        logger.info("Training SUPERVISED CNN Encoder on COMBINED Dataset")
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
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )

        # Initialize model
        model = CGREncoder(embedding_dim=self.embedding_dim).to(self.device)

        # Add classification head (num_classes from dataset)
        num_classes = train_dataset.num_classes
        classifier = nn.Linear(self.embedding_dim, num_classes).to(self.device)

        # Loss functions
        contrastive_loss = SupervisedContrastiveLoss().to(self.device)
        classification_loss = nn.CrossEntropyLoss()

        # Optimizer + scheduler
        params = list(model.parameters()) + list(classifier.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )

        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(self.num_epochs):
            model.train()
            classifier.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                embeddings = model(images)
                logits = classifier(embeddings)

                loss_contrast = contrastive_loss(embeddings, labels)
                loss_cls = classification_loss(logits, labels)
                
                # Weighting the losses
                loss = loss_contrast + 0.5 * loss_cls 

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss /= max(1, len(train_loader))

            # Validation
            model.eval()
            classifier.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    embeddings = model(images)
                    logits = classifier(embeddings)

                    loss_contrast = contrastive_loss(embeddings, labels)
                    loss_cls = classification_loss(logits, labels)
                    loss = loss_contrast + 0.5 * loss_cls

                    val_loss += loss.item()

                    _, predicted = torch.max(logits, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            val_loss /= max(1, len(val_loader))
            val_acc = correct / max(1, total)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            scheduler.step()

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model, train_dataset.label_encoder, 'best')
                logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

        # Save final model
        self.save_model(model, train_dataset.label_encoder, 'final')

        # Save training history
        history_file = self.output_dir / "supervised_training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"\n✓ Training complete!")
        logger.info(f"  Best val loss: {best_val_loss:.4f}")
        logger.info(f"  Models saved to: {self.output_dir}")

        return model, history

    def save_model(self, model, label_encoder, suffix: str):
        """Save model checkpoint (encoder only)"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'label_encoder_classes': label_encoder.classes_.tolist() if label_encoder is not None else [],
            'dataset': 'combined_ITS_LSU_SSU'
        }
        # Save with a specific name for the supervised model
        save_path = self.output_dir / f"cgr_encoder_supervised_{suffix}.pth"
        torch.save(checkpoint, save_path)
        
        if suffix == 'final':
            logger.info(f"    Saved final model: {save_path}")


def main():
    """Main execution"""
    trainer = CNNTrainer(
        cgr_dir="dataset/cgr",
        metadata_file="dataset/metadata.csv",
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
    except ValueError as e:
        # Catch the specific error we added
        logger.error(f"Validation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())