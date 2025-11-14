"""
Step 5: CNN Encoder (Autoencoder)
==============================================
Trains encoder and decoder using Mean Squared Error (MSE) loss for reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import gc
import os
import yaml # Added for config loading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Data Loading (Kept the same) ---

class CGRDataset(Dataset):
    """Memory-mapped CGR dataset"""
    def __init__(self, cgr_path: str, indices: list, augment: bool = False):
        self.cgr = np.load(cgr_path, mmap_mode='r')
        self.indices = list(indices)
        self.augment = augment
        self._needs_unsqueeze = (self.cgr.ndim == 3)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        arr = self.cgr[orig_idx]
        
        if self._needs_unsqueeze:
            # Add channel dimension (C, H, W)
            arr = np.expand_dims(arr, axis=0) 
        
        image = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).float()
        
        # Simple augmentation for robustness in AE
        if self.augment and torch.rand(1).item() > 0.5:
            image = image + torch.randn_like(image) * 0.02
            image = torch.clamp(image, 0.0, 1.0)
        
        return image


# --- Model Architecture (Modified for AE) ---

class CGREncoder(nn.Module):
    """CNN Encoder for feature extraction (AE part)"""
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 4 blocks of Conv -> BatchNorm -> ReLU -> MaxPool(stride 2)
        # Total downsampling factor: 2^4 = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2)
        )
        
        # Global Average Pooling (maps spatial size to 1x1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Final embedding layer
        self.fc_embed = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x).view(x.size(0), -1)
        embedding = self.fc_embed(x)
        return embedding


class CGRDecoder(nn.Module):
    """CNN Decoder for image reconstruction (AE part)"""
    def __init__(self, embedding_dim: int = 128, final_spatial_size: int = 4):
        super().__init__()
        
        # 4 is for 64x64 CGRs (64 / 16 = 4)
        # 8 is for 128x128 CGRs (128 / 16 = 8)
        self.final_spatial_size = final_spatial_size
        initial_input_dim = 256 * final_spatial_size * final_spatial_size
        
        # Linearly map latent vector back to the spatial feature size
        self.fc_decode = nn.Linear(embedding_dim, initial_input_dim)
        
        # 4 upsampling blocks (reverse of encoder)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(128), nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            # Final activation to ensure output is in the 0-1 range like the input CGR
            nn.Sigmoid() 
        )

    def forward(self, z):
        x = self.fc_decode(z)
        # ⚠️ FIX: Use the calculated spatial size for reshaping
        x = x.view(x.size(0), 256, self.final_spatial_size, self.final_spatial_size) 
        
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x


class CGRAutoencoder(nn.Module):
    """Combined Encoder and Decoder"""
    def __init__(self, embedding_dim: int = 128, image_size: int = 64):
        super().__init__()
        # Calculate the spatial size of the feature map before GAP
        final_spatial_size = image_size // (2**4)
        
        self.encoder = CGREncoder(embedding_dim=embedding_dim)
        self.decoder = CGRDecoder(embedding_dim=embedding_dim, final_spatial_size=final_spatial_size)
        
    def forward(self, x):
        # 1. Encode
        z = self.encoder(x)
        # 2. Decode
        x_hat = self.decoder(z)
        return x_hat, z # Return reconstructed image and embedding


# --- Training Logic (Modified for AE) ---

class EncoderTrainer:
    """Train Autoencoder with MSE loss"""
    def __init__(self, cgr_dir="dataset/cgr", output_dir="models", 
                 embedding_dim=128, batch_size=128, num_epochs=50, 
                 learning_rate=1e-3, device='cuda', config_path="config.yaml"):
        
        self.cgr_dir = Path(cgr_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 1. Load image size from config to fix dimension mismatch
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.image_size = config['cgr']['image_size']
        
        logger.info(f"Device: {self.device}, CGR Image Size: {self.image_size}x{self.image_size}")

    # ... (load_data is unchanged)
    def load_data(self):
        logger.info("Loading combined dataset...")
        cgr_file = self.cgr_dir / "combined_cgr.npy"
        cgr_mem = np.load(cgr_file, mmap_mode='r')
        total = cgr_mem.shape[0]
        
        logger.info(f"Total sequences: {total:,}")
        
        all_indices = np.arange(total)
        train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
        
        train_ds = CGRDataset(str(cgr_file), train_idx, augment=True)
        val_ds = CGRDataset(str(cgr_file), val_idx, augment=False)
        
        logger.info(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")
        return train_ds, val_ds

    def train(self):
        logger.info("\nTraining Autoencoder with MSE Loss")
        
        train_ds, val_ds = self.load_data()
        
        num_workers = min(os.cpu_count() // 2, 4)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, 
                                   num_workers=num_workers, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2, shuffle=False,
                                num_workers=num_workers, pin_memory=True, persistent_workers=True)
        
        # Initialize Autoencoder using dynamic image size
        model = CGRAutoencoder(self.embedding_dim, self.image_size).to(self.device)
        
        # Use Mean Squared Error Loss
        criterion = nn.MSELoss() 
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)
            for images in pbar:
                # Images are the input (x) AND the target (y) for reconstruction
                images = images.to(self.device, non_blocking=True)
                
                # Forward pass: get reconstructed image
                reconstructed_images, _ = model(images)
                
                # Calculate MSE Loss between input and output
                loss = criterion(reconstructed_images, images)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'MSE loss': f'{loss.item():.6f}'})
            
            train_loss /= len(train_loader)
            
            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    reconstructed_images, _ = model(images)
                    loss = criterion(reconstructed_images, images)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            scheduler.step()
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Epoch {epoch+1}: Train MSE={train_loss:.6f}, Val MSE={val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model.encoder, 'best_ae_encoder') # Only save the encoder
                logger.info(f"  Best model saved (val_loss={val_loss:.6f})")
        
        self.save_model(model.encoder, 'final_ae_encoder')
        
        with open(self.output_dir / "training_history_ae.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
        return model, history

    def save_model(self, model, suffix):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'dataset': 'combined_ITS_LSU_SSU'
        }
        # Save only the Encoder, as that is what is used for embeddings later
        torch.save(checkpoint, self.output_dir / f"cgr_encoder_{suffix}.pth")


def main():
    trainer = EncoderTrainer(
        cgr_dir="dataset/cgr",
        output_dir="models",
        embedding_dim=128,
        batch_size=128,
        num_epochs=100,
        learning_rate=1e-3,
        device='cuda',
        config_path="config.yaml"
    )
    
    try:
        model, history = trainer.train()
        return 0
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())