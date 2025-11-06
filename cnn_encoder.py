"""
Step 5: CNN Encoder with Contrastive Learning
==============================================
Trains encoder using contrastive loss to maximize cluster separability.
Uses triplet loss to push similar sequences together and dissimilar apart.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CGRDataset(Dataset):
    """Memory-mapped CGR dataset with triplet sampling"""
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
            arr = np.expand_dims(arr, axis=0)
        
        image = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).float()
        
        if self.augment and torch.rand(1).item() > 0.5:
            image = image + torch.randn_like(image) * 0.02
            image = torch.clamp(image, 0.0, 1.0)
        
        return image


class CGREncoder(nn.Module):
    """Improved CNN encoder with projection head for contrastive learning"""
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
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
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_embed = nn.Linear(256, embedding_dim)
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128)
        )

    def forward(self, x, return_projection=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x).view(x.size(0), -1)
        embedding = self.fc_embed(x)
        
        if return_projection:
            projection = self.projection(embedding)
            return embedding, projection
        return embedding


class ContrastiveLoss(nn.Module):
    """NT-Xent contrastive loss"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        mask = torch.eye(2 * batch_size, device=z_i.device).bool()
        similarity_matrix.masked_fill_(mask, -9e15)
        
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0)
        
        negatives = similarity_matrix[~mask].view(2 * batch_size, -1)
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        return F.cross_entropy(logits, labels)


class EncoderTrainer:
    """Train encoder with contrastive learning"""
    def __init__(self, cgr_dir="dataset/cgr", output_dir="models", 
                 embedding_dim=128, batch_size=128, num_epochs=50, 
                 learning_rate=1e-3, device='cuda'):
        self.cgr_dir = Path(cgr_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Device: {self.device}")

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
        logger.info("\nTraining Encoder with Contrastive Learning")
        
        train_ds, val_ds = self.load_data()
        
        num_workers = min(os.cpu_count() // 2, 4)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, 
                                   num_workers=num_workers, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2, shuffle=False,
                                num_workers=num_workers, pin_memory=True, persistent_workers=True)
        
        model = CGREncoder(self.embedding_dim).to(self.device)
        criterion = ContrastiveLoss(temperature=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)
            for images in pbar:
                images = images.to(self.device, non_blocking=True)
                
                # Create augmented view
                aug_images = images + torch.randn_like(images) * 0.03
                aug_images = torch.clamp(aug_images, 0.0, 1.0)
                
                _, z_i = model(images, return_projection=True)
                _, z_j = model(aug_images, return_projection=True)
                
                loss = criterion(z_i, z_j)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    aug_images = images + torch.randn_like(images) * 0.02
                    aug_images = torch.clamp(aug_images, 0.0, 1.0)
                    
                    _, z_i = model(images, return_projection=True)
                    _, z_j = model(aug_images, return_projection=True)
                    loss = criterion(z_i, z_j)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            scheduler.step()
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model, 'best')
                logger.info(f"  Best model saved (val_loss={val_loss:.4f})")
        
        self.save_model(model, 'final')
        
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
        return model, history

    def save_model(self, model, suffix):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'dataset': 'combined_ITS_LSU_SSU'
        }
        torch.save(checkpoint, self.output_dir / f"cgr_encoder_{suffix}.pth")


def main():
    trainer = EncoderTrainer(
        cgr_dir="dataset/cgr",
        output_dir="models",
        embedding_dim=128,
        batch_size=128,
        num_epochs=50,
        learning_rate=1e-3,
        device='cuda'
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