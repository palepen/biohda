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
import math
import torchvision.transforms.functional as TF # <-- ADDED IMPORT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CGRDataset(Dataset):
    """
    Memory-mapped CGR dataset with improved augmentation.
    When augment=True, it returns two strongly augmented views of the same image.
    When augment=False, it returns a single, clean image.
    """
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
        
        # copy=True is essential as transforms cannot be applied to read-only memmap
        image = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).float()
        
        if self.augment:
            # Return TWO augmented views for contrastive learning
            view1 = self.apply_transforms(image)
            view2 = self.apply_transforms(image)
            return view1, view2
        else:
            # Return one clean view for validation
            return image
    
    def apply_transforms(self, image):
        """Applies a pipeline of strong, random augmentations."""
        # Note: TF functions expect [C, H, W]
        
        if torch.rand(1).item() > 0.5:
            # Horizontal flip
            image = TF.hflip(image)
            
        if torch.rand(1).item() > 0.5:
            # Small rotation (±10 degrees)
            angle = (torch.rand(1).item() - 0.5) * 20 # 20 degrees total range
            image = TF.rotate(image, angle)
            
        if torch.rand(1).item() > 0.5:
            # Gaussian noise
            image = image + torch.randn_like(image) * 0.015

        return torch.clamp(image, 0.0, 1.0)


# ===================================================================
# Modern CNN Building Blocks
# ===================================================================

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block: Modern CNN design that rivals transformers
    Uses depthwise convolution + LayerNorm + GELU
    """
    def __init__(self, dim, drop_path=0., layer_scale_init=1e-6):
        super().__init__()
        # Depthwise convolution (7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        # Pointwise convolutions (1x1)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        
        # Layer scale (learnable per-channel scaling)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        
        x = residual + self.drop_path(x)
        return x


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps (channels-first)"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop entire residual paths)"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SpatialAttention(nn.Module):
    """Lightweight spatial attention module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


# ===================================================================
# Enhanced CGR Encoder
# ===================================================================

class EnhancedCGREncoder(nn.Module):
    """
    Enhanced CNN encoder with modern architecture components:
    - ConvNeXt blocks for better feature extraction
    - Multi-scale feature fusion
    - Spatial attention
    - Larger embedding dimension (256 vs 128)
    """
    def __init__(self, embedding_dim=256, input_size=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Stem: Patchify input (4x4 patches)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=4),
            LayerNorm2d(64)
        )
        
        # Calculate feature map sizes after stem
        feat_size = input_size // 4  # 16 for 64x64 input
        
        # Stage 1: 64 channels, feat_size x feat_size
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(64, drop_path=0.0),
            ConvNeXtBlock(64, drop_path=0.05)
        )
        
        # Downsample 1: 64 -> 128 channels, reduce spatial by 2x
        self.downsample1 = nn.Sequential(
            LayerNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, stride=2)
        )
        
        # Stage 2: 128 channels, feat_size/2 x feat_size/2
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(128, drop_path=0.1),
            ConvNeXtBlock(128, drop_path=0.1)
        )
        
        # Downsample 2: 128 -> 256 channels, reduce spatial by 2x
        self.downsample2 = nn.Sequential(
            LayerNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=2, stride=2)
        )
        
        # Stage 3: 256 channels, feat_size/4 x feat_size/4
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(256, drop_path=0.15),
            ConvNeXtBlock(256, drop_path=0.15),
            ConvNeXtBlock(256, drop_path=0.2)
        )
        
        # Spatial attention before pooling
        self.attention = SpatialAttention()
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        # Final normalization and embedding
        self.norm = nn.LayerNorm(512)  # 256 from avg + 256 from max
        self.fc_embed = nn.Linear(512, embedding_dim)
        
        # Projection head for contrastive learning (2-layer MLP)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using modern best practices"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_projection=False):
        # Stem
        x = self.stem(x)
        
        # Stage 1
        x = self.stage1(x)
        x = self.downsample1(x)
        
        # Stage 2
        x = self.stage2(x)
        x = self.downsample2(x)
        
        # Stage 3 with attention
        x = self.stage3(x)
        x = self.attention(x)
        
        # Multi-scale pooling (avg + max for richer features)
        avg_feat = self.avgpool(x).flatten(1)
        max_feat = self.maxpool(x).flatten(1)
        x = torch.cat([avg_feat, max_feat], dim=1)
        
        # Embedding
        x = self.norm(x)
        embedding = self.fc_embed(x)
        
        if return_projection:
            projection = self.projection(embedding)
            return embedding, projection
        
        return embedding


# ===================================================================
# Correct Contrastive Loss (NT-Xent)
# ===================================================================

class NTXentLoss(nn.Module):
    """
    Standard and correct implementation of NT-Xent loss.
    """
    def __init__(self, temperature=0.07, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.device = torch.device(device)
        # Mask to remove positive_self (diagonal)
        # We create a placeholder mask and resize it dynamically in the forward pass
        # to handle the last batch which might be smaller.
        self.mask = self._create_mask(2 * 128) # Default for batch_size 128
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def _create_mask(self, batch_size):
        # Creates a boolean mask to remove diagonal elements
        return torch.eye(batch_size, dtype=torch.bool, device=self.device)

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        full_batch_size = 2 * batch_size
        
        # Resize mask if batch size changes (e.g., last batch)
        if self.mask.shape[0] != full_batch_size:
            self.mask = self._create_mask(full_batch_size)

        # L2 normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate representations [2*B, E]
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute cosine similarity matrix [2*B, 2*B]
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Get positive samples (z_i[k] with z_j[k])
        # These are on the off-diagonals
        sim_i_j = torch.diag(similarity_matrix, batch_size)
        sim_j_i = torch.diag(similarity_matrix, -batch_size)
        
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(full_batch_size, 1)
        
        # Get negative samples (all pairs except self and positive)
        # We use the mask to remove the diagonal (self-similarity)
        negative_samples = similarity_matrix[~self.mask].reshape(full_batch_size, -1)
        
        # Logits: [positive_sample, negative_samples_...]
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        
        # Apply temperature
        logits /= self.temperature
        
        # Labels are always 0, as the positive sample is always in the first column
        labels = torch.zeros(full_batch_size, dtype=torch.long, device=self.device)
        
        loss = self.criterion(logits, labels)
        return loss


# ===================================================================
# Enhanced Training Pipeline
# ===================================================================

class EnhancedEncoderTrainer:
    """Enhanced trainer with modern training techniques"""
    def __init__(self, cgr_dir="dataset/cgr", output_dir="models", 
                 embedding_dim=256, batch_size=128, num_epochs=100, 
                 learning_rate=1e-3, device='cuda'):
        self.cgr_dir = Path(cgr_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Enhanced Encoder Configuration:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Epochs: {num_epochs}")
        
        # Initialize Loss Function
        self.criterion = NTXentLoss(temperature=0.07, device=self.device)


    def load_data(self):
        logger.info("Loading combined dataset...")
        cgr_file = self.cgr_dir / "combined_cgr.npy"
        cgr_mem = np.load(cgr_file, mmap_mode='r')
        total = cgr_mem.shape[0]
        
        logger.info(f"Total sequences: {total:,}")
        logger.info(f"CGR shape: {cgr_mem.shape}")
        
        all_indices = np.arange(total)
        train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
        
        # augment=True for train, False for val
        train_ds = CGRDataset(str(cgr_file), train_idx, augment=True)
        val_ds = CGRDataset(str(cgr_file), val_idx, augment=False)
        
        logger.info(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")
        return train_ds, val_ds

    def train(self):
        logger.info("\n" + "="*70)
        logger.info("Training Enhanced CGR Encoder")
        logger.info("="*70)
        
        train_ds, val_ds = self.load_data()
        
        num_workers = min(os.cpu_count() // 2, 8)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True, persistent_workers=True,
            prefetch_factor=2, drop_last=True # drop_last=True helps stabilize batch size
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=num_workers, pin_memory=True, persistent_workers=True,
            prefetch_factor=2
        )
        
        # Initialize model
        model = EnhancedCGREncoder(self.embedding_dim).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Loss (already initialized in __init__)
        criterion = self.criterion
        
        # AdamW with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.05,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warmup
        warmup_epochs = 5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.num_epochs - warmup_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        # Training state
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        patience = 15
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Warmup learning rate
            if epoch < warmup_epochs:
                lr = self.learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Train
            model.train()
            train_loss = 0.0
            train_steps = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)
            
            # The loader now returns two augmented views directly
            for (aug1, aug2) in pbar:
                aug1 = aug1.to(self.device, non_blocking=True)
                aug2 = aug2.to(self.device, non_blocking=True)
                
                # Forward pass
                _, z_i = model(aug1, return_projection=True)
                _, z_j = model(aug2, return_projection=True)
                
                loss = criterion(z_i, z_j)
                
                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss /= train_steps
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                # val_loader returns a single, clean image
                for images in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    
                    # Create two *different* weak augmentations for validation loss
                    aug1 = torch.clamp(images + torch.randn_like(images) * 0.01, 0.0, 1.0)
                    aug2 = torch.clamp(images + torch.randn_like(images) * 0.01, 0.0, 1.0)
                    
                    _, z_i = model(aug1, return_projection=True)
                    _, z_j = model(aug2, return_projection=True)
                    
                    loss = criterion(z_i, z_j)
                    val_loss += loss.item()
                    val_steps += 1
            
            val_loss /= val_steps
            
            # Learning rate scheduling (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            # Cleanup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Logging
            logger.info(
                f"Epoch {epoch+1:3d}/{self.num_epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(model, 'best')
                logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Save final model
        self.save_model(model, 'final')
        
        # Save training history
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("\n" + "="*70)
        logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
        logger.info("="*70)
        
        return model, history

    def save_model(self, model, suffix):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'architecture': 'EnhancedConvNeXt',
            'dataset': 'combined_ITS_LSU_SSU'
        }
        torch.save(checkpoint, self.output_dir / f"cgr_encoder_{suffix}.pth")


def main():
    trainer = EnhancedEncoderTrainer(
        cgr_dir="dataset/cgr",
        output_dir="models",
        embedding_dim=256,
        batch_size=128,
        num_epochs=100,
        learning_rate=1e-3,
        device='cuda'
    )
    
    try:
        model, history = trainer.train()
        logger.info("\nTraining completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())