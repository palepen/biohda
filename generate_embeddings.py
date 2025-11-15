import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===================================================================
# Model Architecture Definitions (must match training)
# ===================================================================

class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps"""
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
    """Stochastic depth"""
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


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block"""
    def __init__(self, dim, drop_path=0., layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
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


class SpatialAttention(nn.Module):
    """Spatial attention module"""
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


class EnhancedCGREncoder(nn.Module):
    """Enhanced CNN encoder (matches training script)"""
    def __init__(self, embedding_dim=256, input_size=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=4),
            LayerNorm2d(64)
        )
        
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(64, drop_path=0.0),
            ConvNeXtBlock(64, drop_path=0.05)
        )
        
        self.downsample1 = nn.Sequential(
            LayerNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, stride=2)
        )
        
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(128, drop_path=0.1),
            ConvNeXtBlock(128, drop_path=0.1)
        )
        
        self.downsample2 = nn.Sequential(
            LayerNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=2, stride=2)
        )
        
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(256, drop_path=0.15),
            ConvNeXtBlock(256, drop_path=0.15),
            ConvNeXtBlock(256, drop_path=0.2)
        )
        
        self.attention = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.norm = nn.LayerNorm(512)
        self.fc_embed = nn.Linear(512, embedding_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward(self, x, return_projection=False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample1(x)
        x = self.stage2(x)
        x = self.downsample2(x)
        x = self.stage3(x)
        x = self.attention(x)
        
        avg_feat = self.avgpool(x).flatten(1)
        max_feat = self.maxpool(x).flatten(1)
        x = torch.cat([avg_feat, max_feat], dim=1)
        
        x = self.norm(x)
        embedding = self.fc_embed(x)
        
        if return_projection:
            projection = self.projection(embedding)
            return embedding, projection
        
        return embedding


class OldCGREncoder(nn.Module):
    """Original simple CNN encoder (for backward compatibility)"""
    def __init__(self, embedding_dim=128):
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


# ===================================================================
# Dataset
# ===================================================================

class CGRDataset(Dataset):
    """Memory-mapped CGR dataset for inference"""
    def __init__(self, cgr_path: str):
        self.cgr = np.load(cgr_path, mmap_mode='r')
        self._needs_unsqueeze = (self.cgr.ndim == 3)

    def __len__(self):
        return self.cgr.shape[0]

    def __getitem__(self, idx):
        arr = self.cgr[idx]
        
        if self._needs_unsqueeze:
            arr = np.expand_dims(arr, axis=0)
        
        image = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).float()
        return image


# ===================================================================
# Embedding Generator
# ===================================================================

class EmbeddingGenerator:
    """
    Generates embeddings using trained encoder.
    Automatically detects model architecture and embedding dimension.
    """
    def __init__(self, cgr_dir="dataset/cgr", model_path="models/cgr_encoder_best.pth",
                 output_dir="dataset/embeddings", batch_size=256, device='cuda'):
        self.cgr_dir = Path(cgr_dir)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model_path}")

    def load_model(self):
        """Load model and automatically detect architecture"""
        logger.info("Loading model checkpoint...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model info
        embedding_dim = checkpoint.get('embedding_dim', 128)
        architecture = checkpoint.get('architecture', 'OldCNN')
        
        logger.info(f"  Architecture: {architecture}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        
        # Create appropriate model
        if architecture == 'EnhancedConvNeXt' or embedding_dim > 128:
            logger.info("  Loading EnhancedCGREncoder...")
            model = EnhancedCGREncoder(embedding_dim=embedding_dim)
        else:
            logger.info("  Loading OldCGREncoder...")
            model = OldCGREncoder(embedding_dim=embedding_dim)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # --- FIX 1: Return the checkpoint ---
        return model, embedding_dim, checkpoint

    def generate(self):
        """Generate embeddings for all sequences"""
        logger.info("\n" + "="*70)
        logger.info("Generating Embeddings")
        logger.info("="*70)
        
        # --- FIX 2: Receive the checkpoint ---
        model, embedding_dim, checkpoint = self.load_model()
        
        # Load CGR data
        cgr_file = self.cgr_dir / "combined_cgr.npy"
        if not cgr_file.exists():
            raise FileNotFoundError(f"CGR file not found: {cgr_file}")
        
        logger.info(f"\nLoading CGR data from: {cgr_file}")
        dataset = CGRDataset(str(cgr_file))
        logger.info(f"Total sequences: {len(dataset):,}")
        
        num_workers = min(os.cpu_count() // 2, 8)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True,
            persistent_workers=True
        )
        
        # Generate embeddings
        logger.info(f"\nGenerating {embedding_dim}-dimensional embeddings...")
        all_embeddings = []
        
        with torch.no_grad():
            for images in tqdm(dataloader, desc="Processing batches"):
                images = images.to(self.device, non_blocking=True)
                embeddings = model(images, return_projection=False)
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        
        logger.info(f"\nEmbeddings shape: {embeddings_array.shape}")
        logger.info(f"Memory usage: {embeddings_array.nbytes / (1024**2):.1f} MB")
        
        # Save embeddings
        output_file = self.output_dir / "embeddings.npy"
        np.save(output_file, embeddings_array)
        logger.info(f"Saved embeddings to: {output_file}")
        
        # Load and save metadata
        metadata_file = self.cgr_dir / "metadata.pkl"
        output_metadata = self.output_dir / "metadata.pkl" # Define here for logging
        
        if metadata_file.exists():
            logger.info(f"\nCopying metadata from: {metadata_file}")
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            with open(output_metadata, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"Saved metadata to: {output_metadata}")
            
            # Log statistics
            logger.info(f"\nMetadata statistics:")
            logger.info(f"  Total sequences: {len(metadata):,}")
            
            # Count taxonomic levels
            taxa_counts = {}
            for meta in metadata:
                if 'taxonomy' in meta:
                    for level in ['genus', 'family', 'order', 'class', 'phylum']:
                        if level in meta['taxonomy'] and meta['taxonomy'][level]:
                            taxa_counts[level] = taxa_counts.get(level, 0) + 1
            
            logger.info(f"  Sequences with taxonomy:")
            for level in ['genus', 'family', 'order', 'class', 'phylum']:
                count = taxa_counts.get(level, 0)
                pct = 100 * count / len(metadata) if len(metadata) > 0 else 0
                logger.info(f"    {level.capitalize()}: {count:,} ({pct:.1f}%)")
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
        
        # Save generation info
        # Now 'checkpoint' is defined in this scope
        info = {
            'embedding_dim': embedding_dim,
            'num_sequences': len(dataset),
            'model_path': str(self.model_path),
            'architecture': checkpoint.get('architecture', 'Unknown'),
            'batch_size': self.batch_size,
            'device': str(self.device)
        }
        
        info_file = self.output_dir / "generation_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info("\n" + "="*70)
        logger.info("Embedding generation complete!")
        logger.info("="*70)
        logger.info(f"\nOutput files:")
        logger.info(f"  Embeddings: {output_file}")
        logger.info(f"  Metadata: {output_metadata if metadata_file.exists() else 'N/A'}")
        logger.info(f"  Info: {info_file}")
        
        return embeddings_array


def main():
    generator = EmbeddingGenerator(
        cgr_dir="dataset/cgr",
        model_path="models/cgr_encoder_best.pth",
        output_dir="dataset/embeddings",
        batch_size=256,
        device='cuda'
    )
    
    try:
        embeddings = generator.generate()
        logger.info("\nSuccess! Ready for clustering.")
        return 0
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())