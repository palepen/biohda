"""
Step 6: Generate Embeddings (Fixed)
====================================
Loads trained encoder and generates embeddings for all sequences.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CGREncoder(nn.Module):
    """Must match training architecture exactly"""
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
        
        # Projection head (not used for embeddings)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.fc_embed(x)


class EmbeddingGenerator:
    """Generate embeddings using trained encoder"""
    def __init__(self, model_path="models/cgr_encoder_best.pth", cgr_dir="dataset/cgr",
                 output_dir="dataset/embeddings", batch_size=512, device='cuda'):
        self.model_path = Path(model_path)
        self.cgr_dir = Path(cgr_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Device: {self.device}")
        self.model = self.load_model()

    def load_model(self):
        logger.info(f"Loading encoder from {self.model_path}...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        embedding_dim = checkpoint.get('embedding_dim', 128)
        
        model = CGREncoder(embedding_dim=embedding_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded. Embedding dim: {embedding_dim}")
        return model

    def generate_batch(self, cgr_batch):
        cgr_tensor = torch.from_numpy(cgr_batch).float().to(self.device)
        with torch.no_grad():
            embeddings = self.model(cgr_tensor)
        return embeddings.cpu().numpy()

    def generate_embeddings(self):
        logger.info("\nGenerating embeddings...")
        
        cgr_file = self.cgr_dir / "combined_cgr.npy"
        cgr_images = np.load(cgr_file, mmap_mode='r')
        num_seq = cgr_images.shape[0]
        
        logger.info(f"Shape: {cgr_images.shape}, Sequences: {num_seq:,}")
        
        with open(self.cgr_dir / "combined_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        seq_ids = metadata['seq_ids']
        markers = metadata['markers']
        
        embedding_dim = self.model.embedding_dim
        embeddings = np.zeros((num_seq, embedding_dim), dtype=np.float32)
        
        num_batches = (num_seq + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, num_seq, self.batch_size), total=num_batches, desc="Processing"):
            end = min(i + self.batch_size, num_seq)
            batch = cgr_images[i:end]
            embeddings[i:end] = self.generate_batch(batch)
        
        logger.info(f"Generated embeddings: {embeddings.shape}")
        
        metadata = {
            'seq_ids': seq_ids,
            'markers': markers,
            'embedding_dim': embedding_dim,
            'num_sequences': num_seq
        }
        
        return embeddings, metadata

    def save_embeddings(self, embeddings, metadata):
        logger.info("\nSaving embeddings...")
        
        emb_file = self.output_dir / "embeddings.npy"
        np.save(emb_file, embeddings)
        logger.info(f"Saved: {emb_file} ({emb_file.stat().st_size / (1024**2):.2f} MB)")
        
        meta_file = self.output_dir / "embedding_metadata.pkl"
        with open(meta_file, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved: {meta_file}")
        
        stats = {
            'mean': embeddings.mean(axis=0),
            'std': embeddings.std(axis=0),
            'embedding_dim': embeddings.shape[1],
            'num_sequences': embeddings.shape[0]
        }
        
        with open(self.output_dir / "embedding_stats.pkl", 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info(f"Stats: mean_norm={np.linalg.norm(stats['mean']):.4f}")

    def run(self):
        embeddings, metadata = self.generate_embeddings()
        self.save_embeddings(embeddings, metadata)
        logger.info("\nEmbedding generation complete!")


def main():
    generator = EmbeddingGenerator(
        model_path="models/cgr_encoder_best.pth",
        cgr_dir="dataset/cgr",
        output_dir="dataset/embeddings",
        batch_size=512,
        device='cuda'
    )
    
    try:
        generator.run()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())