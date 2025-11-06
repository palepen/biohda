"""
Step 6: Generate Embeddings
============================
Use trained CNN encoder to generate embeddings for all sequences

Output:
- embeddings.npy: (N, 128) array of embeddings
- embedding_metadata.pkl: corresponding seqIDs and markers

Dependencies:
- PyTorch
- NumPy
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CGREncoder(nn.Module):
    """CNN Encoder for CGR images (must match training architecture)"""
    
    def __init__(self, embedding_dim: int = 128, input_size: int = 128):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Projection to embedding space
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


class EmbeddingGenerator:
    """Generate embeddings from trained encoder"""
    
    def __init__(self,
                 model_path: str = "models/models/cgr_autoencoder_encoder_best.pth",
                 cgr_dir: str = "dataset/cgr",
                 output_dir: str = "dataset/embeddings",
                 batch_size: int = 512,
                 device: str = 'cuda'):
        """
        Initialize embedding generator
        
        Args:
            model_path: Path to trained model checkpoint
            cgr_dir: Directory with CGR images
            output_dir: Directory to save embeddings
            batch_size: Batch size for inference
            device: Device to use (cuda/cpu)
        """
        self.model_path = Path(model_path)
        self.cgr_dir = Path(cgr_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model()
    
    def load_model(self) -> nn.Module:
        """Load trained encoder model"""
        logger.info(f"Loading model from {self.model_path}...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize model
        embedding_dim = checkpoint.get('embedding_dim', 128)
        model = CGREncoder(embedding_dim=embedding_dim)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Embedding dimension: {embedding_dim}")
        logger.info(f"  Dataset: {checkpoint.get('dataset', 'unknown')}")
        
        return model
    
    def generate_embeddings_batch(self, cgr_batch: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for a batch of CGR images
        
        Args:
            cgr_batch: Array of shape (batch_size, 1, H, W)
        
        Returns:
            Embeddings array of shape (batch_size, embedding_dim)
        """
        # Convert to tensor
        cgr_tensor = torch.from_numpy(cgr_batch).float().to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(cgr_tensor)
        
        # Convert back to numpy
        return embeddings.cpu().numpy()
    
    def generate_embeddings_for_dataset(self) -> Tuple[np.ndarray, Dict]:
        """
        Generate embeddings for entire combined dataset
        
        Returns:
            (embeddings, metadata) tuple
        """
        logger.info(f"\n{'='*60}")
        logger.info("Generating Embeddings for Combined Dataset")
        logger.info(f"{'='*60}\n")
        
        # Load combined CGR data
        cgr_file = self.cgr_dir / "combined_cgr.npy"
        if not cgr_file.exists():
            raise FileNotFoundError(f"Combined CGR not found: {cgr_file}")
        
        logger.info(f"Loading CGR data from {cgr_file}...")
        cgr_images = np.load(cgr_file, mmap_mode='r')
        num_sequences = cgr_images.shape[0]
        
        logger.info(f"  Shape: {cgr_images.shape}")
        logger.info(f"  Total sequences: {num_sequences:,}")
        
        # Load metadata
        metadata_file = self.cgr_dir / "combined_metadata.pkl"
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        seq_ids = metadata['seq_ids']
        markers = metadata['markers']
        
        logger.info(f"  Sequence IDs: {len(seq_ids):,}")
        logger.info(f"\nMarker distribution:")
        for marker in ['ITS', 'LSU', 'SSU']:
            count = markers.count(marker)
            logger.info(f"  {marker}: {count:,} ({count/len(markers)*100:.1f}%)")
        
        # Initialize embeddings array
        embedding_dim = self.model.embedding_dim
        embeddings = np.zeros((num_sequences, embedding_dim), dtype=np.float32)
        
        # Generate embeddings in batches
        logger.info(f"\nGenerating embeddings (batch_size={self.batch_size})...")
        num_batches = (num_sequences + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, num_sequences, self.batch_size), 
                     total=num_batches,
                     desc="Processing batches"):
            
            # Get batch
            end_idx = min(i + self.batch_size, num_sequences)
            batch = cgr_images[i:end_idx]
            
            # Generate embeddings
            batch_embeddings = self.generate_embeddings_batch(batch)
            embeddings[i:end_idx] = batch_embeddings
        
        logger.info(f"✓ Generated embeddings for {num_sequences:,} sequences")
        
        # Prepare metadata
        embedding_metadata = {
            'seq_ids': seq_ids,
            'markers': markers,
            'embedding_dim': embedding_dim,
            'num_sequences': num_sequences
        }
        
        return embeddings, embedding_metadata
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: Dict):
        """
        Save embeddings and metadata
        
        Args:
            embeddings: Embeddings array (N, embedding_dim)
            metadata: Metadata dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info("Saving Embeddings")
        logger.info(f"{'='*60}")
        
        # Save embeddings
        embeddings_file = self.output_dir / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        logger.info(f"✓ Embeddings saved: {embeddings_file}")
        logger.info(f"  Shape: {embeddings.shape}")
        logger.info(f"  Size: {embeddings_file.stat().st_size / (1024**2):.2f} MB")
        
        # Save metadata
        metadata_file = self.output_dir / "embedding_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✓ Metadata saved: {metadata_file}")
        
        # Compute and save statistics
        stats = {
            'mean': embeddings.mean(axis=0),
            'std': embeddings.std(axis=0),
            'min': embeddings.min(axis=0),
            'max': embeddings.max(axis=0),
            'embedding_dim': embeddings.shape[1],
            'num_sequences': embeddings.shape[0]
        }
        
        stats_file = self.output_dir / "embedding_stats.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info(f"✓ Statistics saved: {stats_file}")
        
        # Print statistics
        logger.info(f"\nEmbedding Statistics:")
        logger.info(f"  Dimension: {stats['embedding_dim']}")
        logger.info(f"  Number of sequences: {stats['num_sequences']:,}")
        logger.info(f"  Mean norm: {np.linalg.norm(stats['mean']):.4f}")
        logger.info(f"  Std norm: {np.linalg.norm(stats['std']):.4f}")
        logger.info(f"  Min value: {stats['min'].min():.4f}")
        logger.info(f"  Max value: {stats['max'].max():.4f}")
    
    def generate_and_save(self):
        """Complete pipeline: generate and save embeddings"""
        # Generate embeddings
        embeddings, metadata = self.generate_embeddings_for_dataset()
        
        # Save results
        self.save_embeddings(embeddings, metadata)
        
        logger.info(f"\n{'='*60}")
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Files created:")
        logger.info(f"  - embeddings.npy")
        logger.info(f"  - embedding_metadata.pkl")
        logger.info(f"  - embedding_stats.pkl")
        logger.info(f"{'='*60}\n")


def main():
    """Main execution"""
    
    # Check if model exists
    model_path = Path("models/cgr_encoder_best.pth")
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please train the model first (Step 5)")
        return 1
    
    # Initialize generator
    generator = EmbeddingGenerator(
        model_path="models/cgr_encoder_best.pth",
        cgr_dir="dataset/cgr",
        output_dir="dataset/embeddings",
        batch_size=512,  # Adjust based on GPU memory
        device='cuda'
    )
    
    try:
        # Generate and save embeddings
        generator.generate_and_save()
        
        logger.info("✓ Embedding generation complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
