"""
Step 4: Chaos Game Representation (CGR)
========================================
Transform DNA sequences into 2D images using CGR algorithm

CGR maps each base to a corner:
- A: bottom-left
- T: top-left  
- G: bottom-right
- C: top-right

Each pixel represents frequency of k-mers at that location.

Dependencies:
- NumPy
- Biopython
- tqdm (progress bars)
"""

import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
from Bio import SeqIO
from tqdm import tqdm
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CGRTransformer:
    """Transform DNA sequences to CGR images"""
    
    def __init__(self, 
                 fasta_dir: str = "dataset/processed",
                 output_dir: str = "dataset/cgr",
                 image_size: int = 128,
                 log_transform: bool = True):
        """
        Initialize CGR transformer
        
        Args:
            fasta_dir: Directory with cleaned FASTA files
            output_dir: Directory to save CGR arrays
            image_size: Size of CGR image (image_size x image_size)
            log_transform: Apply log(1 + count) normalization
        """
        self.fasta_dir = Path(fasta_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_size = image_size
        self.log_transform = log_transform
        
        # Base coordinates (corners of unit square)
        self.base_coords = {
            'A': np.array([0.0, 0.0]),  # bottom-left
            'T': np.array([0.0, 1.0]),  # top-left
            'G': np.array([1.0, 0.0]),  # bottom-right
            'C': np.array([1.0, 1.0])   # top-right
        }
        
        # FASTA files
        self.fasta_files = [
            'ITS_clean.fasta',
            'LSU_clean.fasta',
            'SSU_clean.fasta'
        ]
    
    def sequence_to_cgr_coords(self, sequence: str) -> np.ndarray:
        """
        Convert DNA sequence to CGR coordinates
        
        Args:
            sequence: DNA sequence string (uppercase ATGC)
        
        Returns:
            Array of (x, y) coordinates, shape (seq_length, 2)
        """
        coords = []
        current_pos = np.array([0.5, 0.5])  # Start at center
        
        for base in sequence:
            if base in self.base_coords:
                # Move halfway toward the corner for this base
                corner = self.base_coords[base]
                current_pos = (current_pos + corner) / 2.0
                coords.append(current_pos.copy())
        
        return np.array(coords)
    
    def coords_to_image(self, coords: np.ndarray) -> np.ndarray:
        """
        Convert CGR coordinates to 2D frequency image
        
        Args:
            coords: Array of (x, y) coordinates, shape (N, 2)
        
        Returns:
            2D array (image_size, image_size) with frequency counts
        """
        # Initialize empty image
        image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Convert continuous coords to pixel indices
        pixel_coords = (coords * (self.image_size - 1)).astype(np.int32)
        
        # Clip to valid range (handle edge cases)
        pixel_coords = np.clip(pixel_coords, 0, self.image_size - 1)
        
        # Count frequency at each pixel
        for x, y in pixel_coords:
            image[y, x] += 1  # Note: image[row, col] = image[y, x]
        
        return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize CGR image
        
        Args:
            image: Raw frequency image
        
        Returns:
            Normalized image
        """
        if self.log_transform:
            # Log transform to compress dynamic range
            image = np.log1p(image)  # log(1 + x)
        
        # Normalize to [0, 1]
        max_val = image.max()
        if max_val > 0:
            image = image / max_val
        
        return image
    
    def sequence_to_cgr(self, sequence: str) -> np.ndarray:
        """
        Complete pipeline: sequence -> CGR image
        
        Args:
            sequence: DNA sequence string
        
        Returns:
            Normalized CGR image, shape (image_size, image_size)
        """
        # Filter sequence to only ATGC
        clean_seq = ''.join(base for base in sequence.upper() if base in 'ATGC')
        
        if len(clean_seq) == 0:
            # Return empty image if no valid bases
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Generate coordinates
        coords = self.sequence_to_cgr_coords(clean_seq)
        
        # Convert to image
        image = self.coords_to_image(coords)
        
        # Normalize
        image = self.normalize_image(image)
        
        return image
    
    def process_fasta_file(self, fasta_file: Path, marker: str) -> Tuple[np.ndarray, list]:
        """
        Process all sequences in a FASTA file
        
        Args:
            fasta_file: Path to FASTA file
            marker: Marker name (for progress bar)
        
        Returns:
            (cgr_images, seq_ids) tuple
            - cgr_images: array of shape (N, 1, image_size, image_size)
            - seq_ids: list of sequence IDs
        """
        logger.info(f"Processing {fasta_file.name}...")
        
        # First pass: count sequences
        seq_count = sum(1 for _ in SeqIO.parse(fasta_file, 'fasta'))
        logger.info(f"  Total sequences: {seq_count:,}")
        
        # Pre-allocate arrays
        cgr_images = np.zeros((seq_count, 1, self.image_size, self.image_size), 
                             dtype=np.float32)
        seq_ids = []
        
        # Second pass: generate CGRs
        logger.info(f"  Generating CGR images...")
        for idx, record in enumerate(tqdm(SeqIO.parse(fasta_file, 'fasta'),
                                         total=seq_count,
                                         desc=f"  {marker}",
                                         unit="seq")):
            # Extract seqID
            seqid = record.id
            seq_ids.append(seqid)
            
            # Generate CGR
            cgr = self.sequence_to_cgr(str(record.seq))
            cgr_images[idx, 0, :, :] = cgr
        
        logger.info(f"  ✓ Generated {seq_count:,} CGR images")
        
        return cgr_images, seq_ids
    
    def save_cgr_data(self, 
                     cgr_images: np.ndarray, 
                     seq_ids: list,
                     marker: str):
        """
        Save CGR images and metadata
        
        Args:
            cgr_images: Array of CGR images
            seq_ids: List of sequence IDs
            marker: Marker name
        """
        # Save CGR images as numpy array
        cgr_file = self.output_dir / f"{marker}_cgr.npy"
        np.save(cgr_file, cgr_images)
        logger.info(f"  Saved CGR images: {cgr_file}")
        logger.info(f"    Shape: {cgr_images.shape}")
        logger.info(f"    Size: {cgr_file.stat().st_size / (1024**2):.2f} MB")
        
        # Save sequence IDs
        ids_file = self.output_dir / f"{marker}_seqids.pkl"
        with open(ids_file, 'wb') as f:
            pickle.dump(seq_ids, f)
        logger.info(f"  Saved seq IDs: {ids_file}")
        
        # Save some statistics
        stats = {
            'num_sequences': len(seq_ids),
            'image_size': self.image_size,
            'log_transform': self.log_transform,
            'mean_intensity': float(cgr_images.mean()),
            'std_intensity': float(cgr_images.std()),
            'min_intensity': float(cgr_images.min()),
            'max_intensity': float(cgr_images.max())
        }
        
        stats_file = self.output_dir / f"{marker}_stats.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info(f"  Statistics:")
        logger.info(f"    Mean intensity: {stats['mean_intensity']:.4f}")
        logger.info(f"    Std intensity: {stats['std_intensity']:.4f}")
    
    def process_all(self):
        """Process all FASTA files"""
        logger.info(f"\n{'='*60}")
        logger.info("CGR TRANSFORMATION")
        logger.info(f"{'='*60}")
        logger.info(f"Image size: {self.image_size}x{self.image_size}")
        logger.info(f"Log transform: {self.log_transform}")
        logger.info(f"{'='*60}\n")
        
        all_cgr_images = []
        all_seq_ids = []
        all_markers = []
        
        for fasta_file in self.fasta_files:
            fasta_path = self.fasta_dir / fasta_file
            
            if not fasta_path.exists():
                logger.warning(f"File not found: {fasta_path}")
                continue
            
            # Extract marker name
            marker = fasta_file.split('_')[0]
            
            # Process file
            cgr_images, seq_ids = self.process_fasta_file(fasta_path, marker)
            
            # Save individual marker data
            self.save_cgr_data(cgr_images, seq_ids, marker)
            
            # Collect for combined dataset
            all_cgr_images.append(cgr_images)
            all_seq_ids.extend(seq_ids)
            all_markers.extend([marker] * len(seq_ids))
            
            logger.info("")  # Blank line
        
        # Save combined dataset
        if all_cgr_images:
            logger.info(f"{'='*60}")
            logger.info("Saving combined dataset...")
            
            combined_cgr = np.vstack(all_cgr_images)
            
            # Save combined CGR
            combined_file = self.output_dir / "combined_cgr.npy"
            np.save(combined_file, combined_cgr)
            logger.info(f"✓ Combined CGR: {combined_file}")
            logger.info(f"  Shape: {combined_cgr.shape}")
            logger.info(f"  Size: {combined_file.stat().st_size / (1024**2):.2f} MB")
            
            # Save combined metadata
            metadata = {
                'seq_ids': all_seq_ids,
                'markers': all_markers
            }
            metadata_file = self.output_dir / "combined_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"✓ Combined metadata: {metadata_file}")
            
            # Summary
            logger.info(f"\n{'='*60}")
            logger.info("CGR TRANSFORMATION SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Total sequences: {len(all_seq_ids):,}")
            for marker in ['ITS', 'LSU', 'SSU']:
                count = all_markers.count(marker)
                if count > 0:
                    logger.info(f"  {marker}: {count:,}")
            logger.info(f"Image dimensions: {self.image_size}x{self.image_size}")
            logger.info(f"Total storage: {combined_file.stat().st_size / (1024**2):.2f} MB")
            logger.info(f"{'='*60}")


def main():
    """Main execution"""
    
    # Initialize transformer
    transformer = CGRTransformer(
        fasta_dir="dataset/processed",
        output_dir="dataset/cgr",
        image_size=128,      # 128x128 images
        log_transform=True   # Apply log normalization
    )
    
    # Process all files
    transformer.process_all()
    
    logger.info("\n✓ CGR transformation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
