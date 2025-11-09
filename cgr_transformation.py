"""
Step 4: Chaos Game Representation (CGR)
Transform DNA sequences into 2D images using CGR algorithm
"""

import numpy as np
from pathlib import Path
import logging
from Bio import SeqIO
from tqdm import tqdm
import pickle
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CGRTransformer:
    """Transform DNA sequences to CGR images"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.fasta_dir = Path(config['paths']['processed_dir'])
        self.output_dir = Path(config['paths']['cgr_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_size = config['cgr']['image_size']
        self.log_transform = config['cgr']['log_transform']
        
        self.base_coords = {
            'A': np.array([0.0, 0.0]),
            'T': np.array([0.0, 1.0]),
            'G': np.array([1.0, 0.0]),
            'C': np.array([1.0, 1.0])
        }
        
        self.fasta_files = ['SSU_clean.fasta', 'LSU_clean.fasta', 'ITS_clean.fasta', '16S_clean.fasta']
    
    def sequence_to_cgr_coords(self, sequence: str) -> np.ndarray:
        """Convert DNA sequence to CGR coordinates"""
        coords = []
        current_pos = np.array([0.5, 0.5])
        
        for base in sequence:
            if base in self.base_coords:
                corner = self.base_coords[base]
                current_pos = (current_pos + corner) / 2.0
                coords.append(current_pos.copy())
        
        return np.array(coords)
    
    def coords_to_image(self, coords: np.ndarray) -> np.ndarray:
        """Convert CGR coordinates to 2D frequency image"""
        image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        pixel_coords = (coords * (self.image_size - 1)).astype(np.int32)
        pixel_coords = np.clip(pixel_coords, 0, self.image_size - 1)
        
        for x, y in pixel_coords:
            image[y, x] += 1
        
        return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize CGR image"""
        if self.log_transform:
            image = np.log1p(image)
        
        max_val = image.max()
        if max_val > 0:
            image = image / max_val
        
        return image
    
    def sequence_to_cgr(self, sequence: str) -> np.ndarray:
        """Complete pipeline: sequence -> CGR image"""
        clean_seq = ''.join(base for base in sequence.upper() if base in 'ATGC')
        
        if len(clean_seq) == 0:
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        coords = self.sequence_to_cgr_coords(clean_seq)
        image = self.coords_to_image(coords)
        image = self.normalize_image(image)
        
        return image
    
    def process_fasta_file(self, fasta_file: Path, marker: str):
        """Process all sequences in a FASTA file"""
        logger.info(f"Processing {fasta_file.name}...")
        
        seq_count = sum(1 for _ in SeqIO.parse(fasta_file, 'fasta'))
        logger.info(f"Total sequences: {seq_count:,}")
        
        cgr_images = np.zeros((seq_count, 1, self.image_size, self.image_size), dtype=np.float32)
        seq_ids = []
        
        logger.info("Generating CGR images...")
        for idx, record in enumerate(tqdm(SeqIO.parse(fasta_file, 'fasta'), total=seq_count, desc=f"  {marker}")):
            seq_ids.append(record.id)
            cgr = self.sequence_to_cgr(str(record.seq))
            cgr_images[idx, 0, :, :] = cgr
        
        logger.info(f"Generated {seq_count:,} CGR images")
        
        return cgr_images, seq_ids
    
    def save_cgr_data(self, cgr_images: np.ndarray, seq_ids: list, marker: str):
        """Save CGR images and metadata"""
        cgr_file = self.output_dir / f"{marker}_cgr.npy"
        np.save(cgr_file, cgr_images)
        logger.info(f"Saved: {cgr_file} ({cgr_file.stat().st_size / (1024**2):.2f} MB)")
        
        ids_file = self.output_dir / f"{marker}_seqids.pkl"
        with open(ids_file, 'wb') as f:
            pickle.dump(seq_ids, f)
        logger.info(f"Saved: {ids_file}")
        
        stats = {
            'num_sequences': len(seq_ids),
            'image_size': self.image_size,
            'log_transform': self.log_transform,
            'mean_intensity': float(cgr_images.mean()),
            'std_intensity': float(cgr_images.std())
        }
        
        stats_file = self.output_dir / f"{marker}_stats.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
    
    def process_all(self):
        """Process all FASTA files"""
        logger.info("\nCGR TRANSFORMATION")
        logger.info(f"Image size: {self.image_size}x{self.image_size}")
        logger.info(f"Log transform: {self.log_transform}\n")
        
        all_cgr_images = []
        all_seq_ids = []
        all_markers = []
        
        for fasta_file in self.fasta_files:
            fasta_path = self.fasta_dir / fasta_file
            
            if not fasta_path.exists():
                logger.warning(f"File not found: {fasta_path}")
                continue
            
            marker = fasta_file.split('_')[0]
            cgr_images, seq_ids = self.process_fasta_file(fasta_path, marker)
            self.save_cgr_data(cgr_images, seq_ids, marker)
            
            all_cgr_images.append(cgr_images)
            all_seq_ids.extend(seq_ids)
            all_markers.extend([marker] * len(seq_ids))
        
        if all_cgr_images:
            logger.info("\nSaving combined dataset...")
            
            combined_cgr = np.vstack(all_cgr_images)
            combined_file = self.output_dir / "combined_cgr.npy"
            np.save(combined_file, combined_cgr)
            logger.info(f"Combined CGR: {combined_file} ({combined_file.stat().st_size / (1024**2):.2f} MB)")
            
            metadata = {'seq_ids': all_seq_ids, 'markers': all_markers}
            metadata_file = self.output_dir / "combined_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"Combined metadata: {metadata_file}")
            
            logger.info(f"\nTotal sequences: {len(all_seq_ids):,}")
            for marker in set(all_markers):
                count = all_markers.count(marker)
                logger.info(f"  {marker}: {count:,}")


def main():
    transformer = CGRTransformer(config_path="config.yaml")
    transformer.process_all()
    logger.info("\nCGR transformation complete")
    return 0


if __name__ == "__main__":
    exit(main())