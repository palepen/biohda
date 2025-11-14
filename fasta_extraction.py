"""
Step 1: FASTA Extraction from BLAST Databases
==============================================
Extracts all sequences from NCBI BLAST databases using blastdbcmd

Dependencies:
- BLAST+ tools installed (blastdbcmd must be in PATH)
- Python 3.10+
"""

import subprocess
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FASTAExtractor:
    """Extract sequences from BLAST databases"""
    
    def __init__(self, blast_db_dir: str, output_dir: str = "dataset/raw"):
        """
        Initialize extractor
        
        Args:
            blast_db_dir: Directory containing BLAST database folders
            output_dir: Directory to save extracted FASTA files
        """
        self.blast_db_dir = Path(blast_db_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Database names mapping
        self.databases = {
            'ITS_eukaryote_sequences': 'ITS_eukaryote_sequences',
            'LSU_eukaryote_rRNA': 'LSU_eukaryote_rRNA',
            'SSU_eukaryote_rRNA': 'SSU_eukaryote_rRNA',
            '16S_ribosomal_RNA': '16S_ribosomal_RNA'
        }
    
    def check_blastdbcmd(self) -> bool:
        """Check if blastdbcmd is available"""
        try:
            result = subprocess.run(
                ['blastdbcmd', '-version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"BLAST+ found: {result.stdout.split()[1]}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("blastdbcmd not found. Please install BLAST+ tools.")
            return False
    
    def verify_database(self, db_path: str) -> bool:
        """
        Verify BLAST database exists and is valid
        
        Args:
            db_path: Path to BLAST database (without extension)
        
        Returns:
            True if database is valid
        """
        db_path = Path(db_path)
        required_files = ['.nhr', '.nin', '.nsq']
        
        for ext in required_files:
            if not (db_path.parent / f"{db_path.name}{ext}").exists():
                logger.warning(f"Missing required file: {db_path.name}{ext}")
                return False
        
        logger.info(f"Database verified: {db_path.name}")
        return True
    
    def extract_fasta(self, db_name: str, db_path: str, output_file: str) -> bool:
        """
        Extract all sequences from a BLAST database
        
        Args:
            db_name: Name of the database (for logging)
            db_path: Full path to BLAST database
            output_file: Output FASTA file path
        
        Returns:
            True if extraction successful
        """
        logger.info(f"Extracting {db_name}...")
        
        try:
            # blastdbcmd command to extract all sequences with full headers
            cmd = [
                'blastdbcmd',
                '-db', str(db_path),
                '-entry', 'all',
                '-outfmt', '%f',  # FASTA format with full header
                '-out', str(output_file)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Verify output file was created and has content
            output_path = Path(output_file)
            if output_path.exists() and output_path.stat().st_size > 0:
                # Count sequences
                with open(output_file, 'r') as f:
                    seq_count = sum(1 for line in f if line.startswith('>'))
                
                logger.info(f"Successfully extracted {seq_count:,} sequences to {output_file}")
                logger.info(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")
                return True
            else:
                logger.error(f"Output file is empty or not created: {output_file}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting {db_name}: {e}")
            logger.error(f"STDERR: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
    
    def extract_all(self) -> dict:
        """
        Extract all databases
        
        Returns:
            Dictionary with extraction status for each database
        """
        if not self.check_blastdbcmd():
            return {}
        
        results = {}
        
        for db_key, db_folder in self.databases.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {db_key}")
            logger.info(f"{'='*60}")
            
            # Construct full path to database
            db_path = self.blast_db_dir / db_folder / db_folder
            
            # Verify database exists
            if not self.verify_database(db_path):
                logger.error(f"Database not found or invalid: {db_path}")
                results[db_key] = False
                continue
            
            # Output FASTA file
            output_file = self.output_dir / f"{db_folder}.fasta"
            
            # Extract
            success = self.extract_fasta(db_key, str(db_path), str(output_file))
            results[db_key] = success
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("EXTRACTION SUMMARY")
        logger.info(f"{'='*60}")
        for db_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"{db_name}: {status}")
        
        return results


def main():
    """Main execution"""
    
    # Configure paths
    BLAST_DB_DIR = "dataset/raw"  # Change this to your BLAST database directory
    OUTPUT_DIR = "dataset/fasta"
    
    # Initialize extractor
    extractor = FASTAExtractor(BLAST_DB_DIR, OUTPUT_DIR)
    
    # Extract all databases
    results = extractor.extract_all()
    
    # Exit with appropriate code
    if all(results.values()):
        logger.info("\nAll extractions completed successfully!")
        return 0
    else:
        logger.error("\nSome extractions failed. Check logs above.")
        return 1


if __name__ == "__main__":
    exit(main())
