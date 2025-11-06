"""
Step 2: Minimal Preprocessing
==============================
Clean FASTA sequences while preserving biological diversity

Preprocessing steps:
1. Convert to uppercase
2. Remove sequences with >10% ambiguous bases
3. Keep original orientation (no reverse complement)
4. Retain all duplicates
5. Discard sequences <100 bp

Dependencies:
- Biopython
"""

from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path
import logging
from typing import Dict, Tuple
from collections import Counter
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinimalPreprocessor:
    """Minimal preprocessing for FASTA sequences"""
    
    def __init__(self, 
                 input_dir: str = "dataset/fasta",
                 output_dir: str = "dataset/processed",
                 min_length: int = 100,
                 max_ambiguous_ratio: float = 0.10):
        """
        Initialize preprocessor
        
        Args:
            input_dir: Directory with raw FASTA files
            output_dir: Directory for cleaned FASTA files
            min_length: Minimum sequence length (bp)
            max_ambiguous_ratio: Maximum ratio of ambiguous bases (N, etc.)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_length = min_length
        self.max_ambiguous_ratio = max_ambiguous_ratio
        
        # Standard nucleotides (IUPAC codes)
        self.standard_bases = set('ATGC')
        self.ambiguous_bases = set('NRYWSMKHBVD')  # All IUPAC ambiguous codes
        
        # File mapping
        self.file_mapping = {
            'ITS_eukaryote_sequences.fasta': 'ITS_clean.fasta',
            'LSU_eukaryote_rRNA.fasta': 'LSU_clean.fasta',
            'SSU_eukaryote_rRNA.fasta': 'SSU_clean.fasta'
        }
    
    def count_ambiguous_bases(self, sequence: str) -> Tuple[int, float]:
        """
        Count ambiguous bases in sequence
        
        Args:
            sequence: DNA sequence string
        
        Returns:
            (count, ratio) of ambiguous bases
        """
        seq_upper = sequence.upper()
        ambiguous_count = sum(1 for base in seq_upper 
                            if base not in self.standard_bases)
        ratio = ambiguous_count / len(sequence) if len(sequence) > 0 else 0
        return ambiguous_count, ratio
    
    def is_valid_sequence(self, sequence: str) -> Tuple[bool, str]:
        """
        Check if sequence passes quality filters
        
        Args:
            sequence: DNA sequence string
        
        Returns:
            (is_valid, rejection_reason)
        """
        seq_upper = sequence.upper()
        
        # Check minimum length
        if len(seq_upper) < self.min_length:
            return False, f"too_short_{len(seq_upper)}bp"
        
        # Check ambiguous base ratio
        ambig_count, ambig_ratio = self.count_ambiguous_bases(seq_upper)
        if ambig_ratio > self.max_ambiguous_ratio:
            return False, f"high_ambiguous_{ambig_ratio:.2%}"
        
        return True, "passed"
    
    def process_fasta_file(self, input_file: Path, output_file: Path) -> Dict:
        """
        Process a single FASTA file
        
        Args:
            input_file: Input FASTA file path
            output_file: Output cleaned FASTA file path
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing: {input_file.name}")
        
        stats = {
            'total_sequences': 0,
            'kept_sequences': 0,
            'rejected_too_short': 0,
            'rejected_high_ambiguous': 0,
            'total_bp_kept': 0,
            'min_length_kept': float('inf'),
            'max_length_kept': 0,
            'rejection_reasons': Counter()
        }
        
        try:
            with open(output_file, 'w') as out_handle:
                for record in SeqIO.parse(input_file, 'fasta'):
                    stats['total_sequences'] += 1
                    
                    # Get sequence string
                    seq_str = str(record.seq)
                    
                    # Validate sequence
                    is_valid, reason = self.is_valid_sequence(seq_str)
                    
                    if is_valid:
                        # Convert to uppercase
                        clean_seq = seq_str.upper()
                        
                        # Update record with cleaned sequence
                        record.seq = Seq(clean_seq)
                        
                        # Write to output
                        SeqIO.write(record, out_handle, 'fasta')
                        
                        # Update statistics
                        stats['kept_sequences'] += 1
                        stats['total_bp_kept'] += len(clean_seq)
                        stats['min_length_kept'] = min(stats['min_length_kept'], 
                                                       len(clean_seq))
                        stats['max_length_kept'] = max(stats['max_length_kept'], 
                                                       len(clean_seq))
                    else:
                        # Track rejection reason
                        stats['rejection_reasons'][reason] += 1
                        
                        if 'too_short' in reason:
                            stats['rejected_too_short'] += 1
                        elif 'high_ambiguous' in reason:
                            stats['rejected_high_ambiguous'] += 1
            
            # Calculate additional statistics
            if stats['kept_sequences'] > 0:
                stats['avg_length_kept'] = stats['total_bp_kept'] / stats['kept_sequences']
                stats['retention_rate'] = stats['kept_sequences'] / stats['total_sequences']
            else:
                stats['avg_length_kept'] = 0
                stats['retention_rate'] = 0
            
            logger.info(f"  Total sequences: {stats['total_sequences']:,}")
            logger.info(f"  Kept sequences: {stats['kept_sequences']:,} "
                       f"({stats['retention_rate']:.1%})")
            logger.info(f"  Rejected (too short): {stats['rejected_too_short']:,}")
            logger.info(f"  Rejected (high ambiguous): {stats['rejected_high_ambiguous']:,}")
            
            if stats['kept_sequences'] > 0:
                logger.info(f"  Length range: {stats['min_length_kept']:,} - "
                           f"{stats['max_length_kept']:,} bp")
                logger.info(f"  Average length: {stats['avg_length_kept']:.0f} bp")
            
            return stats
            
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            return stats
        except Exception as e:
            logger.error(f"Error processing {input_file.name}: {e}")
            return stats
    
    def process_all(self) -> Dict[str, Dict]:
        """
        Process all FASTA files
        
        Returns:
            Dictionary mapping filenames to their statistics
        """
        all_stats = {}
        
        logger.info(f"\n{'='*60}")
        logger.info("MINIMAL PREPROCESSING")
        logger.info(f"{'='*60}")
        logger.info(f"Min length threshold: {self.min_length} bp")
        logger.info(f"Max ambiguous bases: {self.max_ambiguous_ratio:.0%}")
        logger.info(f"{'='*60}\n")
        
        for input_filename, output_filename in self.file_mapping.items():
            input_file = self.input_dir / input_filename
            output_file = self.output_dir / output_filename
            
            if not input_file.exists():
                logger.warning(f"Input file not found: {input_file}")
                continue
            
            # Process file
            stats = self.process_fasta_file(input_file, output_file)
            all_stats[input_filename] = stats
            
            logger.info("")  # Blank line between files
        
        # Overall summary
        self.print_summary(all_stats)
        
        return all_stats
    
    def print_summary(self, all_stats: Dict[str, Dict]):
        """Print overall processing summary"""
        logger.info(f"{'='*60}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        
        total_input = sum(s['total_sequences'] for s in all_stats.values())
        total_kept = sum(s['kept_sequences'] for s in all_stats.values())
        total_rejected_short = sum(s['rejected_too_short'] for s in all_stats.values())
        total_rejected_ambig = sum(s['rejected_high_ambiguous'] for s in all_stats.values())
        
        logger.info(f"Total input sequences: {total_input:,}")
        logger.info(f"Total kept sequences: {total_kept:,} "
                   f"({total_kept/total_input:.1%})")
        logger.info(f"Total rejected (too short): {total_rejected_short:,}")
        logger.info(f"Total rejected (ambiguous): {total_rejected_ambig:,}")
        logger.info(f"{'='*60}\n")
        
        # Per-file summary table
        logger.info("Per-file retention rates:")
        for filename, stats in all_stats.items():
            if stats['total_sequences'] > 0:
                marker = filename.split('_')[0]
                retention = stats['retention_rate']
                logger.info(f"  {marker:8s}: {stats['kept_sequences']:>8,} / "
                           f"{stats['total_sequences']:>8,} ({retention:>6.1%})")


def main():
    """Main execution"""
    
    # Configure preprocessing
    preprocessor = MinimalPreprocessor(
        input_dir="dataset/fasta",
        output_dir="dataset/processed",
        min_length=100,           # Minimum 100 bp
        max_ambiguous_ratio=0.10  # Maximum 10% ambiguous bases
    )
    
    # Process all files
    stats = preprocessor.process_all()
    
    # Check success
    if all(s['kept_sequences'] > 0 for s in stats.values()):
        logger.info("✓ All files processed successfully!")
        return 0
    else:
        logger.warning("⚠ Some files had no sequences passing filters")
        return 1


if __name__ == "__main__":
    exit(main())
