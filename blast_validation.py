"""
Step 9: Online BLAST Validation
================================
Validate candidate novel taxa using NCBI online BLAST

This version uses NCBI's online BLAST service (NCBIWWW) instead of local databases.
Rate-limited to avoid overloading NCBI servers.

Dependencies:
- Biopython (Bio.Blast.NCBIWWW, Bio.Blast.NCBIXML)
"""

import time
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast import NCBIWWW, NCBIXML
import pandas as pd
import numpy as np
from collections import Counter
import json
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OnlineBLASTValidator:
    """Online BLAST-based validation of candidate novel taxa"""
    
    def __init__(self,
                 novelty_dir: str = "dataset/novelty",
                 fasta_dir: str = "dataset/processed",
                 output_dir: str = "dataset/validation",
                 identity_threshold_novel: float = 95.0,
                 identity_threshold_species: float = 97.0,
                 max_candidates: int = 50,
                 delay_between_requests: float = 3.0,
                 database: str = "nt"):
        """
        Initialize online BLAST validator
        
        Args:
            novelty_dir: Directory with novelty detection results
            fasta_dir: Directory with cleaned FASTA files
            output_dir: Directory to save validation results
            identity_threshold_novel: Below this % identity = potentially novel
            identity_threshold_species: Above this % identity = same species
            max_candidates: Maximum number of candidates to BLAST (to avoid rate limits)
            delay_between_requests: Seconds to wait between BLAST requests
            database: NCBI database (nt, nr, etc.)
        """
        self.novelty_dir = Path(novelty_dir)
        self.fasta_dir = Path(fasta_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.identity_threshold_novel = identity_threshold_novel
        self.identity_threshold_species = identity_threshold_species
        self.max_candidates = max_candidates
        self.delay_between_requests = delay_between_requests
        self.database = database
        
        logger.info(f"Online BLAST Validation Configuration:")
        logger.info(f"  Novel threshold: <{identity_threshold_novel}% identity")
        logger.info(f"  Species threshold: >{identity_threshold_species}% identity")
        logger.info(f"  Max candidates to BLAST: {max_candidates}")
        logger.info(f"  Delay between requests: {delay_between_requests}s")
        logger.info(f"  Database: {database}")
        
        logger.warning(f"\nIMPORTANT: Online BLAST is rate-limited by NCBI.")
        logger.warning(f"This process may take several minutes to hours depending on number of candidates.")
    
    def load_candidates(self) -> pd.DataFrame:
        """Load candidate novel taxa"""
        logger.info("\nLoading candidate novel taxa...")
        
        candidates_file = self.novelty_dir / "novel_candidates.csv"
        if not candidates_file.exists():
            raise FileNotFoundError(f"Candidates file not found: {candidates_file}")
        
        candidates_df = pd.read_csv(candidates_file)
        logger.info(f"  Total candidates: {len(candidates_df)}")
        logger.info(f"  Total sequences: {candidates_df['size'].sum():,}")
        
        # Prioritize by novelty score if available, otherwise by size
        if 'novelty_score' in candidates_df.columns:
            candidates_df = candidates_df.sort_values('novelty_score', ascending=False)
            logger.info(f"  Sorted by novelty score (highest first)")
        else:
            candidates_df = candidates_df.sort_values('size', ascending=False)
            logger.info(f"  Sorted by size (largest first)")
        
        # Limit to max_candidates
        if len(candidates_df) > self.max_candidates:
            logger.warning(f"  Limiting to top {self.max_candidates} candidates to avoid rate limits")
            candidates_df = candidates_df.head(self.max_candidates)
        
        return candidates_df
    
    def load_fasta_sequences(self, marker: str) -> Dict[str, str]:
        """
        Load sequences from FASTA file
        
        Args:
            marker: Marker name (ITS, LSU, SSU, or mixed)
        
        Returns:
            Dictionary mapping seqID to sequence string
        """
        fasta_files = {
            'ITS': 'ITS_clean.fasta',
            'LSU': 'LSU_clean.fasta',
            'SSU': 'SSU_clean.fasta'
        }
        
        sequences = {}
        
        if marker.lower() == 'mixed':
            markers_to_load = ['ITS', 'LSU', 'SSU']
        else:
            markers_to_load = [marker.upper()]
        
        logger.info(f"Loading sequences for marker(s): {', '.join(markers_to_load)}")
        
        for m in markers_to_load:
            fasta_file = self.fasta_dir / fasta_files.get(m, '')
            if not fasta_file.exists():
                logger.warning(f"FASTA file not found: {fasta_file}")
                continue
            
            for record in SeqIO.parse(fasta_file, 'fasta'):
                sequences[record.id] = str(record.seq)
        
        return sequences
    
    def extract_sequences_for_candidate(self,
                                       candidate_row: pd.Series,
                                       all_sequences: Dict[str, str]) -> List[str]:
        """Extract sequences for a candidate"""
        seqids = str(candidate_row['representative_seqids']).split(',')
        sequences = []
        
        for seqid in seqids:
            seqid = seqid.strip()
            if seqid in all_sequences:
                sequences.append(all_sequences[seqid])
        
        return sequences
    
    def generate_consensus_sequence(self, sequences: List[str]) -> Optional[str]:
        """Generate consensus sequence from multiple sequences"""
        if not sequences:
            return None
        
        if len(sequences) == 1:
            return sequences[0]
        
        try:
            min_len = min(len(s) for s in sequences)
            sequences_aligned = [s[:min_len] for s in sequences]
            
            consensus = []
            for i in range(min_len):
                bases = [s[i] for s in sequences_aligned]
                most_common = Counter(bases).most_common(1)[0][0]
                consensus.append(most_common)
            
            return ''.join(consensus)
        except Exception as e:
            logger.error(f"Error generating consensus: {e}")
            return None
    
    def blast_online(self, sequence: str) -> Optional[Dict]:
        """
        BLAST a sequence against NCBI online
        
        Args:
            sequence: DNA sequence string
        
        Returns:
            Dictionary with top hit information, or None if no hits
        """
        try:
            logger.info(f"    Submitting BLAST query (length: {len(sequence)} bp)...")
            
            # Submit BLAST query
            result_handle = NCBIWWW.qblast(
                program="blastn",
                database=self.database,
                sequence=sequence,
                hitlist_size=10,
                expect=1e-10,
                megablast=True  # Faster for highly similar sequences
            )
            
            # Parse results
            blast_records = list(NCBIXML.parse(result_handle))
            result_handle.close()
            
            if not blast_records or not blast_records[0].alignments:
                logger.info(f"    No BLAST hits found")
                return None
            
            # Get top alignment
            top_alignment = blast_records[0].alignments[0]
            top_hsp = top_alignment.hsps[0]
            
            # Calculate percent identity
            identity = (top_hsp.identities / top_hsp.align_length) * 100
            
            result = {
                'hit_id': top_alignment.hit_id,
                'hit_def': top_alignment.hit_def,
                'identity': identity,
                'alignment_length': top_hsp.align_length,
                'evalue': top_hsp.expect,
                'bitscore': top_hsp.bits,
                'query_start': top_hsp.query_start,
                'query_end': top_hsp.query_end
            }
            
            logger.info(f"    Top hit: {identity:.2f}% identity - {top_alignment.hit_def[:60]}")
            
            return result
            
        except Exception as e:
            logger.error(f"    BLAST error: {e}")
            return None
    
    def classify_novelty(self, blast_result: Optional[Dict]) -> Tuple[str, str]:
        """
        Classify novelty based on BLAST result
        
        Args:
            blast_result: Dictionary with BLAST hit information
        
        Returns:
            (novelty_level, description) tuple
        """
        if blast_result is None:
            return 'HIGH', 'No BLAST hits found'
        
        identity = blast_result['identity']
        
        if identity < self.identity_threshold_novel:
            return 'HIGH', f'Top hit: {identity:.1f}% identity (potentially novel genus/family)'
        elif identity < self.identity_threshold_species:
            return 'MEDIUM', f'Top hit: {identity:.1f}% identity (potentially novel species)'
        else:
            return 'LOW', f'Top hit: {identity:.1f}% identity (known species)'
    
    def validate_candidate(self,
                          candidate_row: pd.Series,
                          all_sequences: Dict[str, str],
                          candidate_num: int,
                          total_candidates: int) -> Dict:
        """
        Validate a single candidate using online BLAST
        
        Args:
            candidate_row: Candidate row from DataFrame
            all_sequences: All available sequences
            candidate_num: Current candidate number (for logging)
            total_candidates: Total number of candidates
        
        Returns:
            Dictionary with validation results
        """
        candidate_id = candidate_row['candidate_id']
        marker = str(candidate_row['marker']).upper()
        
        logger.info(f"\n[{candidate_num}/{total_candidates}] Validating Candidate {candidate_id} (marker: {marker})")
        
        # Extract sequences
        sequences = self.extract_sequences_for_candidate(candidate_row, all_sequences)
        
        if not sequences:
            logger.warning(f"  No sequences found for candidate {candidate_id}")
            return {
                'candidate_id': candidate_id,
                'validation_status': 'FAILED',
                'novelty_level': 'UNKNOWN',
                'top_identity': None,
                'top_hit_name': None,
                'description': 'No sequences found'
            }
        
        # Generate consensus
        consensus = self.generate_consensus_sequence(sequences)
        
        if not consensus:
            logger.warning(f"  Failed to generate consensus for candidate {candidate_id}")
            return {
                'candidate_id': candidate_id,
                'validation_status': 'FAILED',
                'novelty_level': 'UNKNOWN',
                'top_identity': None,
                'top_hit_name': None,
                'description': 'Failed to generate consensus'
            }
        
        logger.info(f"  Consensus length: {len(consensus)} bp")
        
        # BLAST online
        blast_result = self.blast_online(consensus)
        
        # Rate limiting delay
        if candidate_num < total_candidates:
            logger.info(f"  Waiting {self.delay_between_requests}s before next query...")
            time.sleep(self.delay_between_requests)
        
        # Classify novelty
        novelty_level, description = self.classify_novelty(blast_result)
        
        if blast_result:
            return {
                'candidate_id': candidate_id,
                'validation_status': 'SUCCESS',
                'novelty_level': novelty_level,
                'top_identity': blast_result['identity'],
                'top_hit_name': blast_result['hit_def'],
                'top_evalue': blast_result['evalue'],
                'top_bitscore': blast_result['bitscore'],
                'description': description
            }
        else:
            return {
                'candidate_id': candidate_id,
                'validation_status': 'SUCCESS',
                'novelty_level': novelty_level,
                'top_identity': None,
                'top_hit_name': None,
                'top_evalue': None,
                'top_bitscore': None,
                'description': description
            }
    
    def validate_all_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Validate all candidates"""
        logger.info(f"\n{'='*60}")
        logger.info("Validating Candidates with Online BLAST")
        logger.info(f"{'='*60}")
        
        # Load all sequences
        logger.info("\nLoading all sequences (ITS, LSU, SSU)...")
        all_sequences = {}
        all_sequences.update(self.load_fasta_sequences('ITS'))
        all_sequences.update(self.load_fasta_sequences('LSU'))
        all_sequences.update(self.load_fasta_sequences('SSU'))
        
        logger.info(f"  Loaded {len(all_sequences):,} total sequences")
        
        # Validate each candidate
        results = []
        total_candidates = len(candidates_df)
        
        logger.info(f"\nStarting validation of {total_candidates} candidates...")
        logger.info(f"Estimated time: ~{total_candidates * (self.delay_between_requests + 30) / 60:.1f} minutes")
        
        for idx, (_, row) in enumerate(candidates_df.iterrows(), 1):
            result = self.validate_candidate(row, all_sequences, idx, total_candidates)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Validation Summary")
        logger.info(f"{'='*60}")
        
        status_counts = results_df['validation_status'].value_counts()
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
        
        if 'novelty_level' in results_df.columns:
            logger.info(f"\nNovelty levels:")
            novelty_counts = results_df['novelty_level'].value_counts()
            for level, count in novelty_counts.items():
                logger.info(f"  {level}: {count}")
        
        return results_df
    
    def save_results(self, 
                    candidates_df: pd.DataFrame,
                    validation_df: pd.DataFrame):
        """Save validation results"""
        logger.info(f"\n{'='*60}")
        logger.info("Saving Validation Results")
        logger.info(f"{'='*60}")
        
        # Merge candidates with validation results
        cols_to_drop = ['validation_status', 'novelty_level', 'top_identity', 
                        'top_hit_name', 'description_y', 'top_evalue', 
                        'top_bitscore', 'description']
        
        candidates_clean = candidates_df.drop(columns=[col for col in cols_to_drop if col in candidates_df.columns], errors='ignore')
        
        if 'description_x' in candidates_clean.columns:
            candidates_clean = candidates_clean.rename(columns={'description_x': 'candidate_description'})
        
        merged_df = candidates_clean.merge(validation_df, on='candidate_id', how='left')
        
        # Save merged results
        output_file = self.output_dir / "validated_candidates.csv"
        merged_df.to_csv(output_file, index=False)
        logger.info(f"  Validated candidates: {output_file}")
        logger.info(f"    Records: {len(merged_df)}")
        
        # Save high-novelty candidates
        if 'novelty_level' in merged_df.columns:
            high_novelty = merged_df[merged_df['novelty_level'] == 'HIGH']
            high_novelty_file = self.output_dir / "high_novelty_candidates.csv"
            high_novelty.to_csv(high_novelty_file, index=False)
            logger.info(f"  High novelty candidates: {high_novelty_file}")
            logger.info(f"    Records: {len(high_novelty)}")
        
        # Save summary statistics
        summary = {
            'total_candidates': len(merged_df),
            'validated': int((merged_df['validation_status'] == 'SUCCESS').sum()),
            'failed': int((merged_df['validation_status'] == 'FAILED').sum()),
            'parameters': {
                'identity_threshold_novel': self.identity_threshold_novel,
                'identity_threshold_species': self.identity_threshold_species,
                'database': self.database,
                'max_candidates': self.max_candidates
            }
        }
        
        if 'novelty_level' in merged_df.columns:
            summary['novelty_levels'] = merged_df['novelty_level'].value_counts().to_dict()
        
        summary_file = self.output_dir / "validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Summary: {summary_file}")
    
    def run_complete_pipeline(self):
        """Execute complete validation pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info("ONLINE BLAST VALIDATION PIPELINE")
        logger.info(f"{'='*60}\n")
        
        # Load candidates
        candidates_df = self.load_candidates()
        
        # Validate all candidates
        validation_df = self.validate_all_candidates(candidates_df)
        
        # Save results
        self.save_results(candidates_df, validation_df)
        
        logger.info(f"\n{'='*60}")
        logger.info("VALIDATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Files created:")
        logger.info(f"  - validated_candidates.csv")
        logger.info(f"  - high_novelty_candidates.csv")
        logger.info(f"  - validation_summary.json")
        logger.info(f"{'='*60}\n")


def main():
    """Main execution"""
    
    # Initialize validator
    validator = OnlineBLASTValidator(
        novelty_dir="dataset/novelty",
        fasta_dir="dataset/processed",
        output_dir="dataset/validation",
        identity_threshold_novel=95.0,
        identity_threshold_species=97.0,
        max_candidates=50,  # Limit to avoid rate limits
        delay_between_requests=3.0,  # NCBI recommends 3+ seconds
        database="nt"  # Can also use "nr" for proteins
    )
    
    try:
        # Run complete pipeline
        validator.run_complete_pipeline()
        
        logger.info("Validation pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in validation pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())