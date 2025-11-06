"""
Step 9: Biological Validation using BLAST
==========================================
Validate candidate novel taxa by BLASTing against reference databases

This regenerated version includes new logic to:
- Define paths for ALL THREE marker databases (ITS, LSU, SSU).
- Dynamically select the correct database based on the candidate's
  'marker' column ('ITS', 'LSU', 'SSU').
- Use the ITS database as the default for 'mixed' clusters.
"""

import subprocess
import tempfile
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
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


class BLASTValidator:
    """BLAST-based validation of candidate novel taxa"""
    
    def __init__(self,
                 novelty_dir: str = "dataset/novelty",
                 fasta_dir: str = "dataset/processed",
                 blast_db_paths: Dict[str, str] = None, # <-- Changed
                 output_dir: str = "dataset/validation",
                 identity_threshold_novel: float = 95.0,
                 identity_threshold_species: float = 97.0,
                 evalue_threshold: float = 1e-10,
                 max_target_seqs: int = 10,
                 num_threads: int = 4):
        """
        Initialize BLAST validator
        
        Args:
            novelty_dir: Directory with novelty detection results
            fasta_dir: Directory with cleaned FASTA files
            blast_db_paths: Dictionary mapping 'ITS', 'LSU', 'SSU' to their DB paths
            output_dir: Directory to save validation results
            identity_threshold_novel: Below this % identity = potentially novel
            identity_threshold_species: Above this % identity = same species
            evalue_threshold: E-value threshold for BLAST
            max_target_seqs: Number of BLAST hits to retrieve
            num_threads: Number of CPU threads for BLAST
        """
        self.novelty_dir = Path(novelty_dir)
        self.fasta_dir = Path(fasta_dir)
        
        # --- NEW DATABASE LOGIC ---
        self.blast_db_paths = blast_db_paths
        if self.blast_db_paths is None:
            logger.warning("No BLAST database paths provided. BLAST validation will be skipped.")
            self.dbs_available = False
        else:
            self.dbs_available = True
            logger.info("BLAST databases configured:")
            for marker, path in self.blast_db_paths.items():
                logger.info(f"  {marker}: {path}")
        # --- END NEW LOGIC ---

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.identity_threshold_novel = identity_threshold_novel
        self.identity_threshold_species = identity_threshold_species
        self.evalue_threshold = evalue_threshold
        self.max_target_seqs = max_target_seqs
        self.num_threads = num_threads
        
        logger.info(f"BLAST Validation Configuration:")
        logger.info(f"  Novel threshold: <{identity_threshold_novel}% identity")
        logger.info(f"  Species threshold: >{identity_threshold_species}% identity")
        logger.info(f"  E-value threshold: {evalue_threshold}")
        logger.info(f"  BLAST threads: {num_threads}")
    
    def check_blast_installation(self) -> bool:
        """Check if blastn is available"""
        try:
            result = subprocess.run(
                ['blastn', '-version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"BLAST+ found: {result.stdout.split()[1]}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("blastn not found. Please install BLAST+ tools.")
            return False
    
    def load_candidates(self) -> pd.DataFrame:
        """Load candidate novel taxa"""
        logger.info("\nLoading candidate novel taxa...")
        
        candidates_file = self.novelty_dir / "novel_candidates.csv"
        if not candidates_file.exists():
            raise FileNotFoundError(f"Candidates file not found: {candidates_file}")
        
        candidates_df = pd.read_csv(candidates_file)
        logger.info(f"  Total candidates: {len(candidates_df)}")
        logger.info(f"  Total sequences: {candidates_df['size'].sum():,}")
        
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
            # Load all markers
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
        """
        Extract sequences for a candidate
        """
        seqids = str(candidate_row['representative_seqids']).split(',')
        sequences = []
        
        for seqid in seqids:
            seqid = seqid.strip()
            if seqid in all_sequences:
                sequences.append(all_sequences[seqid])
        
        return sequences
    
    def generate_consensus_sequence(self, sequences: List[str]) -> Optional[str]:
        """
        Generate consensus sequence from multiple sequences
        """
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
    
    def run_blastn(self,
                  query_file: Path,
                  output_file: Path,
                  db_path: str) -> bool: # db_path is now required
        """
        Run blastn search
        """
        logger.info(f"  BLASTing against: {db_path}")
        
        # BLAST command
        cmd = [
            'blastn',
            '-query', str(query_file),
            '-db', db_path,
            '-out', str(output_file),
            '-outfmt', '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore stitle',
            '-evalue', str(self.evalue_threshold),
            '-max_target_seqs', str(self.max_target_seqs),
            '-num_threads', str(self.num_threads)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"BLAST failed: {e.stderr}")
            return False
    
    def parse_blast_results(self, blast_output_file: Path) -> List[Dict]:
        """
        Parse BLAST output
        """
        if not blast_output_file.exists() or blast_output_file.stat().st_size == 0:
            return []
        
        hits = []
        
        with open(blast_output_file, 'r') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) >= 13:
                    hits.append({
                        'query_id': fields[0],
                        'subject_id': fields[1],
                        'identity': float(fields[2]),
                        'alignment_length': int(fields[3]),
                        'mismatches': int(fields[4]),
                        'gap_opens': int(fields[5]),
                        'query_start': int(fields[6]),
                        'query_end': int(fields[7]),
                        'subject_start': int(fields[8]),
                        'subject_end': int(fields[9]),
                        'evalue': float(fields[10]),
                        'bitscore': float(fields[11]),
                        'subject_title': fields[12] if len(fields) > 12 else ''
                    })
        
        return hits
    
    def classify_novelty(self, top_hit: Optional[Dict]) -> Tuple[str, str]:
        """
        Classify novelty based on top BLAST hit
        """
        if top_hit is None:
            return 'HIGH', 'No BLAST hits found'
        
        identity = top_hit['identity']
        
        if identity < self.identity_threshold_novel:
            return 'HIGH', f'Top hit: {identity:.1f}% identity (potentially novel genus/family)'
        elif identity < self.identity_threshold_species:
            return 'MEDIUM', f'Top hit: {identity:.1f}% identity (potentially novel species)'
        else:
            return 'LOW', f'Top hit: {identity:.1f}% identity (known species)'
    
    def validate_candidate(self,
                          candidate_row: pd.Series,
                          all_sequences: Dict[str, str],
                          temp_dir: Path) -> Dict:
        """
        Validate a single candidate using BLAST
        """
        candidate_id = candidate_row['candidate_id']
        marker = str(candidate_row['marker']).upper()
        
        # --- NEW LOGIC: Select correct DB ---
        db_path = None
        if marker == 'ITS':
            db_path = self.blast_db_paths.get('ITS')
        elif marker == 'LSU':
            db_path = self.blast_db_paths.get('LSU')
        elif marker == 'SSU':
            db_path = self.blast_db_paths.get('SSU')
        elif marker == 'MIXED':
            # Default to ITS for mixed clusters (usually largest)
            db_path = self.blast_db_paths.get('ITS')
            logger.info(f"  Candidate {candidate_id} is 'mixed', defaulting to ITS db.")
        
        if db_path is None:
            logger.warning(f"  No database found for marker '{marker}'. Skipping candidate {candidate_id}.")
            return {
                'candidate_id': candidate_id,
                'validation_status': 'SKIPPED',
                'novelty_level': 'UNKNOWN',
                'top_identity': None,
                'top_hit_name': None,
                'description': f'BLAST skipped (no DB for marker {marker})'
            }
        # --- END NEW LOGIC ---

        # Extract sequences
        sequences = self.extract_sequences_for_candidate(candidate_row, all_sequences)
        
        if not sequences:
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
            return {
                'candidate_id': candidate_id,
                'validation_status': 'FAILED',
                'novelty_level': 'UNKNOWN',
                'top_identity': None,
                'top_hit_name': None,
                'description': 'Failed to generate consensus'
            }
        
        # Create query file
        query_file = temp_dir / f"candidate_{candidate_id}.fasta"
        with open(query_file, 'w') as f:
            record = SeqRecord(
                Seq(consensus),
                id=f"candidate_{candidate_id}",
                description=f"Consensus sequence for candidate {candidate_id} (marker: {marker})"
            )
            SeqIO.write(record, f, 'fasta')
        
        # Run BLAST
        blast_output = temp_dir / f"candidate_{candidate_id}_blast.txt"
        blast_success = self.run_blastn(query_file, blast_output, db_path)
        
        if not blast_success:
            return {
                'candidate_id': candidate_id,
                'validation_status': 'FAILED',
                'novelty_level': 'UNKNOWN',
                'top_identity': None,
                'top_hit_name': None,
                'description': 'BLAST run failed'
            }
        
        # Parse BLAST results
        hits = self.parse_blast_results(blast_output)
        
        if hits:
            top_hit = hits[0]
            novelty_level, description = self.classify_novelty(top_hit)
            
            return {
                'candidate_id': candidate_id,
                'validation_status': 'SUCCESS',
                'novelty_level': novelty_level,
                'top_identity': top_hit['identity'],
                'top_hit_name': top_hit['subject_title'],
                'top_evalue': top_hit['evalue'],
                'top_bitscore': top_hit['bitscore'],
                'num_hits': len(hits),
                'description': description
            }
        else:
            novelty_level, description = self.classify_novelty(None)
            
            return {
                'candidate_id': candidate_id,
                'validation_status': 'SUCCESS',
                'novelty_level': novelty_level,
                'top_identity': None,
                'top_hit_name': None,
                'top_evalue': None,
                'top_bitscore': None,
                'num_hits': 0,
                'description': description
            }
    
    def validate_all_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate all candidates
        """
        logger.info(f"\n{'='*60}")
        logger.info("Validating Candidates with BLAST")
        logger.info(f"{'='*60}")
        
        if not self.dbs_available:
            logger.warning("\nNo BLAST databases configured! Skipping all BLAST validation.\n")
            results = []
            for _, row in candidates_df.iterrows():
                results.append({
                    'candidate_id': row['candidate_id'],
                    'validation_status': 'SKIPPED',
                    'novelty_level': 'UNKNOWN',
                    'top_identity': None,
                    'top_hit_name': None,
                    'description': 'BLAST validation skipped (no database)'
                })
            return pd.DataFrame(results)
        
        # Load all sequences (ITS, LSU, SSU)
        logger.info("\nLoading all sequences (ITS, LSU, SSU)...")
        all_sequences = {}
        all_sequences.update(self.load_fasta_sequences('ITS'))
        all_sequences.update(self.load_fasta_sequences('LSU'))
        all_sequences.update(self.load_fasta_sequences('SSU'))
        
        logger.info(f"  Loaded {len(all_sequences):,} total sequences")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results = []
            
            # Validate each candidate
            for _, row in tqdm(candidates_df.iterrows(), 
                             total=len(candidates_df),
                             desc="Validating candidates"):
                
                result = self.validate_candidate(row, all_sequences, temp_path)
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
        """
        Save validation results
        """
        logger.info(f"\n{'='*60}")
        logger.info("Saving Validation Results")
        logger.info(f"{'='*60}")
        
        # Merge candidates with validation results
        # Drop old validation columns if they exist
        cols_to_drop = ['validation_status', 'novelty_level', 'top_identity', 
                        'top_hit_name', 'description_y', 'top_evalue', 
                        'top_bitscore', 'num_hits', 'description']
        
        candidates_clean = candidates_df.drop(columns=[col for col in cols_to_drop if col in candidates_df.columns], errors='ignore')
        
        # Rename description_x to avoid clash
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
            'skipped': int((merged_df['validation_status'] == 'SKIPPED').sum()),
            'failed': int((merged_df['validation_status'] == 'FAILED').sum()),
            'parameters': {
                'identity_threshold_novel': self.identity_threshold_novel,
                'identity_threshold_species': self.identity_threshold_species,
                'evalue_threshold': self.evalue_threshold
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
        logger.info("BLAST VALIDATION PIPELINE")
        logger.info(f"{'='*60}\n")
        
        # Check BLAST installation
        if not self.check_blast_installation():
            logger.error("BLAST not available. Cannot proceed.")
            return
        
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
    
    # --- THIS IS THE PART TO EDIT ---
    # Define the paths to your 3 databases
    db_paths = {
        "ITS": "dataset/raw/ITS_eukaryote_sequences/ITS_eukaryote_sequences",
        "LSU": "dataset/raw/LSU_eukaryote_rRNA/LSU_eukaryote_rRNA",
        "SSU": "dataset/raw/SSU_eukaryote_rRNA/SSU_eukaryote_rRNA"
    }
    # --------------------------------
    
    # Initialize validator
    validator = BLASTValidator(
        novelty_dir="dataset/novelty",
        fasta_dir="dataset/processed",
        blast_db_paths=db_paths,  # <-- Pass the dictionary here
        output_dir="dataset/validation",
        identity_threshold_novel=95.0,
        identity_threshold_species=97.0,
        evalue_threshold=1e-10,
        max_target_seqs=10,
        num_threads=4 # Adjust this to the number of CPU cores you want to use
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