"""
cluster_annotation.py
=====================
Cluster Annotation via Representative BLAST

Assigns taxonomic labels to clusters by BLASTing one representative sequence per cluster.
This creates "Predicted OTU" labels for evaluation against Ground Truth.

Usage:
    python cluster_annotation.py
    python cluster_annotation.py --batch-size 25 --resume
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import argparse
from typing import Dict, Optional
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
import json
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClusterAnnotator:
    """Annotate clusters using representative BLAST hits"""
    
    def __init__(self,
                 clusters_file: str = "dataset/clusters/clusters.csv",
                 fasta_dir: str = "dataset/processed",
                 output_dir: str = "dataset/annotation",
                 identity_threshold_novel: float = 95.0,
                 identity_threshold_species: float = 97.0,
                 delay_between_requests: float = 3.0,
                 database: str = "./dataset/raw/SSU_eukaryote_rRNA/SSU_eukaryote_rRNA",
                 use_local_blast: bool = True,
                 batch_size: int = None,
                 resume: bool = False):
        """
        Initialize cluster annotator
        
        Args:
            clusters_file: Path to clusters.csv from HDBSCAN
            fasta_dir: Directory with cleaned FASTA files
            output_dir: Directory to save annotation results
            identity_threshold_novel: Below this % = novel
            identity_threshold_species: Above this % = same species
            delay_between_requests: Seconds between BLAST requests
            database: NCBI database (nt, nr, etc.)
            use_local_blast: Use local BLAST instead of online
            batch_size: Process clusters in batches (for checkpointing)
            resume: Resume from checkpoint
        """
        self.clusters_file = Path(clusters_file)
        self.fasta_dir = Path(fasta_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.identity_threshold_novel = identity_threshold_novel
        self.identity_threshold_species = identity_threshold_species
        self.delay_between_requests = delay_between_requests
        self.database = database
        self.use_local_blast = use_local_blast
        self.batch_size = batch_size
        self.resume = resume
        
        logger.info("Cluster Annotation Configuration:")
        logger.info(f"  Novel threshold: <{identity_threshold_novel}% identity")
        logger.info(f"  Species threshold: >{identity_threshold_species}% identity")
        logger.info(f"  BLAST mode: {'Local' if use_local_blast else 'Online'}")
        logger.info(f"  Delay between requests: {delay_between_requests}s")
        logger.info(f"  Database: {database}")
        if batch_size:
            logger.info(f"  Batch size: {batch_size}")
            logger.info(f"  Resume mode: {resume}")
    
    def load_clusters(self) -> pd.DataFrame:
        """Load clustering results"""
        logger.info("\nLoading clustering results...")
        clusters_df = pd.read_csv(self.clusters_file)
        
        logger.info(f"  Total sequences: {len(clusters_df):,}")
        logger.info(f"  Total clusters: {clusters_df['cluster_id'].nunique()}")
        logger.info(f"  Noise sequences: {(clusters_df['cluster_id'] == -1).sum():,}")
        
        return clusters_df
    
    def load_fasta_sequences(self) -> Dict[str, str]:
        """Load all sequences from FASTA files"""
        logger.info("\nLoading FASTA sequences...")
        
        fasta_files = ['SSU_clean.fasta']
        sequences = {}
        
        for fasta_file in fasta_files:
            fasta_path = self.fasta_dir / fasta_file
            if not fasta_path.exists():
                logger.warning(f"FASTA file not found: {fasta_path}")
                continue
            
            for record in SeqIO.parse(fasta_path, 'fasta'):
                sequences[record.id] = str(record.seq)
        
        logger.info(f"  Loaded {len(sequences):,} sequences")
        return sequences
    
    def select_representative(self, cluster_df: pd.DataFrame) -> str:
        """
        Select representative sequence from cluster
        Prefers sequence with highest cluster_probability
        """
        if 'cluster_probability' in cluster_df.columns:
            rep_idx = cluster_df['cluster_probability'].idxmax()
            return cluster_df.loc[rep_idx, 'seqID']
        else:
            return cluster_df.iloc[0]['seqID']
    
    def handle_large_clusters(self, cluster_df: pd.DataFrame, max_size: int = 1000) -> pd.DataFrame:
        """
        For very large clusters, subsample before selecting representative
        """
        if len(cluster_df) > max_size:
            logger.info(f"    Large cluster ({len(cluster_df)} sequences), subsampling to {max_size}")
            if 'cluster_probability' in cluster_df.columns:
                cluster_df = cluster_df.nlargest(max_size, 'cluster_probability')
            else:
                cluster_df = cluster_df.sample(n=max_size, random_state=42)
        return cluster_df
    
    def blast_online(self, sequence: str) -> Optional[Dict]:
        """BLAST a sequence against NCBI online"""
        try:
            logger.info(f"    Submitting BLAST query (length: {len(sequence)} bp)...")
            
            result_handle = NCBIWWW.qblast(
                program="blastn",
                database=self.database,
                sequence=sequence,
                hitlist_size=10,
                expect=1e-10,
                megablast=True
            )
            
            blast_records = list(NCBIXML.parse(result_handle))
            result_handle.close()
            
            if not blast_records or not blast_records[0].alignments:
                logger.info(f"    No BLAST hits found")
                return None
            
            top_alignment = blast_records[0].alignments[0]
            top_hsp = top_alignment.hsps[0]
            
            identity = (top_hsp.identities / top_hsp.align_length) * 100
            
            result = {
                'hit_id': top_alignment.hit_id,
                'hit_def': top_alignment.hit_def,
                'identity': identity,
                'alignment_length': top_hsp.align_length,
                'evalue': top_hsp.expect,
                'bitscore': top_hsp.bits
            }
            
            logger.info(f"    Top hit: {identity:.2f}% identity - {top_alignment.hit_def[:60]}")
            return result
            
        except Exception as e:
            logger.error(f"    BLAST error: {e}")
            return None
    
    def blast_local(self, sequence: str) -> Optional[Dict]:
        """
        BLAST against local database using blastn command
        Requires: Local BLAST+ installation and downloaded database
        """
        try:
            temp_fasta = self.output_dir / "temp_query.fasta"
            with open(temp_fasta, 'w') as f:
                f.write(f">query\n{sequence}\n")
            
            cmd = [
                'blastn',
                '-query', str(temp_fasta),
                '-db', self.database,
                '-outfmt', '6 sacc stitle pident length evalue bitscore',
                '-max_target_seqs', '1',
                '-evalue', '1e-10'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if not result.stdout.strip():
                return None
            
            fields = result.stdout.strip().split('\t')
            
            return {
                'hit_id': fields[0],
                'hit_def': fields[1],
                'identity': float(fields[2]),
                'alignment_length': int(fields[3]),
                'evalue': float(fields[4]),
                'bitscore': float(fields[5])
            }
            
        except Exception as e:
            logger.error(f"    Local BLAST error: {e}")
            return None
        finally:
            if temp_fasta.exists():
                temp_fasta.unlink()
    
    def validate_blast_result(self, blast_result: Dict) -> bool:
        """Validate BLAST result quality"""
        if blast_result is None:
            return False
        
        min_alignment_length = 100
        max_evalue = 1e-5
        
        if blast_result['alignment_length'] < min_alignment_length:
            logger.warning(f"    Low alignment length: {blast_result['alignment_length']} bp")
            return False
        
        if blast_result['evalue'] > max_evalue:
            logger.warning(f"    Poor e-value: {blast_result['evalue']}")
            return False
        
        return True
    
    def extract_lineage_from_hit(self, hit_def: str) -> str:
        """Extract taxonomic lineage from BLAST hit definition"""
        parts = hit_def.split('|')
        if len(parts) > 1:
            species_part = parts[-1].strip()
            words = species_part.split()[:2]
            return ', '.join(words)
        return hit_def[:50]
    
    def classify_cluster(self, blast_result: Optional[Dict]) -> Dict[str, str]:
        """Classify cluster based on BLAST result"""
        if blast_result is None:
            return {
                'predicted_otu': 'Novel (No BLAST hits)',
                'classification_type': 'novel_no_hits',
                'identity': None
            }
        
        identity = blast_result['identity']
        
        if identity < self.identity_threshold_novel:
            return {
                'predicted_otu': 'Novel (BLAST-Divergent)',
                'classification_type': 'novel_divergent',
                'identity': identity
            }
        elif identity < self.identity_threshold_species:
            lineage = self.extract_lineage_from_hit(blast_result['hit_def'])
            return {
                'predicted_otu': lineage,
                'classification_type': 'known_genus',
                'identity': identity
            }
        else:
            lineage = self.extract_lineage_from_hit(blast_result['hit_def'])
            return {
                'predicted_otu': lineage,
                'classification_type': 'known_species',
                'identity': identity
            }
    
    def annotate_cluster(self,
                        cluster_id: int,
                        cluster_df: pd.DataFrame,
                        sequences: Dict[str, str],
                        cluster_num: int,
                        total_clusters: int) -> Dict:
        """Annotate a single cluster"""
        logger.info(f"\n[{cluster_num}/{total_clusters}] Annotating Cluster {cluster_id}")
        logger.info(f"  Size: {len(cluster_df)} sequences")
        
        # Handle noise cluster
        if cluster_id == -1:
            return {
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_df),
                'predicted_otu': 'Novel (Noise)',
                'classification_type': 'noise',
                'identity': None,
                'representative_seqid': None,
                'blast_quality': None
            }
        
        # Handle large clusters
        cluster_df = self.handle_large_clusters(cluster_df)
        
        # Select representative
        rep_seqid = self.select_representative(cluster_df)
        logger.info(f"  Representative: {rep_seqid}")
        
        # Get sequence
        if rep_seqid not in sequences:
            logger.warning(f"  Representative sequence not found in FASTA")
            return {
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_df),
                'predicted_otu': 'Unknown (Sequence missing)',
                'classification_type': 'error',
                'identity': None,
                'representative_seqid': rep_seqid,
                'blast_quality': None
            }
        
        sequence = sequences[rep_seqid]
        
        # Validate sequence length
        if len(sequence) < 100:
            logger.warning(f"  Sequence too short for reliable BLAST: {len(sequence)} bp")
            return {
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_df),
                'predicted_otu': 'Unknown (Sequence too short)',
                'classification_type': 'error',
                'identity': None,
                'representative_seqid': rep_seqid,
                'blast_quality': 'poor'
            }
        
        # BLAST representative
        if self.use_local_blast:
            blast_result = self.blast_local(sequence)
        else:
            blast_result = self.blast_online(sequence)
        
        # Validate BLAST result
        blast_quality = 'good' if self.validate_blast_result(blast_result) else 'poor'
        
        # Rate limiting delay for online BLAST
        if not self.use_local_blast and cluster_num < total_clusters:
            logger.info(f"  Waiting {self.delay_between_requests}s before next query...")
            time.sleep(self.delay_between_requests)
        
        # Classify cluster
        classification = self.classify_cluster(blast_result)
        
        return {
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_df),
            'predicted_otu': classification['predicted_otu'],
            'classification_type': classification['classification_type'],
            'identity': classification.get('identity'),
            'representative_seqid': rep_seqid,
            'blast_quality': blast_quality
        }
    
    def load_checkpoint(self) -> set:
        """Load checkpoint of completed clusters"""
        checkpoint_file = self.output_dir / "annotation_checkpoint.json"
        
        if self.resume and checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                completed = set(checkpoint.get('completed_clusters', []))
            logger.info(f"Resuming from checkpoint: {len(completed)} clusters already completed")
            return completed
        return set()
    
    def save_checkpoint(self, completed_clusters: set, total_clusters: int):
        """Save checkpoint of completed clusters"""
        checkpoint_file = self.output_dir / "annotation_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'completed_clusters': list(completed_clusters),
                'total_clusters': total_clusters,
                'timestamp': pd.Timestamp.now().isoformat()
            }, f)
    
    def annotate_all_clusters(self, clusters_df: pd.DataFrame, sequences: Dict[str, str]) -> pd.DataFrame:
        """Annotate all clusters with optional batching"""
        logger.info(f"\n{'='*60}")
        logger.info("Annotating All Clusters")
        logger.info(f"{'='*60}")
        
        unique_clusters = sorted(clusters_df['cluster_id'].unique())
        total_clusters = len(unique_clusters)
        
        logger.info(f"\nTotal clusters to annotate: {total_clusters}")
        if not self.use_local_blast:
            logger.info(f"Estimated time: ~{total_clusters * (self.delay_between_requests + 30) / 60:.1f} minutes")
        
        # Load checkpoint if resuming
        completed_clusters = self.load_checkpoint()
        
        annotations = []
        
        # Process clusters
        for idx, cluster_id in enumerate(unique_clusters, 1):
            if cluster_id in completed_clusters:
                logger.info(f"\nSkipping already completed cluster {cluster_id}")
                continue
            
            cluster_data = clusters_df[clusters_df['cluster_id'] == cluster_id]
            annotation = self.annotate_cluster(cluster_id, cluster_data, sequences, idx, total_clusters)
            annotations.append(annotation)
            completed_clusters.add(cluster_id)
            
            # Save checkpoint after batch
            if self.batch_size and len(annotations) % self.batch_size == 0:
                self.save_checkpoint(completed_clusters, total_clusters)
                logger.info(f"Checkpoint saved: {len(completed_clusters)}/{total_clusters} completed")
        
        # Final checkpoint
        if self.batch_size:
            self.save_checkpoint(completed_clusters, total_clusters)
        
        annotations_df = pd.DataFrame(annotations)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Annotation Summary")
        logger.info(f"{'='*60}")
        
        type_counts = annotations_df['classification_type'].value_counts()
        for cls_type, count in type_counts.items():
            logger.info(f"  {cls_type}: {count}")
        
        return annotations_df
    
    def create_master_table(self, clusters_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """Create master table with seqID, cluster_id, and predicted_otu"""
        logger.info("\nCreating master annotation table...")
        
        master_df = clusters_df.merge(
            annotations_df[['cluster_id', 'predicted_otu', 'classification_type', 'identity']],
            on='cluster_id',
            how='left'
        )
        
        logger.info(f"  Master table: {len(master_df):,} sequences")
        return master_df
    
    def save_results(self, annotations_df: pd.DataFrame, master_df: pd.DataFrame):
        """Save annotation results"""
        logger.info(f"\n{'='*60}")
        logger.info("Saving Results")
        logger.info(f"{'='*60}")
        
        # Save cluster annotations
        annotations_file = self.output_dir / "cluster_annotations.csv"
        annotations_df.to_csv(annotations_file, index=False)
        logger.info(f"  Cluster annotations: {annotations_file}")
        
        # Save master table
        master_file = self.output_dir / "sequence_predictions.csv"
        master_df.to_csv(master_file, index=False)
        logger.info(f"  Sequence predictions: {master_file}")
        
        # Save summary
        summary = {
            'total_clusters': len(annotations_df),
            'total_sequences': len(master_df),
            'classification_distribution': annotations_df['classification_type'].value_counts().to_dict(),
            'parameters': {
                'identity_threshold_novel': self.identity_threshold_novel,
                'identity_threshold_species': self.identity_threshold_species,
                'database': self.database,
                'use_local_blast': self.use_local_blast
            }
        }
        
        summary_file = self.output_dir / "annotation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Summary: {summary_file}")
    
    def run_pipeline(self):
        """Execute complete annotation pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info("CLUSTER ANNOTATION PIPELINE")
        logger.info(f"{'='*60}\n")
        
        clusters_df = self.load_clusters()
        sequences = self.load_fasta_sequences()
        annotations_df = self.annotate_all_clusters(clusters_df, sequences)
        master_df = self.create_master_table(clusters_df, annotations_df)
        self.save_results(annotations_df, master_df)
        
        logger.info(f"\n{'='*60}")
        logger.info("ANNOTATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Output directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Annotate clusters via representative BLAST')
    parser.add_argument('--clusters', default='dataset/clusters/clusters.csv', help='Path to clusters CSV')
    parser.add_argument('--fasta-dir', default='dataset/processed', help='Directory with FASTA files')
    parser.add_argument('--output', default='dataset/annotation', help='Output directory')
    parser.add_argument('--local-blast', action='store_true', help='Use local BLAST instead of online')
    parser.add_argument('--batch-size', type=int, help='Process in batches with checkpointing')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--delay', type=float, default=3.0, help='Delay between online BLAST requests')
    
    args = parser.parse_args()
    
    annotator = ClusterAnnotator(
        clusters_file=args.clusters,
        fasta_dir=args.fasta_dir,
        output_dir=args.output,
        use_local_blast=args.local_blast,
        batch_size=args.batch_size,
        resume=args.resume,
        delay_between_requests=args.delay
    )
    
    try:
        annotator.run_pipeline()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())