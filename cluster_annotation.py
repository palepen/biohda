import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import argparse
from typing import Dict, Optional, List, Tuple
from Bio import SeqIO
from Bio.Blast import NCBIXML
import json
import subprocess
import re
import os
import yaml
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# Import shared taxonomy parser
from taxonomy_parser import parse_ncbi_lineage, get_taxonomy_levels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClusterAnnotator:
    """Annotate clusters using representative BLAST hits and LOCAL taxonomy files"""
    
    def __init__(self, config: Dict):
        """
        Initialize cluster annotator from config dictionary
        """
        # Extract settings from config
        paths_cfg = config.get('paths', {})
        annot_cfg = config.get('annotation', {})
        res_cfg = config.get('resources', {})
        
        # Get paths
        self.clusters_file = Path(paths_cfg.get('clusters_dir', 'dataset/clusters')) / "clusters.csv"
        self.fasta_dir = Path(paths_cfg.get('processed_dir', 'dataset/processed'))
        self.output_dir = Path(paths_cfg.get('annotation_dir', 'dataset/annotation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get annotation settings
        self.identity_threshold_novel = annot_cfg.get('identity_threshold_novel', 95.0)
        self.identity_threshold_species = annot_cfg.get('identity_threshold_species', 97.0)
        
        # Get BLAST settings (UPDATED: now supports multiple databases)
        self.databases = annot_cfg.get('databases', [])
        if isinstance(self.databases, str):
            self.databases = [self.databases]  # Convert single string to list
        self.use_local_blast = annot_cfg.get('use_local_blast', True)
        
        # Get batching settings
        self.batch_size = annot_cfg.get('batch_size', None)
        self.resume = annot_cfg.get('resume_from_checkpoint', False)
        
        # Get resource settings
        self.num_threads = res_cfg.get('num_workers', os.cpu_count() // 2 or 1)
        
        # NEW: Parallel processing settings
        self.max_workers = res_cfg.get('max_workers', os.cpu_count() or 4)
        self.parallel_blast = annot_cfg.get('parallel_blast', True)
        self.chunk_size = annot_cfg.get('chunk_size', 1000)  # Sequences per BLAST chunk
        
        # Standard taxonomic levels - USE SHARED DEFINITION
        self.tax_levels = get_taxonomy_levels()
        
        # UPDATED: Get taxonomy files (now a list)
        taxonomy_files = paths_cfg.get('taxonomy_files', [])
        if isinstance(taxonomy_files, str):
            taxonomy_files = [taxonomy_files]  # Convert single string to list
        
        
        self.local_tax_map = self._load_taxonomy_maps(taxonomy_files)

        # print("TAX:", len(self.local_tax_map))
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info("=" * 80)
        logger.info("Cluster Annotation Configuration (OPTIMIZED LOCAL-ONLY MODE)")
        logger.info("=" * 80)
        if self.local_tax_map:
            logger.info(f"  ✓ Loaded {len(self.local_tax_map):,} taxonomy entries from {len(taxonomy_files)} file(s)")
        else:
            logger.warning("  ✗ No local taxonomy map loaded!")
        logger.info(f"  ✓ Using {len(self.databases)} BLAST database(s)")
        logger.info(f"  Novel threshold: <{self.identity_threshold_novel}% identity")
        logger.info(f"  Species threshold: >{self.identity_threshold_species}% identity")
        logger.info(f"  BLAST threads per job: {self.num_threads}")
        logger.info(f"  Parallel workers: {self.max_workers}")
        logger.info(f"  Parallel BLAST: {'Enabled' if self.parallel_blast else 'Disabled'}")
        logger.info(f"  Chunk size: {self.chunk_size:,} sequences")
        logger.info("=" * 80)

    def _validate_configuration(self):
        """Validate that all required files and databases exist"""
        errors = []
        
        # Check BLAST databases
        if not self.databases:
            errors.append("No BLAST databases specified in config")
        else:
            for db in self.databases:
                db_path = Path(db)
                # Check if database files exist (.nhr, .nin, .nsq)
                if not (db_path.parent / f"{db_path.name}.nhr").exists():
                    errors.append(f"BLAST database not found: {db} (missing .nhr file)")
        
        # Check taxonomy map
        if not self.local_tax_map:
            errors.append("No taxonomy entries loaded")
        
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  ✗ {error}")
            raise ValueError("Invalid configuration. Please check your config.yaml")
        
        logger.info("✓ Configuration validated successfully")

    def _load_taxonomy_maps(self, taxonomy_files: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Load and merge multiple taxonomy TSV files with PARALLEL processing.
        """
        if not taxonomy_files:
            logger.warning("No taxonomy files specified in config")
            return {}
        
        logger.info(f"\n{'='*60}")
        logger.info("Loading Local Taxonomy Files (PARALLEL)")
        logger.info(f"{'='*60}")
        
        # Use parallel processing to load multiple files simultaneously
        tax_map = {}
        total_entries = 0
        skipped_entries = 0

        
        def load_single_tax_file(tax_file):
            """Helper function to load a single taxonomy file"""
            tax_path = Path(tax_file)
            
            if not tax_path.exists():
                return None, 0, 0, tax_file
            
            file_tax_map = {}
            file_entries = 0
            file_skipped = 0
            
            try:
                with open(tax_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            fields = line.strip().split('\t')
                            
                            if len(fields) < 4:
                                continue
                            
                            accession_full = fields[0]
                            accession = accession_full.split('.')[0]
                            taxid = fields[1]
                            species_name = fields[2]
                            full_lineage = fields[3]
                            
                            if taxid == '0' or taxid == 'N/A' or not full_lineage or full_lineage == 'N/A':
                                file_skipped += 1
                                continue
                            
                            tax_entry = {
                                'taxid': taxid,
                                'species_name': species_name,
                                'full_lineage': full_lineage
                            }
                            
                            tax_entry.update(parse_ncbi_lineage(full_lineage, species_name))
                            
                            if accession not in file_tax_map:
                                file_tax_map[accession] = tax_entry
                                file_entries += 1
                            
                        except Exception:
                            continue
                
                return file_tax_map, file_entries, file_skipped, tax_file
                
            except Exception as e:
                logger.error(f"    ✗ Error reading file {tax_file}: {e}")
                return None, 0, 0, tax_file
        
        # Load files in parallel
        with ThreadPoolExecutor(max_workers=min(len(taxonomy_files), self.max_workers)) as executor:
            futures = [executor.submit(load_single_tax_file, tf) for tf in taxonomy_files]
            
            for future in as_completed(futures):
                file_tax_map, file_entries, file_skipped, tax_file = future.result()
                
                if file_tax_map is not None:
                    logger.info(f"  Loading: {Path(tax_file).name}")
                    logger.info(f"    ✓ Loaded {file_entries:,} entries")
                    
                    # Merge into main map (first occurrence wins)
                    for acc, entry in file_tax_map.items():
                        if acc not in tax_map:
                            tax_map[acc] = entry
                    
                    total_entries += file_entries
                    skipped_entries += file_skipped
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Total taxonomy entries loaded: {len(tax_map):,}")
        logger.info(f"Skipped entries (no valid taxonomy): {skipped_entries:,}")
        logger.info(f"{'='*60}\n")
        
        return tax_map

    def load_clusters(self) -> pd.DataFrame:
        """Load clustering results"""
        logger.info("\nLoading clustering results...")
        clusters_df = pd.read_csv(self.clusters_file)
        
        logger.info(f"  Total sequences: {len(clusters_df):,}")
        logger.info(f"  Total clusters: {clusters_df['cluster_id'].nunique()}")
        logger.info(f"  Noise sequences: {(clusters_df['cluster_id'] == -1).sum():,}")
        
        return clusters_df
    
    def load_fasta_sequences(self) -> Dict[str, str]:
        """Load all sequences from FASTA files with PARALLEL processing"""
        logger.info("\nLoading FASTA sequences (PARALLEL)...")
        
        fasta_files = list(self.fasta_dir.glob('*.fasta'))
        
        if not fasta_files:
            logger.error(f"No FASTA files found in {self.fasta_dir}")
            return {}

        def load_single_fasta(fasta_path):
            """Helper to load a single FASTA file"""
            file_seqs = {}
            for record in SeqIO.parse(fasta_path, 'fasta'):
                file_seqs[record.id] = str(record.seq)
            return file_seqs, fasta_path.name
        
        sequences = {}
        
        # Load FASTA files in parallel
        with ThreadPoolExecutor(max_workers=min(len(fasta_files), self.max_workers)) as executor:
            futures = [executor.submit(load_single_fasta, fp) for fp in fasta_files]
            
            for future in as_completed(futures):
                file_seqs, filename = future.result()
                logger.info(f"  Loaded {len(file_seqs):,} sequences from {filename}")
                sequences.update(file_seqs)
        
        logger.info(f"  Total loaded: {len(sequences):,} sequences")
        return sequences
    
    def select_representative(self, cluster_df: pd.DataFrame) -> str:
        """Select representative sequence from cluster"""
        if 'cluster_probability' in cluster_df.columns:
            rep_idx = cluster_df['cluster_probability'].idxmax()
            return cluster_df.loc[rep_idx, 'seqID']
        else:
            return cluster_df.iloc[0]['seqID']

    def handle_large_clusters(self, cluster_df: pd.DataFrame, max_size: int = 1000) -> pd.DataFrame:
        """For very large clusters, subsample before selecting representative"""
        if len(cluster_df) > max_size:
            logger.info(f"    Large cluster ({len(cluster_df)} sequences), subsampling to {max_size}")
            if 'cluster_probability' in cluster_df.columns:
                cluster_df = cluster_df.nlargest(max_size, 'cluster_probability')
            else:
                cluster_df = cluster_df.sample(n=max_size, random_state=42)
        return cluster_df
    
    def _run_blast_chunk(self, chunk_idx: int, chunk_sequences: List[Tuple[str, str]], 
                        database: str, db_idx: int) -> Tuple[int, int, Dict[str, Dict]]:
        """
        Run BLAST on a chunk of sequences against a single database.
        Returns: (chunk_idx, db_idx, results_dict)
        """
        chunk_fasta = self.output_dir / f"chunk_{chunk_idx}_db{db_idx}.fasta"
        chunk_results = self.output_dir / f"chunk_{chunk_idx}_db{db_idx}.tsv"
        
        try:
            # Write chunk FASTA
            with open(chunk_fasta, 'w') as f:
                for seqid, seq in chunk_sequences:
                    f.write(f">{seqid}\n{seq}\n")
            
            # Run BLAST
            cmd = [
                'blastn',
                '-query', str(chunk_fasta),
                '-task', 'megablast',
                '-db', database,
                '-out', str(chunk_results),
                '-outfmt', '6 qseqid sacc stitle pident length evalue bitscore',
                '-max_target_seqs', '1',
                '-evalue', '1e-10',
                '-num_threads', str(self.num_threads)
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse results
            results = self._parse_blast_results(chunk_results)
            
            # Cleanup
            chunk_fasta.unlink()
            chunk_results.unlink()
            
            return chunk_idx, db_idx, results
            
        except Exception as e:
            logger.error(f"  ✗ BLAST chunk {chunk_idx} db {db_idx} failed: {e}")
            if chunk_fasta.exists():
                chunk_fasta.unlink()
            if chunk_results.exists():
                chunk_results.unlink()
            return chunk_idx, db_idx, {}
    
    def run_batch_blast_local_parallel(self,
                                      rep_sequences: List[Tuple[str, str]]) -> Dict[str, Dict]:
        """
        Run local BLAST with PARALLEL execution across chunks and databases.
        Major speedup for large datasets.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running PARALLEL BLAST")
        logger.info(f"  Total sequences: {len(rep_sequences):,}")
        logger.info(f"  Chunk size: {self.chunk_size:,}")
        logger.info(f"  Databases: {len(self.databases)}")
        logger.info(f"  Max parallel workers: {self.max_workers}")
        logger.info(f"{'='*60}")
        
        # Split sequences into chunks
        chunks = []
        for i in range(0, len(rep_sequences), self.chunk_size):
            chunk = rep_sequences[i:i + self.chunk_size]
            chunks.append((i // self.chunk_size, chunk))
        
        logger.info(f"  Created {len(chunks)} chunks")
        
        all_results = {}
        
        if self.parallel_blast and len(chunks) > 1:
            # Parallel execution: submit all chunk-database combinations
            tasks = []
            for chunk_idx, chunk_seqs in chunks:
                for db_idx, database in enumerate(self.databases, 1):
                    tasks.append((chunk_idx, chunk_seqs, database, db_idx))
            
            logger.info(f"  Submitting {len(tasks)} BLAST jobs...")
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._run_blast_chunk, chunk_idx, chunk_seqs, db, db_idx)
                    for chunk_idx, chunk_seqs, db, db_idx in tasks
                ]
                
                completed = 0
                for future in as_completed(futures):
                    chunk_idx, db_idx, results = future.result()
                    completed += 1
                    
                    logger.info(f"  [{completed}/{len(tasks)}] Completed chunk {chunk_idx}, db {db_idx}: {len(results)} hits")
                    
                    # Merge results (keep best hit)
                    for qseqid, hit_data in results.items():
                        if qseqid not in all_results:
                            all_results[qseqid] = hit_data
                        else:
                            if hit_data['identity'] > all_results[qseqid]['identity']:
                                all_results[qseqid] = hit_data
                            elif hit_data['identity'] == all_results[qseqid]['identity']:
                                if hit_data['evalue'] < all_results[qseqid]['evalue']:
                                    all_results[qseqid] = hit_data
        else:
            # Sequential execution (fallback)
            logger.info("  Running sequentially...")
            for chunk_idx, chunk_seqs in chunks:
                for db_idx, database in enumerate(self.databases, 1):
                    _, _, results = self._run_blast_chunk(chunk_idx, chunk_seqs, database, db_idx)
                    
                    for qseqid, hit_data in results.items():
                        if qseqid not in all_results:
                            all_results[qseqid] = hit_data
                        else:
                            if hit_data['identity'] > all_results[qseqid]['identity']:
                                all_results[qseqid] = hit_data
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Total unique hits: {len(all_results):,}")
        logger.info(f"{'='*60}\n")
        
        return all_results
    
    def _parse_blast_results(self, results_path: Path) -> Dict[str, Dict]:
        """Parse BLAST tabular output"""
        results_dict = {}
        
        try:
            with open(results_path, 'r') as f:
                for line in f:
                    fields = line.strip().split('\t')
                    if len(fields) == 7:
                        qseqid = fields[0]
                        results_dict[qseqid] = {
                            'hit_id': fields[1],
                            'hit_def': fields[2],
                            'identity': float(fields[3]),
                            'alignment_length': int(fields[4]),
                            'evalue': float(fields[5]),
                            'bitscore': float(fields[6])
                        }
            return results_dict
        except Exception as e:
            logger.error(f"  Failed to parse BLAST results: {e}")
            return {}
    
    def validate_blast_result(self, blast_result: Dict) -> bool:
        """Validate BLAST result quality"""
        if blast_result is None:
            return False
        
        min_alignment_length = 100
        max_evalue = 1e-5
        
        if blast_result['alignment_length'] < min_alignment_length:
            return False
        
        if blast_result['evalue'] > max_evalue:
            return False
        
        return True
    
    def get_structured_taxonomy(self, hit_id: str, hit_def: str) -> Dict[str, str]:
        """
        Fetch structured taxonomy from LOCAL TSV map ONLY.
        NO Entrez calls - 100% local operation.
        """
        tax_dict = {level: None for level in self.tax_levels}
        
        accession = hit_id.split('|')[-1].split('.')[0]
        
        if self.local_tax_map and accession in self.local_tax_map:
            tax_entry = self.local_tax_map[accession]
            
            for level in self.tax_levels:
                tax_dict[level] = tax_entry.get(level)
            
            return tax_dict
        else:
            return self._parse_hit_def_fallback(hit_def)

    def _parse_hit_def_fallback(self, hit_def: str) -> Dict[str, str]:
        """Fallback parser for hit_def"""
        tax_dict = {level: None for level in self.tax_levels}
        
        match = re.search(r'([A-Z][a-z]+)\s+([a-z]+)', hit_def)
        if match:
            tax_dict['genus'] = match.group(1)
            tax_dict['species'] = match.group(0)
        
        return tax_dict
    
    def classify_cluster(self, blast_result: Optional[Dict]) -> Dict:
        """Classify cluster based on BLAST result identity"""
        if blast_result is None:
            return {'classification_type': 'novel_no_hits', 'identity': None}
        
        identity = blast_result['identity']
        
        if identity < self.identity_threshold_novel:
            return {'classification_type': 'novel_divergent', 'identity': identity}
        elif identity < self.identity_threshold_species:
            return {'classification_type': 'known_genus', 'identity': identity}
        else:
            return {'classification_type': 'known_species', 'identity': identity}

    def _annotate_cluster_with_result(self,
                                       cluster_id: int,
                                       cluster_df: pd.DataFrame,
                                       rep_seqid: str,
                                       sequence: str,
                                       blast_result: Optional[Dict]) -> Dict:
        """Helper function to annotate a cluster given a pre-computed BLAST result."""
        base_annotation = {
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_df),
            'classification_type': 'unknown',
            'identity': None,
            'representative_seqid': rep_seqid,
            'blast_quality': None,
            'predicted_otu': None
        }
        
        for level in self.tax_levels:
            base_annotation[f'pred_{level}'] = None
        
        if cluster_id == -1:
            base_annotation['classification_type'] = 'noise'
            base_annotation['predicted_otu'] = 'Novel (Noise)'
            return base_annotation

        if rep_seqid is None:
            base_annotation['classification_type'] = 'error_no_rep'
            base_annotation['predicted_otu'] = 'Unknown (Error)'
            return base_annotation
        
        if sequence is None:
            base_annotation['classification_type'] = 'error_seq_missing'
            base_annotation['predicted_otu'] = 'Unknown (Sequence missing)'
            return base_annotation
        
        if len(sequence) < 100:
            base_annotation['classification_type'] = 'error_seq_short'
            base_annotation['predicted_otu'] = 'Unknown (Sequence too short)'
            base_annotation['blast_quality'] = 'poor'
            return base_annotation

        base_annotation['blast_quality'] = 'good' if self.validate_blast_result(blast_result) else 'poor'
        
        classification = self.classify_cluster(blast_result)
        base_annotation.update(classification)
        
        if classification['classification_type'] not in ['novel_no_hits', 'novel_divergent']:
            try:
                tax_dict = self.get_structured_taxonomy(blast_result['hit_id'], blast_result['hit_def'])
                
                for level in self.tax_levels:
                    base_annotation[f'pred_{level}'] = tax_dict.get(level)
                
                base_annotation['predicted_otu'] = tax_dict.get('species', blast_result['hit_def'][:100])
                
            except Exception as e:
                logger.error(f"    Error during taxonomy fetch: {e}")
                base_annotation['classification_type'] = 'error_tax_fetch'
                base_annotation['predicted_otu'] = 'Unknown (Taxonomy fetch error)'
        else:
            base_annotation['predicted_otu'] = 'Novel'

        return base_annotation

    def annotate_all_clusters(self, clusters_df: pd.DataFrame, sequences: Dict[str, str]) -> pd.DataFrame:
        """Annotate all clusters with OPTIMIZED LOCAL BLAST and taxonomy lookup"""
        logger.info(f"\n{'='*60}")
        logger.info("Annotating All Clusters (OPTIMIZED LOCAL-ONLY MODE)")
        logger.info(f"{'='*60}")
        
        unique_clusters = sorted(clusters_df['cluster_id'].unique())
        total_clusters = len(unique_clusters)
        annotations = []
        
        # Phase 1: Select representatives
        logger.info("\nPhase 1: Selecting representative sequences...")
        rep_seq_map = {} 
        reps_to_blast = []
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            
            cluster_data = clusters_df[clusters_df['cluster_id'] == cluster_id]
            cluster_data = self.handle_large_clusters(cluster_data)
            rep_seqid = self.select_representative(cluster_data)
            sequence = sequences.get(rep_seqid)
            
            rep_seq_map[cluster_id] = (rep_seqid, sequence)
            
            if sequence and len(sequence) >= 100:
                reps_to_blast.append((rep_seqid, sequence))
        
        logger.info(f"  Selected {len(rep_seq_map)} representatives")

        # Phase 2: Run PARALLEL BLAST
        logger.info("\nPhase 2: Running PARALLEL BLAST...")
        blast_results_dict = self.run_batch_blast_local_parallel(reps_to_blast)
        
        # Phase 3: Annotate with taxonomy (with progress logging)
        logger.info("\nPhase 3: Annotating clusters with taxonomy...")
        
        taxonomy_hits = 0
        taxonomy_misses = 0
        
        for idx, cluster_id in enumerate(unique_clusters, 1):
            if idx % 100 == 0 or idx == total_clusters:
                logger.info(f"  [{idx}/{total_clusters}] Processing...")
            
            cluster_data = clusters_df[clusters_df['cluster_id'] == cluster_id]
            
            if cluster_id == -1:
                annotation = self._annotate_cluster_with_result(-1, cluster_data, None, None, None)
                annotations.append(annotation)
                continue

            rep_seqid, sequence = rep_seq_map.get(cluster_id, (None, None))
            blast_result = blast_results_dict.get(rep_seqid)
            
            if blast_result:
                accession = blast_result['hit_id'].split('|')[-1].split('.')[0]
                if accession in self.local_tax_map:
                    taxonomy_hits += 1
                else:
                    taxonomy_misses += 1

            annotation = self._annotate_cluster_with_result(
                cluster_id, cluster_data, rep_seqid, sequence, blast_result
            )
            annotations.append(annotation)

        # Summary
        annotations_df = pd.DataFrame(annotations)
        
        logger.info(f"\n{'='*60}")
        logger.info("Annotation Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Taxonomy Resolution:")
        logger.info(f"  ✓ Resolved from local TSV: {taxonomy_hits}")
        logger.info(f"  ✗ Not found in local TSV: {taxonomy_misses}")
        if taxonomy_hits + taxonomy_misses > 0:
            logger.info(f"  Resolution rate: {taxonomy_hits/(taxonomy_hits+taxonomy_misses)*100:.1f}%")
        logger.info(f"\nClassification Distribution:")
        
        type_counts = annotations_df['classification_type'].value_counts()
        for cls_type, count in type_counts.items():
            logger.info(f"  {cls_type}: {count}")
        
        return annotations_df
    
    def create_master_table(self, clusters_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """Create master table with seqID, cluster_id, and predicted_otu"""
        logger.info("\nCreating master annotation table...")
        
        merge_cols = [
            'cluster_id', 'predicted_otu', 'classification_type', 'identity'
        ] + [f'pred_{level}' for level in self.tax_levels]
        
        for col in merge_cols:
            if col not in annotations_df.columns:
                annotations_df[col] = None
        
        master_df = clusters_df.merge(
            annotations_df[merge_cols],
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
        
        annotations_file = self.output_dir / "cluster_annotations.csv"
        annotations_df.to_csv(annotations_file, index=False)
        logger.info(f"  Cluster annotations: {annotations_file}")
        
        master_file = self.output_dir / "sequence_predictions.csv"
        master_df.to_csv(master_file, index=False)
        logger.info(f"  Sequence predictions: {master_file}")
        
        summary_file = self.output_dir / "annotation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'total_clusters': len(annotations_df),
                'total_sequences': len(master_df),
                'classification_distribution': annotations_df['classification_type'].value_counts().to_dict(),
                'parameters': {
                    'identity_threshold_novel': self.identity_threshold_novel,
                    'identity_threshold_species': self.identity_threshold_species,
                    'databases': self.databases,
                    'num_databases': len(self.databases),
                    'use_local_blast': self.use_local_blast,
                    'num_threads': self.num_threads,
                    'max_workers': self.max_workers,
                    'parallel_blast': self.parallel_blast,
                    'chunk_size': self.chunk_size,
                    'local_taxonomy_entries': len(self.local_tax_map)
                }
            }, f, indent=2)
        logger.info(f"  Summary: {summary_file}")
    
    def run_pipeline(self):
        """Execute complete annotation pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info("CLUSTER ANNOTATION PIPELINE (OPTIMIZED LOCAL-ONLY MODE)")
        logger.info(f"{'='*60}\n")
        
        start_time = time.time()
        
        clusters_df = self.load_clusters()
        sequences = self.load_fasta_sequences()
        
        if not sequences:
            logger.error("No sequences loaded. Check 'processed_dir' in config.yaml.")
            return 1
        
        if not self.local_tax_map:
            logger.error("No taxonomy map loaded. Check 'taxonomy_files' in config.yaml.")
            return 1
        
        annotations_df = self.annotate_all_clusters(clusters_df, sequences)
        master_df = self.create_master_table(clusters_df, annotations_df)
        self.save_results(annotations_df, master_df)
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info("ANNOTATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Time per cluster: {elapsed_time/len(annotations_df):.3f} seconds")
        logger.info(f"{'='*60}")
        return 0


def load_config(config_path: str) -> Dict:
    """Loads YAML config file."""
    logger.info(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return None
    except Exception as e:
        logger.error(f"Error parsing config file: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Annotate clusters via representative BLAST (OPTIMIZED LOCAL-ONLY MODE)',
        epilog='''
Example usage:
  python cluster_annotation.py --config config.yaml
  
OPTIMIZATIONS:
  - Parallel BLAST execution across multiple chunks
  - Parallel taxonomy file loading
  - Parallel FASTA file loading
  - Batch processing with configurable chunk sizes
  - Multi-database support with result aggregation
  
Config options for optimization:
  resources:
    num_workers: 4           # Threads per BLAST job
    max_workers: 8           # Number of parallel BLAST jobs
  
  annotation:
    parallel_blast: true     # Enable parallel BLAST
    chunk_size: 1000         # Sequences per BLAST chunk
    databases:               # List of BLAST databases
      - /path/to/db1
      - /path/to/db2
        '''
    )
    parser.add_argument('--config', default='config.yaml', help='Path to the config.yaml file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Configure logging level from config
    log_level = config.get('logging', {}).get('level', 'INFO').upper()
    logging.getLogger().setLevel(log_level)
    
    # Create and run the annotator
    annotator = ClusterAnnotator(config=config)
    
    try:
        return annotator.run_pipeline()
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())