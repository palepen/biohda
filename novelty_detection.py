"""
Step 8: Novelty Detection
==========================
Identify candidate novel taxa from clustering results

Novelty criteria:
1. Sequences labeled as noise (cluster_id = -1)
2. Clusters with low taxonomic purity (<0.3)
3. Clusters with no known taxonomy
4. Distance-based outliers in embedding space

Dependencies:
- NumPy
- pandas
- scikit-learn
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from collections import Counter, defaultdict
from tqdm import tqdm
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NoveltyDetector:
    """Detect candidate novel taxa from clustering results"""
    
    def __init__(self,
                 clusters_dir: str = "dataset/clusters",
                 embeddings_dir: str = "dataset/embeddings",
                 metadata_file: str = "dataset/metadata.csv",
                 output_dir: str = "dataset/novelty",
                 purity_threshold: float = 0.3,
                 min_novel_cluster_size: int = 5,
                 isolation_threshold: float = 0.1):
        """
        Initialize novelty detector
        
        Args:
            clusters_dir: Directory with clustering results
            embeddings_dir: Directory with embeddings
            metadata_file: Path to taxonomy metadata
            output_dir: Directory to save novelty results
            purity_threshold: Clusters below this purity are candidate novel
            min_novel_cluster_size: Minimum size for novel cluster candidates
            isolation_threshold: Distance threshold for isolation score
        """
        self.clusters_dir = Path(clusters_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.purity_threshold = purity_threshold
        self.min_novel_cluster_size = min_novel_cluster_size
        self.isolation_threshold = isolation_threshold
        
        logger.info(f"Novelty Detection Configuration:")
        logger.info(f"  Purity threshold: {purity_threshold}")
        logger.info(f"  Min novel cluster size: {min_novel_cluster_size}")
        logger.info(f"  Isolation threshold: {isolation_threshold}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict]:
        """
        Load all required data
        
        Returns:
            (clusters_df, cluster_analysis_df, embeddings, embedding_metadata)
        """
        logger.info("\nLoading data...")
        
        # Load cluster assignments
        clusters_file = self.clusters_dir / "clusters.csv"
        clusters_df = pd.read_csv(clusters_file)
        logger.info(f"  Cluster assignments: {len(clusters_df):,} sequences")
        
        # Load cluster analysis
        analysis_file = self.clusters_dir / "cluster_analysis.csv"
        cluster_analysis_df = pd.read_csv(analysis_file)
        logger.info(f"  Cluster analysis: {len(cluster_analysis_df)} clusters")
        
        # Load embeddings
        embeddings_file = self.embeddings_dir / "embeddings.npy"
        embeddings = np.load(embeddings_file)
        logger.info(f"  Embeddings shape: {embeddings.shape}")
        
        # Load embedding metadata
        metadata_file = self.embeddings_dir / "embedding_metadata.pkl"
        with open(metadata_file, 'rb') as f:
            embedding_metadata = pickle.load(f)
        
        return clusters_df, cluster_analysis_df, embeddings, embedding_metadata
    
    def identify_noise_sequences(self, clusters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify sequences labeled as noise by HDBSCAN
        
        Args:
            clusters_df: Cluster assignments DataFrame
        
        Returns:
            DataFrame of noise sequences
        """
        logger.info("\nIdentifying noise sequences...")
        
        noise_df = clusters_df[clusters_df['cluster_id'] == -1].copy()
        logger.info(f"  Found {len(noise_df):,} noise sequences ({len(noise_df)/len(clusters_df):.2%})")
        
        return noise_df
    
    def identify_low_purity_clusters(self, 
                                    cluster_analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify clusters with low taxonomic purity
        
        Args:
            cluster_analysis_df: Cluster analysis DataFrame
        
        Returns:
            DataFrame of low-purity clusters
        """
        logger.info("\nIdentifying low-purity clusters...")
        
        # Exclude noise cluster
        valid_clusters = cluster_analysis_df[cluster_analysis_df['cluster_id'] != -1].copy()
        
        # Filter by purity threshold
        low_purity = valid_clusters[valid_clusters['purity'] < self.purity_threshold].copy()
        
        # Filter by minimum size
        low_purity = low_purity[low_purity['size'] >= self.min_novel_cluster_size]
        
        logger.info(f"  Found {len(low_purity)} low-purity clusters (purity < {self.purity_threshold})")
        
        if len(low_purity) > 0:
            logger.info(f"    Size range: {low_purity['size'].min():,} - {low_purity['size'].max():,}")
            logger.info(f"    Purity range: {low_purity['purity'].min():.3f} - {low_purity['purity'].max():.3f}")
        
        return low_purity
    
    def identify_unknown_taxonomy_clusters(self,
                                          cluster_analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify clusters with no known taxonomy
        
        Args:
            cluster_analysis_df: Cluster analysis DataFrame
        
        Returns:
            DataFrame of unknown taxonomy clusters
        """
        logger.info("\nIdentifying clusters with unknown taxonomy...")
        
        # Exclude noise cluster
        valid_clusters = cluster_analysis_df[cluster_analysis_df['cluster_id'] != -1].copy()
        
        # Find clusters without taxonomy
        unknown = valid_clusters[~valid_clusters['has_taxonomy']].copy()
        
        # Filter by minimum size
        unknown = unknown[unknown['size'] >= self.min_novel_cluster_size]
        
        logger.info(f"  Found {len(unknown)} clusters with no taxonomy")
        
        if len(unknown) > 0:
            logger.info(f"    Total sequences: {unknown['size'].sum():,}")
        
        return unknown
    
    def compute_isolation_scores(self,
                                embeddings: np.ndarray,
                                clusters_df: pd.DataFrame,
                                k: int = 50) -> np.ndarray:
        """
        Compute isolation score for each sequence
        
        Isolation score measures how far a sequence is from its k nearest neighbors
        
        Args:
            embeddings: Embedding vectors
            clusters_df: Cluster assignments
            k: Number of nearest neighbors
        
        Returns:
            Array of isolation scores
        """
        logger.info(f"\nComputing isolation scores (k={k})...")
        
        # Fit nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1)
        nbrs.fit(embeddings)
        
        # Find distances to k nearest neighbors
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Isolation score = mean distance to k nearest neighbors (excluding self)
        isolation_scores = distances[:, 1:].mean(axis=1)
        
        logger.info(f"  Isolation score range: {isolation_scores.min():.4f} - {isolation_scores.max():.4f}")
        logger.info(f"  Mean isolation: {isolation_scores.mean():.4f}")
        logger.info(f"  Median isolation: {np.median(isolation_scores):.4f}")
        
        return isolation_scores
    
    def identify_isolated_sequences(self,
                                   isolation_scores: np.ndarray,
                                   clusters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify highly isolated sequences
        
        Args:
            isolation_scores: Isolation scores for each sequence
            clusters_df: Cluster assignments
        
        Returns:
            DataFrame of isolated sequences
        """
        logger.info("\nIdentifying isolated sequences...")
        
        # Compute threshold (e.g., top 10% most isolated)
        threshold = np.percentile(isolation_scores, 90)
        
        # Find isolated sequences
        isolated_mask = isolation_scores > threshold
        isolated_df = clusters_df[isolated_mask].copy()
        isolated_df['isolation_score'] = isolation_scores[isolated_mask]
        
        logger.info(f"  Threshold: {threshold:.4f}")
        logger.info(f"  Found {len(isolated_df):,} isolated sequences ({len(isolated_df)/len(clusters_df):.2%})")
        
        return isolated_df
    
    def extract_representative_sequences(self,
                                       cluster_id: int,
                                       clusters_df: pd.DataFrame,
                                       embeddings: np.ndarray,
                                       n_representatives: int = 5) -> List[str]:
        """
        Extract representative sequences from a cluster
        
        Uses sequences closest to cluster centroid
        
        Args:
            cluster_id: Cluster ID
            clusters_df: Cluster assignments
            embeddings: Embedding vectors
            n_representatives: Number of representatives to extract
        
        Returns:
            List of representative seqIDs
        """
        # Get sequences in this cluster
        cluster_mask = clusters_df['cluster_id'] == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            return []
        
        # Get cluster embeddings
        cluster_embeddings = embeddings[cluster_indices]
        
        # Compute centroid
        centroid = cluster_embeddings.mean(axis=0)
        
        # Find distances to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Get closest sequences
        n_rep = min(n_representatives, len(cluster_indices))
        closest_indices = np.argsort(distances)[:n_rep]
        
        # Get seqIDs
        representative_seqids = clusters_df.iloc[cluster_indices[closest_indices]]['seqID'].tolist()
        
        return representative_seqids
    
    def compile_novel_candidates(self,
                                noise_df: pd.DataFrame,
                                low_purity_df: pd.DataFrame,
                                unknown_taxonomy_df: pd.DataFrame,
                                isolated_df: pd.DataFrame,
                                clusters_df: pd.DataFrame,
                                cluster_analysis_df: pd.DataFrame,
                                embeddings: np.ndarray) -> pd.DataFrame:
        """
        Compile all candidate novel taxa
        
        Args:
            noise_df: Noise sequences
            low_purity_df: Low purity clusters
            unknown_taxonomy_df: Unknown taxonomy clusters
            isolated_df: Isolated sequences
            clusters_df: All cluster assignments
            cluster_analysis_df: Cluster analysis
            embeddings: Embeddings
        
        Returns:
            DataFrame of novel candidates
        """
        logger.info(f"\n{'='*60}")
        logger.info("Compiling Novel Candidates")
        logger.info(f"{'='*60}")
        
        candidates = []
        candidate_id = 1
        
        # 1. Noise sequences
        if len(noise_df) > 0:
            logger.info("\nProcessing noise sequences...")
            
            # Group noise by marker
            for marker in noise_df['marker'].unique():
                marker_noise = noise_df[noise_df['marker'] == marker]
                
                if len(marker_noise) >= self.min_novel_cluster_size:
                    # Extract representatives
                    representatives = marker_noise['seqID'].head(5).tolist()
                    
                    candidates.append({
                        'candidate_id': candidate_id,
                        'type': 'noise',
                        'cluster_id': -1,
                        'marker': marker,
                        'size': len(marker_noise),
                        'purity': 0.0,
                        'majority_taxid': 'NOISE',
                        'majority_name': 'NOISE',
                        'representative_seqids': ','.join(representatives),
                        'description': f'HDBSCAN noise sequences from {marker}'
                    })
                    candidate_id += 1
            
            logger.info(f"  Created {candidate_id - 1} noise candidate groups")
        
        # 2. Low purity clusters
        if len(low_purity_df) > 0:
            logger.info("\nProcessing low-purity clusters...")
            
            for _, row in low_purity_df.iterrows():
                cluster_id = row['cluster_id']
                representatives = self.extract_representative_sequences(
                    cluster_id, clusters_df, embeddings, n_representatives=5
                )
                
                candidates.append({
                    'candidate_id': candidate_id,
                    'type': 'low_purity',
                    'cluster_id': cluster_id,
                    'marker': 'mixed',
                    'size': row['size'],
                    'purity': row['purity'],
                    'majority_taxid': row['majority_taxid'],
                    'majority_name': row['majority_name'],
                    'representative_seqids': ','.join(representatives),
                    'description': f'Low purity cluster (purity={row["purity"]:.3f})'
                })
                candidate_id += 1
            
            logger.info(f"  Added {len(low_purity_df)} low-purity clusters")
        
        # 3. Unknown taxonomy clusters
        if len(unknown_taxonomy_df) > 0:
            logger.info("\nProcessing unknown taxonomy clusters...")
            
            for _, row in unknown_taxonomy_df.iterrows():
                cluster_id = row['cluster_id']
                representatives = self.extract_representative_sequences(
                    cluster_id, clusters_df, embeddings, n_representatives=5
                )
                
                candidates.append({
                    'candidate_id': candidate_id,
                    'type': 'unknown_taxonomy',
                    'cluster_id': cluster_id,
                    'marker': 'mixed',
                    'size': row['size'],
                    'purity': 0.0,
                    'majority_taxid': 'UNKNOWN',
                    'majority_name': 'UNKNOWN',
                    'representative_seqids': ','.join(representatives),
                    'description': 'Cluster with no known taxonomy'
                })
                candidate_id += 1
            
            logger.info(f"  Added {len(unknown_taxonomy_df)} unknown taxonomy clusters")
        
        # 4. Isolated sequences (group by marker)
        if len(isolated_df) > 0:
            logger.info("\nProcessing isolated sequences...")
            
            for marker in isolated_df['marker'].unique():
                marker_isolated = isolated_df[isolated_df['marker'] == marker]
                
                if len(marker_isolated) >= self.min_novel_cluster_size:
                    # Sort by isolation score and take top representatives
                    top_isolated = marker_isolated.nlargest(5, 'isolation_score')
                    representatives = top_isolated['seqID'].tolist()
                    
                    candidates.append({
                        'candidate_id': candidate_id,
                        'type': 'isolated',
                        'cluster_id': -1,
                        'marker': marker,
                        'size': len(marker_isolated),
                        'purity': 0.0,
                        'majority_taxid': 'ISOLATED',
                        'majority_name': 'ISOLATED',
                        'representative_seqids': ','.join(representatives),
                        'description': f'Highly isolated sequences from {marker}'
                    })
                    candidate_id += 1
            
            logger.info(f"  Created {candidate_id - len(candidates)} isolated candidate groups")
        
        candidates_df = pd.DataFrame(candidates)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Total novel candidates: {len(candidates_df)}")
        logger.info(f"{'='*60}")
        
        if len(candidates_df) > 0:
            logger.info("\nBreakdown by type:")
            for ctype in candidates_df['type'].unique():
                count = (candidates_df['type'] == ctype).sum()
                total_seqs = candidates_df[candidates_df['type'] == ctype]['size'].sum()
                logger.info(f"  {ctype}: {count} candidates ({total_seqs:,} sequences)")
        
        return candidates_df
    
    def save_results(self, 
                    candidates_df: pd.DataFrame,
                    noise_df: pd.DataFrame,
                    isolated_df: pd.DataFrame):
        """
        Save novelty detection results
        
        Args:
            candidates_df: Novel candidates DataFrame
            noise_df: Noise sequences
            isolated_df: Isolated sequences
        """
        logger.info(f"\n{'='*60}")
        logger.info("Saving Results")
        logger.info(f"{'='*60}")
        
        # Save candidate novel taxa
        candidates_file = self.output_dir / "novel_candidates.csv"
        candidates_df.to_csv(candidates_file, index=False)
        logger.info(f"  Novel candidates: {candidates_file}")
        logger.info(f"    Records: {len(candidates_df)}")
        
        # Save detailed noise sequences
        noise_file = self.output_dir / "noise_sequences.csv"
        noise_df.to_csv(noise_file, index=False)
        logger.info(f"  Noise sequences: {noise_file}")
        logger.info(f"    Records: {len(noise_df):,}")
        
        # Save isolated sequences
        isolated_file = self.output_dir / "isolated_sequences.csv"
        isolated_df.to_csv(isolated_file, index=False)
        logger.info(f"  Isolated sequences: {isolated_file}")
        logger.info(f"    Records: {len(isolated_df):,}")
        
        # Save summary statistics
        summary = {
            'total_candidates': len(candidates_df),
            'total_sequences_in_candidates': int(candidates_df['size'].sum()),
            'candidates_by_type': candidates_df['type'].value_counts().to_dict(),
            'parameters': {
                'purity_threshold': self.purity_threshold,
                'min_novel_cluster_size': self.min_novel_cluster_size,
                'isolation_threshold': self.isolation_threshold
            }
        }
        
        summary_file = self.output_dir / "novelty_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Summary: {summary_file}")
    
    def run_complete_pipeline(self):
        """Execute complete novelty detection pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info("NOVELTY DETECTION PIPELINE")
        logger.info(f"{'='*60}\n")
        
        # Load data
        clusters_df, cluster_analysis_df, embeddings, embedding_metadata = self.load_data()
        
        # 1. Identify noise sequences
        noise_df = self.identify_noise_sequences(clusters_df)
        
        # 2. Identify low-purity clusters
        low_purity_df = self.identify_low_purity_clusters(cluster_analysis_df)
        
        # 3. Identify unknown taxonomy clusters
        unknown_taxonomy_df = self.identify_unknown_taxonomy_clusters(cluster_analysis_df)
        
        # 4. Compute isolation scores
        isolation_scores = self.compute_isolation_scores(embeddings, clusters_df, k=50)
        
        # 5. Identify isolated sequences
        isolated_df = self.identify_isolated_sequences(isolation_scores, clusters_df)
        
        # 6. Compile all novel candidates
        candidates_df = self.compile_novel_candidates(
            noise_df,
            low_purity_df,
            unknown_taxonomy_df,
            isolated_df,
            clusters_df,
            cluster_analysis_df,
            embeddings
        )
        
        # 7. Save results
        self.save_results(candidates_df, noise_df, isolated_df)
        
        logger.info(f"\n{'='*60}")
        logger.info("NOVELTY DETECTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Files created:")
        logger.info(f"  - novel_candidates.csv")
        logger.info(f"  - noise_sequences.csv")
        logger.info(f"  - isolated_sequences.csv")
        logger.info(f"  - novelty_summary.json")
        logger.info(f"{'='*60}\n")


def main():
    """Main execution"""
    
    # Initialize detector
    detector = NoveltyDetector(
        clusters_dir="dataset/clusters",
        embeddings_dir="dataset/embeddings",
        metadata_file="dataset/metadata.csv",
        output_dir="dataset/novelty",
        purity_threshold=0.3,           # Clusters below this are candidate novel
        min_novel_cluster_size=5,       # Minimum size for novel candidates
        isolation_threshold=0.1         # For distance-based detection
    )
    
    try:
        # Run complete pipeline
        detector.run_complete_pipeline()
        
        logger.info("Novelty detection completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in novelty detection: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
