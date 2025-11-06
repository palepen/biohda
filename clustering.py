"""
Step 7: HDBSCAN Clustering
===========================
Perform unsupervised clustering on CNN embeddings using HDBSCAN

HDBSCAN advantages:
- No need to specify number of clusters
- Identifies noise points (potential novel taxa)
- Hierarchical density-based clustering
- Robust to varying cluster densities

Dependencies:
- HDBSCAN
- NumPy
- pandas
- scikit-learn
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import json
from collections import Counter
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HDBSCANClustering:
    """HDBSCAN clustering on embeddings"""
    
    def __init__(self,
                 embeddings_dir: str = "dataset/embeddings",
                 metadata_file: str = "dataset/metadata.csv",
                 output_dir: str = "dataset/clusters",
                 min_cluster_size: int = 50,
                 min_samples: int = 10,
                 cluster_selection_method: str = 'eom',
                 metric: str = 'euclidean',
                 normalize: bool = True):
        """
        Initialize HDBSCAN clustering
        
        Args:
            embeddings_dir: Directory with embeddings
            metadata_file: Path to metadata CSV
            output_dir: Directory to save clustering results
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in neighborhood
            cluster_selection_method: 'eom' or 'leaf'
            metric: Distance metric
            normalize: Whether to normalize embeddings before clustering
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self.normalize = normalize
        
        logger.info(f"HDBSCAN Configuration:")
        logger.info(f"  min_cluster_size: {min_cluster_size}")
        logger.info(f"  min_samples: {min_samples}")
        logger.info(f"  cluster_selection_method: {cluster_selection_method}")
        logger.info(f"  metric: {metric}")
        logger.info(f"  normalize: {normalize}")
    
    def load_embeddings(self) -> Tuple[np.ndarray, Dict]:
        """
        Load embeddings and metadata
        
        Returns:
            (embeddings, metadata) tuple
        """
        logger.info("\nLoading embeddings...")
        
        embeddings_file = self.embeddings_dir / "embeddings.npy"
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")
        
        embeddings = np.load(embeddings_file)
        logger.info(f"  Embeddings shape: {embeddings.shape}")
        logger.info(f"  Embeddings dtype: {embeddings.dtype}")
        
        # Load metadata
        metadata_file = self.embeddings_dir / "embedding_metadata.pkl"
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"  Number of sequences: {len(metadata['seq_ids']):,}")
        
        return embeddings, metadata
    
    def load_taxonomy_metadata(self) -> pd.DataFrame:
        """Load taxonomy metadata CSV"""
        logger.info("\nLoading taxonomy metadata...")
        
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_file}")
        
        metadata_df = pd.read_csv(self.metadata_file)
        logger.info(f"  Metadata records: {len(metadata_df):,}")
        logger.info(f"  Columns: {list(metadata_df.columns)}")
        
        return metadata_df
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """
        Normalize embeddings using StandardScaler
        
        Args:
            embeddings: Raw embeddings
        
        Returns:
            (normalized_embeddings, scaler)
        """
        logger.info("\nNormalizing embeddings...")
        
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        
        logger.info(f"  Original mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
        logger.info(f"  Normalized mean: {embeddings_normalized.mean():.4f}, std: {embeddings_normalized.std():.4f}")
        
        return embeddings_normalized, scaler
    
    def perform_clustering(self, embeddings: np.ndarray) -> hdbscan.HDBSCAN:
        """
        Perform HDBSCAN clustering
        
        Args:
            embeddings: Embedding vectors
        
        Returns:
            Fitted HDBSCAN clusterer
        """
        logger.info(f"\n{'='*60}")
        logger.info("Performing HDBSCAN Clustering")
        logger.info(f"{'='*60}")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            metric=self.metric,
            core_dist_n_jobs=-1  # Use all CPU cores
        )
        
        logger.info("Running HDBSCAN (this may take a few minutes)...")
        clusterer.fit(embeddings)
        
        # Extract results
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        
        # Count clusters and noise
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"\nClustering Results:")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Number of noise points: {n_noise:,} ({n_noise/len(labels):.2%})")
        logger.info(f"  Number of clustered points: {len(labels) - n_noise:,}")
        
        # Cluster size distribution
        cluster_sizes = Counter(labels[labels != -1])
        if cluster_sizes:
            sizes = list(cluster_sizes.values())
            logger.info(f"\nCluster size statistics:")
            logger.info(f"  Min size: {min(sizes):,}")
            logger.info(f"  Max size: {max(sizes):,}")
            logger.info(f"  Mean size: {np.mean(sizes):.1f}")
            logger.info(f"  Median size: {np.median(sizes):.1f}")
        
        return clusterer
    
    def compute_clustering_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Compute clustering quality metrics
        
        Args:
            embeddings: Embedding vectors
            labels: Cluster labels
        
        Returns:
            Dictionary of metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info("Computing Clustering Metrics")
        logger.info(f"{'='*60}")
        
        metrics = {}
        
        # Filter out noise points for some metrics
        mask = labels != -1
        embeddings_clustered = embeddings[mask]
        labels_clustered = labels[mask]
        
        if len(np.unique(labels_clustered)) > 1:
            # Silhouette Score
            logger.info("Computing Silhouette Score...")
            silhouette = silhouette_score(embeddings_clustered, labels_clustered, 
                                         sample_size=min(10000, len(labels_clustered)))
            metrics['silhouette_score'] = float(silhouette)
            logger.info(f"  Silhouette Score: {silhouette:.4f}")
            
            # Calinski-Harabasz Index
            logger.info("Computing Calinski-Harabasz Score...")
            ch_score = calinski_harabasz_score(embeddings_clustered, labels_clustered)
            metrics['calinski_harabasz_score'] = float(ch_score)
            logger.info(f"  Calinski-Harabasz Score: {ch_score:.2f}")
            
            # Davies-Bouldin Index
            logger.info("Computing Davies-Bouldin Score...")
            db_score = davies_bouldin_score(embeddings_clustered, labels_clustered)
            metrics['davies_bouldin_score'] = float(db_score)
            logger.info(f"  Davies-Bouldin Score: {db_score:.4f}")
        else:
            logger.warning("Not enough clusters for metric computation")
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz_score'] = None
            metrics['davies_bouldin_score'] = None
        
        return metrics
    
    def analyze_cluster_taxonomy(self, 
                                 cluster_labels: np.ndarray,
                                 seq_ids: list,
                                 metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze taxonomic composition of each cluster
        
        Args:
            cluster_labels: Cluster labels for each sequence
            seq_ids: Sequence IDs
            metadata_df: Taxonomy metadata
        
        Returns:
            DataFrame with cluster taxonomy analysis
        """
        logger.info("\nAnalyzing cluster taxonomy...")
        
        # Create mapping
        seqid_to_taxid = dict(zip(metadata_df['seqID'], metadata_df['taxID']))
        seqid_to_name = dict(zip(metadata_df['seqID'], metadata_df['scientific_name']))
        
        cluster_analysis = []
        
        unique_clusters = sorted(set(cluster_labels))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        for cluster_id in tqdm(unique_clusters, desc="Analyzing clusters"):
            # Get sequences in this cluster
            mask = cluster_labels == cluster_id
            cluster_seqids = [seq_ids[i] for i in range(len(seq_ids)) if mask[i]]
            
            # Get taxonomies
            cluster_taxids = [seqid_to_taxid.get(sid, 'NA') for sid in cluster_seqids]
            cluster_names = [seqid_to_name.get(sid, 'NA') for sid in cluster_seqids]
            
            # Filter out NA
            valid_taxids = [t for t in cluster_taxids if t != 'NA']
            valid_names = [n for n in cluster_names if n != 'NA']
            
            # Count taxonomies
            taxid_counts = Counter(valid_taxids)
            name_counts = Counter(valid_names)
            
            # Compute purity (fraction of most common taxon)
            if valid_taxids:
                most_common_taxid, most_common_count = taxid_counts.most_common(1)[0]
                purity = most_common_count / len(valid_taxids)
                majority_taxid = most_common_taxid
                majority_name = name_counts.most_common(1)[0][0]
            else:
                purity = 0.0
                majority_taxid = 'NA'
                majority_name = 'NA'
            
            cluster_analysis.append({
                'cluster_id': cluster_id,
                'size': len(cluster_seqids),
                'num_taxa': len(taxid_counts),
                'purity': purity,
                'majority_taxid': majority_taxid,
                'majority_name': majority_name,
                'majority_count': most_common_count if valid_taxids else 0,
                'has_taxonomy': len(valid_taxids) > 0
            })
        
        # Analyze noise
        noise_mask = cluster_labels == -1
        n_noise = noise_mask.sum()
        if n_noise > 0:
            noise_seqids = [seq_ids[i] for i in range(len(seq_ids)) if noise_mask[i]]
            noise_taxids = [seqid_to_taxid.get(sid, 'NA') for sid in noise_seqids]
            valid_noise_taxids = [t for t in noise_taxids if t != 'NA']
            
            cluster_analysis.append({
                'cluster_id': -1,
                'size': n_noise,
                'num_taxa': len(set(valid_noise_taxids)),
                'purity': 0.0,
                'majority_taxid': 'NOISE',
                'majority_name': 'NOISE',
                'majority_count': 0,
                'has_taxonomy': len(valid_noise_taxids) > 0
            })
        
        df = pd.DataFrame(cluster_analysis)
        
        logger.info(f"  Analyzed {len(df)-1} clusters + noise")
        
        return df
    
    def save_results(self,
                    clusterer: hdbscan.HDBSCAN,
                    embeddings: np.ndarray,
                    metadata: Dict,
                    cluster_analysis: pd.DataFrame,
                    metrics: Dict):
        """
        Save clustering results
        
        Args:
            clusterer: Fitted HDBSCAN object
            embeddings: Embeddings used for clustering
            metadata: Embedding metadata
            cluster_analysis: Cluster taxonomy analysis
            metrics: Clustering metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info("Saving Clustering Results")
        logger.info(f"{'='*60}")
        
        # Save cluster labels and probabilities
        results_df = pd.DataFrame({
            'seqID': metadata['seq_ids'],
            'marker': metadata['markers'],
            'cluster_id': clusterer.labels_,
            'cluster_probability': clusterer.probabilities_
        })
        
        results_file = self.output_dir / "clusters.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"  Cluster assignments: {results_file}")
        logger.info(f"    Records: {len(results_df):,}")
        
        # Save cluster analysis
        analysis_file = self.output_dir / "cluster_analysis.csv"
        cluster_analysis.to_csv(analysis_file, index=False)
        logger.info(f"  Cluster analysis: {analysis_file}")
        
        # Save HDBSCAN model
        model_file = self.output_dir / "hdbscan_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(clusterer, f)
        logger.info(f"  HDBSCAN model: {model_file}")
        
        # Save configuration and metrics
        config = {
            'parameters': {
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_method': self.cluster_selection_method,
                'metric': self.metric,
                'normalize': self.normalize
            },
            'results': {
                'n_clusters': int(len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)),
                'n_noise': int(list(clusterer.labels_).count(-1)),
                'n_sequences': len(clusterer.labels_)
            },
            'metrics': metrics
        }
        
        config_file = self.output_dir / "clustering_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"  Configuration: {config_file}")
        
        # Print summary statistics
        logger.info(f"\n{'='*60}")
        logger.info("Cluster Purity Summary")
        logger.info(f"{'='*60}")
        
        # Exclude noise from purity analysis
        valid_clusters = cluster_analysis[cluster_analysis['cluster_id'] != -1]
        if len(valid_clusters) > 0:
            logger.info(f"High purity clusters (>0.8): {(valid_clusters['purity'] > 0.8).sum()}")
            logger.info(f"Medium purity clusters (0.5-0.8): {((valid_clusters['purity'] > 0.5) & (valid_clusters['purity'] <= 0.8)).sum()}")
            logger.info(f"Low purity clusters (<0.5): {(valid_clusters['purity'] <= 0.5).sum()}")
            logger.info(f"Mean purity: {valid_clusters['purity'].mean():.3f}")
            logger.info(f"Median purity: {valid_clusters['purity'].median():.3f}")
    
    def run_complete_pipeline(self):
        """Execute complete clustering pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info("HDBSCAN CLUSTERING PIPELINE")
        logger.info(f"{'='*60}\n")
        
        # Load data
        embeddings, embedding_metadata = self.load_embeddings()
        taxonomy_metadata = self.load_taxonomy_metadata()
        
        # Normalize if requested
        if self.normalize:
            embeddings, scaler = self.normalize_embeddings(embeddings)
            
            # Save scaler
            scaler_file = self.output_dir / "scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"  Saved scaler: {scaler_file}")
        
        # Perform clustering
        clusterer = self.perform_clustering(embeddings)
        
        # Compute metrics
        metrics = self.compute_clustering_metrics(embeddings, clusterer.labels_)
        
        # Analyze cluster taxonomy
        cluster_analysis = self.analyze_cluster_taxonomy(
            clusterer.labels_,
            embedding_metadata['seq_ids'],
            taxonomy_metadata
        )
        
        # Save results
        self.save_results(
            clusterer,
            embeddings,
            embedding_metadata,
            cluster_analysis,
            metrics
        )
        
        logger.info(f"\n{'='*60}")
        logger.info("CLUSTERING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Files created:")
        logger.info(f"  - clusters.csv")
        logger.info(f"  - cluster_analysis.csv")
        logger.info(f"  - hdbscan_model.pkl")
        logger.info(f"  - clustering_config.json")
        if self.normalize:
            logger.info(f"  - scaler.pkl")
        logger.info(f"{'='*60}\n")


def main():
    """Main execution"""
    
    # Initialize clustering
    clustering = HDBSCANClustering(
        embeddings_dir="dataset/embeddings",
        metadata_file="dataset/metadata.csv",
        output_dir="dataset/clusters",
        min_cluster_size=50,        # Adjust based on dataset size
        min_samples=10,              # Controls how conservative clustering is
        cluster_selection_method='eom',  # 'eom' or 'leaf'
        metric='euclidean',
        normalize=True               # Normalize embeddings before clustering
    )
    
    try:
        # Run complete pipeline
        clustering.run_complete_pipeline()
        
        logger.info("Clustering pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in clustering pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
