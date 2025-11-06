"""
Step 10: Evaluation Metrics
============================
Compute comprehensive evaluation metrics for clustering and novelty detection

This file is regenerated with two fixes:
1.  In `load_data`, forces `taxID` to be read as a string to prevent 'NA'
    from becoming `numpy.nan`.
2.  In `prepare_labels`, the filter is strengthened to catch 'NA', 'nan',
    and `numpy.nan` values to prevent the `Input contains NaN` error.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,  # Fixed import name
    silhouette_score,
    v_measure_score,
    fowlkes_mallows_score,
    confusion_matrix
)
from collections import Counter
import json
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Compute evaluation metrics for clustering and novelty detection"""
    
    def __init__(self,
                 clusters_dir: str = "dataset/clusters",
                 embeddings_dir: str = "dataset/embeddings",
                 novelty_dir: str = "dataset/novelty",
                 metadata_file: str = "dataset/metadata.csv",
                 output_dir: str = "dataset/evaluation"):
        
        self.clusters_dir = Path(clusters_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.novelty_dir = Path(novelty_dir)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Load all required data
        
        Returns:
            (clusters_df, metadata_df, cluster_analysis_df, embeddings)
        """
        logger.info("\nLoading data...")
        
        # Load cluster assignments
        clusters_file = self.clusters_dir / "clusters.csv"
        clusters_df = pd.read_csv(clusters_file)
        logger.info(f"  Cluster assignments: {len(clusters_df):,} sequences")
        
        # --- FIX 1 ---
        # Load metadata, forcing taxID to be a string.
        # This prevents pandas from auto-converting "NA" to numpy.nan.
        try:
            metadata_df = pd.read_csv(self.metadata_file, dtype={'taxID': str})
        except Exception as e:
            logger.error(f"Error reading {self.metadata_file}: {e}")
            raise
        logger.info(f"  Metadata: {len(metadata_df):,} records")
        
        # Load cluster analysis
        analysis_file = self.clusters_dir / "cluster_analysis.csv"
        cluster_analysis_df = pd.read_csv(analysis_file)
        logger.info(f"  Cluster analysis: {len(cluster_analysis_df)} clusters")
        
        # Load embeddings
        embeddings_file = self.embeddings_dir / "embeddings.npy"
        embeddings = np.load(embeddings_file)
        logger.info(f"  Embeddings: {embeddings.shape}")
        
        return clusters_df, metadata_df, cluster_analysis_df, embeddings
    
    def prepare_labels(self, 
                      clusters_df: pd.DataFrame,
                      metadata_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare predicted and true labels
        
        Args:
            clusters_df: Cluster assignments
            metadata_df: Taxonomy metadata
        
        Returns:
            (predicted_labels, true_labels, valid_mask)
        """
        logger.info("\nPreparing labels...")
        
        # Create seqID to taxID mapping
        # Because dtype=str, all values are strings (e.g., "12345", "NA")
        # or nan if the original CSV field was empty.
        seqid_to_taxid = dict(zip(metadata_df['seqID'], metadata_df['taxID']))
        
        # Get predicted labels (cluster IDs)
        predicted_labels = clusters_df['cluster_id'].values
        
        # Get true labels (taxIDs)
        true_labels = []
        valid_mask = []
        
        for seqid in clusters_df['seqID']:
            taxid = seqid_to_taxid.get(seqid, 'NA') # taxid is a string or nan
            
            # --- FIX 2 ---
            # Robustly check for invalid labels.
            # 1. pd.notna(taxid) filters out numpy.nan
            # 2. str(taxid).upper() != 'NA' filters out "NA"
            # 3. str(taxid).upper() != 'NAN' filters out "nan"
            taxid_str = str(taxid).upper()
            is_valid = (
                pd.notna(taxid) and 
                taxid_str != 'NA' and 
                taxid_str != 'NAN'
            )
            
            if is_valid:
                true_labels.append(str(taxid)) # Append original string
                valid_mask.append(True)
            else:
                true_labels.append('NA')
                valid_mask.append(False)
        
        true_labels = np.array(true_labels)
        valid_mask = np.array(valid_mask)
        
        # These logs will now be correct
        logger.info(f"  Total sequences: {len(predicted_labels):,}")
        logger.info(f"  Sequences with taxonomy: {valid_mask.sum():,} ({valid_mask.sum()/len(valid_mask):.1%})")
        
        if valid_mask.sum() > 0:
            logger.info(f"  Unique taxa: {len(set(true_labels[valid_mask])):,}")
        else:
            logger.info("  Unique taxa: 0 (No valid taxonomy found)")
            
        logger.info(f"  Unique clusters: {len(set(predicted_labels)):,}")
        
        return predicted_labels, true_labels, valid_mask
    
    def compute_clustering_metrics(self,
                                  predicted_labels: np.ndarray,
                                  true_labels: np.ndarray,
                                  valid_mask: np.ndarray,
                                  embeddings: np.ndarray) -> Dict:
        """
        Compute clustering quality metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info("Computing Clustering Metrics")
        logger.info(f"{'='*60}")
        
        metrics = {}
        
        # Filter to valid sequences (those with taxonomy)
        pred_valid = predicted_labels[valid_mask]
        true_valid = true_labels[valid_mask]
        emb_valid = embeddings[valid_mask]
        
        # Check if we have any valid data left
        if len(true_valid) == 0 or len(pred_valid) == 0:
            logger.warning("No valid taxonomy data found. Skipping all clustering metrics.")
            metrics['adjusted_rand_index'] = None
            metrics['normalized_mutual_information'] = None
            metrics['v_measure'] = None
            metrics['fowlkes_mallows_score'] = None
            metrics['cluster_purity'] = None
            metrics['silhouette_score'] = None
            return metrics

        # Adjusted Rand Index
        logger.info("\nComputing Adjusted Rand Index...")
        ari = adjusted_rand_score(true_valid, pred_valid)
        metrics['adjusted_rand_index'] = float(ari)
        logger.info(f"  ARI: {ari:.4f}")
        
        # Normalized Mutual Information
        logger.info("Computing Normalized Mutual Information...")
        nmi = normalized_mutual_info_score(true_valid, pred_valid)
        metrics['normalized_mutual_information'] = float(nmi)
        logger.info(f"  NMI: {nmi:.4f}")
        
        # V-Measure
        logger.info("Computing V-Measure...")
        v_measure = v_measure_score(true_valid, pred_valid)
        metrics['v_measure'] = float(v_measure)
        logger.info(f"  V-Measure: {v_measure:.4f}")
        
        # Fowlkes-Mallows Score
        logger.info("Computing Fowlkes-Mallows Score...")
        fm_score = fowlkes_mallows_score(true_valid, pred_valid)
        metrics['fowlkes_mallows_score'] = float(fm_score)
        logger.info(f"  Fowlkes-Mallows: {fm_score:.4f}")
        
        # Cluster Purity
        logger.info("Computing Cluster Purity...")
        purity = self.compute_purity(pred_valid, true_valid)
        metrics['cluster_purity'] = float(purity)
        logger.info(f"  Purity: {purity:.4f}")
        
        # Silhouette Score (on subset if too large)
        logger.info("Computing Silhouette Score...")
        
        # Exclude noise points for silhouette
        non_noise_mask = pred_valid != -1
        pred_non_noise = pred_valid[non_noise_mask]
        emb_non_noise = emb_valid[non_noise_mask]
        
        if len(set(pred_non_noise)) > 1:
            sample_size = min(10000, len(pred_non_noise))
            if sample_size < len(pred_non_noise):
                indices = np.random.choice(len(pred_non_noise), sample_size, replace=False)
                silhouette = silhouette_score(emb_non_noise[indices], pred_non_noise[indices])
            else:
                silhouette = silhouette_score(emb_non_noise, pred_non_noise)
            
            metrics['silhouette_score'] = float(silhouette)
            logger.info(f"  Silhouette: {silhouette:.4f}")
        else:
            metrics['silhouette_score'] = None
            logger.warning("  Not enough clusters for silhouette score")
        
        return metrics
    
    def compute_purity(self, predicted_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Compute cluster purity
        """
        total = 0
        correct = 0
        
        for cluster_id in set(predicted_labels):
            mask = predicted_labels == cluster_id
            cluster_true_labels = true_labels[mask]
            
            if len(cluster_true_labels) > 0:
                most_common_count = Counter(cluster_true_labels).most_common(1)[0][1]
                correct += most_common_count
                total += len(cluster_true_labels)
        
        return correct / total if total > 0 else 0.0
    
    def compute_per_cluster_metrics(self,
                                   clusters_df: pd.DataFrame,
                                   cluster_analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-cluster statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info("Computing Per-Cluster Metrics")
        logger.info(f"{'='*60}")
        
        cluster_stats = []
        
        for _, row in cluster_analysis_df.iterrows():
            cluster_id = row['cluster_id']
            
            if cluster_id == -1:
                continue
            
            cluster_seqs = clusters_df[clusters_df['cluster_id'] == cluster_id]
            probs = cluster_seqs['cluster_probability']
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': row['size'],
                'purity': row['purity'],
                'num_taxa': row['num_taxa'],
                'majority_name': row['majority_name'],
                'mean_probability': probs.mean(),
                'median_probability': probs.median(),
                'min_probability': probs.min(),
                'quality_score': row['purity'] * probs.mean()
            })
        
        if not cluster_stats:
            logger.warning("No clusters found (only noise). Skipping per-cluster metrics.")
            return pd.DataFrame()

        stats_df = pd.DataFrame(cluster_stats)
        stats_df = stats_df.sort_values('quality_score', ascending=False)
        logger.info(f"  Computed metrics for {len(stats_df)} clusters")
        
        return stats_df
    
    def compute_marker_specific_metrics(self,
                                       clusters_df: pd.DataFrame,
                                       metadata_df: pd.DataFrame) -> Dict:
        """
        Compute metrics for each marker separately
        """
        logger.info(f"\n{'='*60}")
        logger.info("Computing Marker-Specific Metrics")
        logger.info(f"{'='*60}")
        
        marker_metrics = {}
        
        # Use dtype=str again to be safe
        seqid_to_taxid = dict(zip(metadata_df['seqID'], metadata_df['taxID'].astype(str)))
        
        for marker in ['ITS', 'LSU', 'SSU']:
            logger.info(f"\nMarker: {marker}")
            
            marker_df = clusters_df[clusters_df['marker'] == marker].copy()
            if len(marker_df) == 0:
                logger.warning(f"  No sequences for {marker}")
                continue
            
            pred_labels = marker_df['cluster_id'].values
            true_labels = [seqid_to_taxid.get(sid, 'NA') for sid in marker_df['seqID']]
            true_labels = np.array(true_labels)
            
            # Robust valid mask
            valid_mask = [
                pd.notna(t) and t.upper() != 'NA' and t.upper() != 'NAN'
                for t in true_labels
            ]
            valid_mask = np.array(valid_mask)
            
            if valid_mask.sum() == 0:
                logger.warning(f"  No valid taxonomy for {marker}")
                continue
            
            pred_valid = pred_labels[valid_mask]
            true_valid = true_labels[valid_mask]
            
            ari = adjusted_rand_score(true_valid, pred_valid)
            nmi = normalized_mutual_info_score(true_valid, pred_valid)
            purity = self.compute_purity(pred_valid, true_valid)
            
            n_clusters = len(set(pred_valid)) - (1 if -1 in pred_valid else 0)
            n_noise = list(pred_valid).count(-1)
            
            marker_metrics[marker] = {
                'num_sequences': len(marker_df),
                'num_with_taxonomy': int(valid_mask.sum()),
                'num_clusters': n_clusters,
                'num_noise': n_noise,
                'noise_ratio': n_noise / len(pred_valid),
                'ari': float(ari),
                'nmi': float(nmi),
                'purity': float(purity)
            }
            
            logger.info(f"  Sequences: {len(marker_df):,}")
            logger.info(f"  Clusters: {n_clusters}")
            logger.info(f"  ARI: {ari:.4f}")
            logger.info(f"  NMI: {nmi:.4f}")
            logger.info(f"  Purity: {purity:.4f}")
        
        return marker_metrics
    
    def compute_novelty_metrics(self, novelty_dir: Path) -> Dict:
        """
        Compute novelty detection statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info("Computing Novelty Detection Metrics")
        logger.info(f"{'='*60}")
        
        metrics = {}
        
        candidates_file = novelty_dir / "novel_candidates.csv"
        if not candidates_file.exists():
            logger.warning("Novel candidates file not found")
            return metrics
        
        candidates_df = pd.read_csv(candidates_file)
        
        metrics['total_candidates'] = len(candidates_df)
        metrics['total_sequences_in_candidates'] = int(candidates_df['size'].sum())
        metrics['candidates_by_type'] = candidates_df['type'].value_counts().to_dict()
        
        metrics['candidate_size_stats'] = {
            'min': int(candidates_df['size'].min()),
            'max': int(candidates_df['size'].max()),
            'mean': float(candidates_df['size'].mean()),
            'median': float(candidates_df['size'].median())
        }
        
        clustered = candidates_df[candidates_df['type'].isin(['low_purity', 'unknown_taxonomy'])]
        if len(clustered) > 0:
            metrics['clustered_candidate_purity'] = {
                'mean': float(clustered['purity'].mean()),
                'median': float(clustered['purity'].median()),
                'min': float(clustered['purity'].min()),
                'max': float(clustered['purity'].max())
            }
        
        logger.info(f"  Total candidates: {metrics['total_candidates']}")
        logger.info(f"  Total sequences: {metrics['total_sequences_in_candidates']:,}")
        if 'candidates_by_type' in metrics:
            logger.info(f"\n  By type:")
            for ctype, count in metrics['candidates_by_type'].items():
                logger.info(f"    {ctype}: {count}")
        
        return metrics
    
    def generate_summary_report(self,
                              clustering_metrics: Dict,
                              marker_metrics: Dict,
                              cluster_stats: pd.DataFrame,
                              novelty_metrics: Dict) -> str:
        """
        Generate human-readable summary report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EVALUATION SUMMARY REPORT")
        lines.append("=" * 80)
        
        # Clustering metrics
        lines.append("\nOVERALL CLUSTERING METRICS")
        lines.append("-" * 80)
        lines.append(f"Adjusted Rand Index (ARI):              {clustering_metrics.get('adjusted_rand_index', 'N/A')}")
        lines.append(f"Normalized Mutual Information (NMI):    {clustering_metrics.get('normalized_mutual_information', 'N/A')}")
        lines.append(f"V-Measure:                              {clustering_metrics.get('v_measure', 'N/A')}")
        lines.append(f"Fowlkes-Mallows Score:                  {clustering_metrics.get('fowlkes_mallows_score', 'N/A')}")
        lines.append(f"Cluster Purity:                         {clustering_metrics.get('cluster_purity', 'N/A')}")
        
        if clustering_metrics.get('silhouette_score'):
            lines.append(f"Silhouette Score:                       {clustering_metrics['silhouette_score']:.4f}")
        
        # Marker-specific metrics
        lines.append("\nMARKER-SPECIFIC METRICS")
        lines.append("-" * 80)
        if marker_metrics:
            for marker, metrics in marker_metrics.items():
                lines.append(f"\n{marker}:")
                lines.append(f"  Sequences: {metrics['num_sequences']:,}")
                lines.append(f"  Clusters: {metrics['num_clusters']}")
                lines.append(f"  ARI: {metrics['ari']:.4f}")
                lines.append(f"  NMI: {metrics['nmi']:.4f}")
                lines.append(f"  Purity: {metrics['purity']:.4f}")
        else:
            lines.append("No marker-specific metrics computed (no valid taxonomy).")

        # Top clusters
        lines.append("\nTOP 10 HIGHEST QUALITY CLUSTERS")
        lines.append("-" * 80)
        if not cluster_stats.empty:
            top_clusters = cluster_stats.head(10)
            for _, row in top_clusters.iterrows():
                lines.append(f"Cluster {int(row['cluster_id']):4d} | Size: {int(row['size']):6,} | "
                            f"Purity: {row['purity']:.3f} | Quality: {row['quality_score']:.3f} | "
                            f"{row['majority_name']}")
        else:
            lines.append("No clusters found (only noise).")

        # Novelty detection
        if novelty_metrics:
            lines.append("\nNOVELTY DETECTION")
            lines.append("-" * 80)
            lines.append(f"Total candidates: {novelty_metrics['total_candidates']}")
            lines.append(f"Total sequences in candidates: {novelty_metrics['total_sequences_in_candidates']:,}")
            if 'candidates_by_type' in novelty_metrics:
                lines.append("\nBy type:")
                for ctype, count in novelty_metrics['candidates_by_type'].items():
                    lines.append(f"  {ctype}: {count}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def save_results(self,
                    clustering_metrics: Dict,
                    marker_metrics: Dict,
                    cluster_stats: pd.DataFrame,
                    novelty_metrics: Dict,
                    summary_report: str):
        """
        Save all evaluation results
        """
        logger.info(f"\n{'='*60}")
        logger.info("Saving Evaluation Results")
        logger.info(f"{'='*60}")
        
        # Save clustering metrics
        metrics_file = self.output_dir / "clustering_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(clustering_metrics, f, indent=2, default=str) # Add default=str
        logger.info(f"  Clustering metrics: {metrics_file}")
        
        # Save marker metrics
        marker_file = self.output_dir / "marker_metrics.json"
        with open(marker_file, 'w') as f:
            json.dump(marker_metrics, f, indent=2, default=str)
        logger.info(f"  Marker metrics: {marker_file}")
        
        # Save cluster statistics
        stats_file = self.output_dir / "cluster_statistics.csv"
        cluster_stats.to_csv(stats_file, index=False)
        logger.info(f"  Cluster statistics: {stats_file}")
        
        # Save novelty metrics
        if novelty_metrics:
            novelty_file = self.output_dir / "novelty_metrics.json"
            with open(novelty_file, 'w') as f:
                json.dump(novelty_metrics, f, indent=2, default=str)
            logger.info(f"  Novelty metrics: {novelty_file}")
        
        # Save summary report
        report_file = self.output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(summary_report)
        logger.info(f"  Summary report: {report_file}")
        
        # Print summary
        logger.info(f"\n{summary_report}")
    
    def run_complete_pipeline(self):
        """Execute complete evaluation pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION PIPELINE")
        logger.info(f"{'='*60}\n")
        
        # Load data
        clusters_df, metadata_df, cluster_analysis_df, embeddings = self.load_data()
        
        # Prepare labels
        predicted_labels, true_labels, valid_mask = self.prepare_labels(clusters_df, metadata_df)
        
        # Compute overall clustering metrics
        clustering_metrics = self.compute_clustering_metrics(
            predicted_labels, true_labels, valid_mask, embeddings
        )
        
        # Compute per-cluster metrics
        cluster_stats = self.compute_per_cluster_metrics(clusters_df, cluster_analysis_df)
        
        # Compute marker-specific metrics
        marker_metrics = self.compute_marker_specific_metrics(clusters_df, metadata_df)
        
        # Compute novelty metrics
        novelty_metrics = self.compute_novelty_metrics(self.novelty_dir)
        
        # Generate summary report
        summary_report = self.generate_summary_report(
            clustering_metrics, marker_metrics, cluster_stats, novelty_metrics
        )
        
        # Save results
        self.save_results(
            clustering_metrics, marker_metrics, cluster_stats, 
            novelty_metrics, summary_report
        )
        
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Files created:")
        logger.info(f"  - clustering_metrics.json")
        logger.info(f"  - marker_metrics.json")
        logger.info(f"  - cluster_statistics.csv")
        logger.info(f"  - novelty_metrics.json")
        logger.info(f"  - evaluation_report.txt")
        logger.info(f"{'='*60}\n")


def main():
    """Main execution"""
    
    # Initialize evaluation
    evaluation = EvaluationMetrics(
        clusters_dir="dataset/clusters",
        embeddings_dir="dataset/embeddings",
        novelty_dir="dataset/novelty",
        metadata_file="dataset/metadata.csv",
        output_dir="dataset/evaluation"
    )
    
    try:
        # Run complete pipeline
        evaluation.run_complete_pipeline()
        
        logger.info("Evaluation pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())