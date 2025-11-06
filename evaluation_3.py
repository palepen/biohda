"""
Step 10: Unsupervised Evaluation Metrics
=========================================
Compute comprehensive UNSUPERVISED evaluation metrics:
- Cluster quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Novelty estimation
- Biodiversity indices (Shannon, Evenness)

Since no taxonomy TSV is available, this focuses on internal cluster quality
and biological diversity estimation.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
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


class UnsupervisedEvaluationMetrics:
    """Unsupervised evaluation metrics for clustering"""
    
    def __init__(self,
                 clusters_dir: str = "dataset/clusters",
                 embeddings_dir: str = "dataset/embeddings",
                 novelty_dir: str = "dataset/novelty",
                 validation_dir: str = "dataset/validation",
                 output_dir: str = "dataset/evaluation"):
        
        self.clusters_dir = Path(clusters_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.novelty_dir = Path(novelty_dir)
        self.validation_dir = Path(validation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Tuple:
        """Load all required data"""
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
        logger.info(f"  Embeddings: {embeddings.shape}")
        
        # Try to load BLAST validation results if available
        blast_results = None
        validated_file = self.validation_dir / "validated_candidates.csv"
        if validated_file.exists():
            blast_results = pd.read_csv(validated_file)
            logger.info(f"  BLAST validation: {len(blast_results)} candidates validated")
        else:
            logger.warning(f"  No BLAST validation found (run Step 9 first)")
        
        return clusters_df, cluster_analysis_df, embeddings, blast_results
    
    def compute_cluster_quality_metrics(self,
                                       predicted_labels: np.ndarray,
                                       embeddings: np.ndarray) -> Dict:
        """
        Compute unsupervised cluster quality metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info("UNSUPERVISED CLUSTER QUALITY METRICS")
        logger.info(f"{'='*60}")
        
        metrics = {}
        
        # Filter out noise points
        non_noise_mask = predicted_labels != -1
        pred_non_noise = predicted_labels[non_noise_mask]
        emb_non_noise = embeddings[non_noise_mask]
        
        n_clusters = len(set(pred_non_noise))
        n_noise = (predicted_labels == -1).sum()
        
        metrics['n_clusters'] = int(n_clusters)
        metrics['n_noise'] = int(n_noise)
        metrics['noise_ratio'] = float(n_noise / len(predicted_labels))
        
        logger.info(f"\nBasic Statistics:")
        logger.info(f"  Total sequences: {len(predicted_labels):,}")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Noise points: {n_noise:,} ({metrics['noise_ratio']:.2%})")
        
        if n_clusters < 2:
            logger.warning("Not enough clusters for quality metrics")
            return metrics
        
        # Silhouette Score
        logger.info("\nComputing Silhouette Score...")
        sample_size = min(10000, len(pred_non_noise))
        if sample_size < len(pred_non_noise):
            indices = np.random.choice(len(pred_non_noise), sample_size, replace=False)
            silhouette = silhouette_score(emb_non_noise[indices], pred_non_noise[indices])
        else:
            silhouette = silhouette_score(emb_non_noise, pred_non_noise)
        
        metrics['silhouette_score'] = float(silhouette)
        logger.info(f"  Silhouette Score: {silhouette:.4f}")
        logger.info(f"    Range: -1 to 1 (higher = better separation)")
        
        # Davies-Bouldin Index
        logger.info("\nComputing Davies-Bouldin Index...")
        db_score = davies_bouldin_score(emb_non_noise, pred_non_noise)
        metrics['davies_bouldin_index'] = float(db_score)
        logger.info(f"  Davies-Bouldin Index: {db_score:.4f}")
        logger.info(f"    Lower is better (0 = perfect)")
        
        # Calinski-Harabasz Index
        logger.info("\nComputing Calinski-Harabasz Score...")
        ch_score = calinski_harabasz_score(emb_non_noise, pred_non_noise)
        metrics['calinski_harabasz_score'] = float(ch_score)
        logger.info(f"  Calinski-Harabasz Score: {ch_score:.2f}")
        logger.info(f"    Higher is better")
        
        return metrics
    
    def compute_novelty_scores(self,
                              clusters_df: pd.DataFrame,
                              cluster_analysis_df: pd.DataFrame,
                              embeddings: np.ndarray) -> pd.DataFrame:
        """
        Compute novelty score for each cluster
        Novelty = 1 - max_cosine_similarity to other clusters
        """
        logger.info(f"\n{'='*60}")
        logger.info("NOVELTY ESTIMATION")
        logger.info(f"{'='*60}")
        
        novelty_scores = []
        
        valid_clusters = cluster_analysis_df[cluster_analysis_df['cluster_id'] != -1].copy()
        
        if len(valid_clusters) == 0:
            logger.warning("No valid clusters found")
            return pd.DataFrame()
        
        # Compute cluster centroids
        cluster_centroids = {}
        for _, row in valid_clusters.iterrows():
            cluster_id = row['cluster_id']
            mask = clusters_df['cluster_id'] == cluster_id
            cluster_indices = np.where(mask)[0]
            if len(cluster_indices) > 0:
                centroid = embeddings[cluster_indices].mean(axis=0)
                cluster_centroids[cluster_id] = centroid
        
        logger.info(f"Computing pairwise distances for {len(cluster_centroids)} clusters...")
        
        # For each cluster, find distance to nearest other cluster
        cluster_ids = list(cluster_centroids.keys())
        for cluster_id in tqdm(cluster_ids, desc="Computing novelty"):
            centroid = cluster_centroids[cluster_id]
            
            # Find distances to all other clusters
            other_centroids = np.array([cluster_centroids[cid] for cid in cluster_ids if cid != cluster_id])
            
            if len(other_centroids) > 0:
                # Compute cosine similarity
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
                other_norms = other_centroids / (np.linalg.norm(other_centroids, axis=1, keepdims=True) + 1e-10)
                similarities = np.dot(other_norms, centroid_norm)
                max_similarity = similarities.max()
                
                # Novelty = 1 - max_similarity
                novelty = 1.0 - max_similarity
            else:
                novelty = 1.0
            
            novelty_scores.append({
                'cluster_id': cluster_id,
                'novelty_score': float(novelty)
            })
        
        novelty_df = pd.DataFrame(novelty_scores)
        
        logger.info(f"\n  Novelty score range: {novelty_df['novelty_score'].min():.4f} - {novelty_df['novelty_score'].max():.4f}")
        logger.info(f"  Mean novelty: {novelty_df['novelty_score'].mean():.4f}")
        logger.info(f"  High novelty clusters (>0.5): {(novelty_df['novelty_score'] > 0.5).sum()}")
        
        return novelty_df
    
    def compute_biodiversity_indices(self,
                                    cluster_analysis_df: pd.DataFrame) -> Dict:
        """
        Compute Shannon diversity and evenness
        """
        logger.info(f"\n{'='*60}")
        logger.info("BIODIVERSITY INDICES")
        logger.info(f"{'='*60}")
        
        # Get cluster sizes (excluding noise)
        valid_clusters = cluster_analysis_df[cluster_analysis_df['cluster_id'] != -1]
        
        if len(valid_clusters) == 0:
            return {'shannon_diversity': 0.0, 'evenness': 0.0, 'richness': 0}
        
        cluster_sizes = valid_clusters['size'].values
        total = cluster_sizes.sum()
        
        # Shannon diversity: H' = -sum(pi * log(pi))
        proportions = cluster_sizes / total
        shannon = -np.sum(proportions * np.log(proportions + 1e-10))
        
        # Evenness: J' = H' / log(S)
        richness = len(cluster_sizes)
        max_shannon = np.log(richness)
        evenness = shannon / max_shannon if max_shannon > 0 else 0.0
        
        logger.info(f"\n  Richness (cluster count): {richness}")
        logger.info(f"  Shannon Diversity (H'): {shannon:.4f}")
        logger.info(f"    Range: 0 to log(S), higher = more diverse")
        logger.info(f"  Evenness (J'): {evenness:.4f}")
        logger.info(f"    Range: 0 to 1, 1 = perfectly even distribution")
        
        return {
            'richness': int(richness),
            'shannon_diversity': float(shannon),
            'evenness': float(evenness)
        }
    
    def generate_unified_cluster_table(self,
                                      cluster_analysis_df: pd.DataFrame,
                                      novelty_df: pd.DataFrame,
                                      blast_results: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Generate unified table combining cluster metrics, novelty, and BLAST results
        """
        logger.info(f"\n{'='*60}")
        logger.info("GENERATING UNIFIED CLUSTER TABLE")
        logger.info(f"{'='*60}")
        
        # Start with cluster analysis
        unified = cluster_analysis_df[cluster_analysis_df['cluster_id'] != -1].copy()
        
        # Merge novelty scores
        unified = unified.merge(novelty_df, on='cluster_id', how='left')
        
        # Calculate Shannon contribution per cluster
        total_seqs = unified['size'].sum()
        unified['shannon_contribution'] = -unified['size'] / total_seqs * np.log(unified['size'] / total_seqs + 1e-10)
        
        # Merge BLAST results if available
        if blast_results is not None:
            # Map candidate_id to cluster_id from novelty detection
            novelty_file = self.novelty_dir / "novel_candidates.csv"
            if novelty_file.exists():
                novelty_candidates = pd.read_csv(novelty_file)

                # --- START: FIX ---
                # Create a local copy of blast_results to modify
                blast_results_local = blast_results.copy()

                # Drop 'cluster_id' from blast_results_local if it exists
                # This prevents a column name collision during the merge,
                # as we only want the cluster_id from novelty_candidates.
                if 'cluster_id' in blast_results_local.columns:
                    blast_results_local = blast_results_local.drop(columns=['cluster_id'])
                
                # Perform the merge using the modified blast_results_local
                blast_with_cluster = blast_results_local.merge(
                    novelty_candidates[['candidate_id', 'cluster_id']], 
                    on='candidate_id', 
                    how='inner'
                )
                # --- END: FIX ---

                # This was the original problematic merge:
                # blast_with_cluster = blast_results.merge(
                #     novelty_candidates[['candidate_id', 'cluster_id']], 
                #     on='candidate_id', 
                #     how='inner'
                # )
                
                # This line will now work, as 'cluster_id' is guaranteed
                # to be the single column from novelty_candidates
                blast_with_cluster = blast_with_cluster[blast_with_cluster['cluster_id'] != -1]
                
                # Select relevant BLAST columns
                blast_cols = ['cluster_id', 'novelty_level', 'top_identity', 'top_hit_name']
                
                # Handle potential missing columns if blast_with_cluster is empty
                # or if the columns don't exist
                cols_to_use = [col for col in blast_cols if col in blast_with_cluster.columns]
                
                if 'cluster_id' in cols_to_use:
                    blast_data = blast_with_cluster[cols_to_use].drop_duplicates('cluster_id')
                    
                    # Merge with unified table
                    unified = unified.merge(blast_data, on='cluster_id', how='left', suffixes=('', '_blast'))
                    logger.info(f"  Merged BLAST results for {len(blast_data)} clusters")
                else:
                    logger.warning("  Could not merge BLAST results: 'cluster_id' column was missing after filtering.")

            else:
                logger.warning(f"  BLAST results available, but missing novelty file: {novelty_file}")
        
        logger.info(f"  Generated unified table with {len(unified)} clusters")
        
        return unified
    
    def save_results(self,
                    cluster_metrics: Dict,
                    biodiversity_metrics: Dict,
                    unified_table: pd.DataFrame):
        """Save all evaluation results"""
        logger.info(f"\n{'='*60}")
        logger.info("SAVING RESULTS")
        logger.info(f"{'='*60}")
        
        # Combine all metrics
        all_metrics = {
            'cluster_quality_metrics': cluster_metrics,
            'biodiversity_metrics': biodiversity_metrics
        }
        
        # Save metrics JSON
        metrics_file = self.output_dir / "unsupervised_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.info(f"  Metrics: {metrics_file}")
        
        # Save unified cluster table
        table_file = self.output_dir / "unified_cluster_table.csv"
        unified_table.to_csv(table_file, index=False)
        logger.info(f"  Unified table: {table_file}")
        
        # Generate summary report
        report = self.generate_summary_report(cluster_metrics, biodiversity_metrics, unified_table)
        
        report_file = self.output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"  Summary report: {report_file}")
        
        logger.info(f"\n{report}")
    
    def generate_summary_report(self,
                               cluster_metrics: Dict,
                               biodiversity_metrics: Dict,
                               unified_table: pd.DataFrame) -> str:
        """Generate comprehensive summary report"""
        lines = []
        lines.append("="*80)
        lines.append("UNSUPERVISED EVALUATION REPORT")
        lines.append("="*80)
        
        # Cluster Quality Metrics
        lines.append("\nCLUSTER QUALITY METRICS")
        lines.append("-"*80)
        lines.append(f"Number of Clusters:      {cluster_metrics.get('n_clusters', 'N/A')}")
        lines.append(f"Noise Points:            {cluster_metrics.get('n_noise', 'N/A'):,} ({cluster_metrics.get('noise_ratio', 0)*100:.1f}%)")
        lines.append(f"Silhouette Score:        {cluster_metrics.get('silhouette_score', 'N/A')}")
        lines.append(f"Davies-Bouldin Index:    {cluster_metrics.get('davies_bouldin_index', 'N/A')}")
        lines.append(f"Calinski-Harabasz Score: {cluster_metrics.get('calinski_harabasz_score', 'N/A')}")
        
        # Biodiversity Metrics
        lines.append("\nBIODIVERSITY INDICES")
        lines.append("-"*80)
        lines.append(f"Richness (cluster count): {biodiversity_metrics.get('richness', 'N/A')}")
        lines.append(f"Shannon Diversity (H'):   {biodiversity_metrics.get('shannon_diversity', 'N/A')}")
        lines.append(f"Evenness (J'):            {biodiversity_metrics.get('evenness', 'N/A')}")
        
        # Top clusters by size
        lines.append("\nTOP 10 CLUSTERS (by size)")
        lines.append("-"*80)
        if not unified_table.empty:
            top10 = unified_table.nlargest(10, 'size')
            for _, row in top10.iterrows():
                blast_info = ""
                if 'novelty_level' in row and pd.notna(row['novelty_level']):
                    blast_info = f" | BLAST: {row['novelty_level']} ({row.get('top_identity', 0):.1f}%)"
                
                lines.append(
                    f"C{int(row['cluster_id']):3d} | "
                    f"Size: {int(row['size']):6,} | "
                    f"Novelty: {row.get('novelty_score', 0):.3f} | "
                    f"Purity: {row.get('purity', 0):.3f}{blast_info}"
                )
        
        # High novelty clusters
        if 'novelty_score' in unified_table.columns:
            high_novelty = unified_table[unified_table['novelty_score'] > 0.5]
            lines.append(f"\nHIGH NOVELTY CLUSTERS (novelty > 0.5): {len(high_novelty)}")
            lines.append("-"*80)
            for _, row in high_novelty.nlargest(5, 'novelty_score').iterrows():
                lines.append(
                    f"C{int(row['cluster_id']):3d} | "
                    f"Novelty: {row['novelty_score']:.3f} | "
                    f"Size: {int(row['size']):,}"
                )
        
        lines.append("\n" + "="*80)
        
        return "\n".join(lines)
    
    def run_complete_pipeline(self):
        """Execute complete evaluation pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info("UNSUPERVISED EVALUATION PIPELINE")
        logger.info(f"{'='*60}\n")
        
        # Load data
        clusters_df, cluster_analysis_df, embeddings, blast_results = self.load_data()
        
        # Get predicted labels
        predicted_labels = clusters_df['cluster_id'].values
        
        # Compute cluster quality metrics
        cluster_metrics = self.compute_cluster_quality_metrics(predicted_labels, embeddings)
        
        # Compute novelty scores
        novelty_df = self.compute_novelty_scores(clusters_df, cluster_analysis_df, embeddings)
        
        # Compute biodiversity indices
        biodiversity_metrics = self.compute_biodiversity_indices(cluster_analysis_df)
        
        # Generate unified table
        unified_table = self.generate_unified_cluster_table(
            cluster_analysis_df, novelty_df, blast_results
        )
        
        # Save results
        self.save_results(cluster_metrics, biodiversity_metrics, unified_table)
        
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"{'='*60}")


def main():
    """Main execution"""
    
    evaluator = UnsupervisedEvaluationMetrics(
        clusters_dir="dataset/clusters",
        embeddings_dir="dataset/embeddings",
        novelty_dir="dataset/novelty",
        validation_dir="dataset/validation",
        output_dir="dataset/evaluation"
    )
    
    try:
        evaluator.run_complete_pipeline()
        logger.info("Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())