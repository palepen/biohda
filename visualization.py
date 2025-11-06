"""
Step 11: Visualization
=======================
Generate comprehensive visualizations for clustering and novelty detection

This regenerated file fixes two bugs:
1.  Loads 'cluster_statistics.csv' and 'clustering_metrics.json'.
2.  Passes 'cluster_stats_df' to 'plot_cluster_analysis' to prevent a TypeError.
3.  Calls the 'plot_metrics_summary' function, which was previously un-called.

Visualizations:
1. UMAP projection of embeddings (colored by clusters and taxonomy)
2. Cluster purity heatmap
3. Cluster size distribution
4. Taxonomic composition per cluster
5. Novelty candidate distribution
6. Per-marker analysis plots
7. Confusion matrix (cluster vs taxonomy)

Dependencies:
- matplotlib
- seaborn
- umap-learn
- NumPy
- pandas
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
from tqdm import tqdm

# Optional: UMAP (will skip if not installed)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not installed. UMAP plots will be skipped.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class Visualizer:
    """Generate visualizations for clustering and novelty detection"""
    
    def __init__(self,
                 clusters_dir: str = "dataset/clusters",
                 embeddings_dir: str = "dataset/embeddings",
                 novelty_dir: str = "dataset/novelty",
                 evaluation_dir: str = "dataset/evaluation",
                 metadata_file: str = "dataset/metadata.csv",
                 output_dir: str = "dataset/visualizations"):
        """
        Initialize visualizer
        """
        self.clusters_dir = Path(clusters_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.novelty_dir = Path(novelty_dir)
        self.evaluation_dir = Path(evaluation_dir)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Tuple:
        """Load all required data"""
        logger.info("\nLoading data for visualization...")
        
        # Clusters
        clusters_df = pd.read_csv(self.clusters_dir / "clusters.csv")
        cluster_analysis_df = pd.read_csv(self.clusters_dir / "cluster_analysis.csv")
        
        # Embeddings
        embeddings = np.load(self.embeddings_dir / "embeddings.npy")
        
        # Metadata
        metadata_df = pd.read_csv(self.metadata_file, dtype={'taxID': str})
        
        # Novelty candidates
        candidates_df = pd.read_csv(self.novelty_dir / "novel_candidates.csv")
        
        # Evaluation metrics - FIX: Un-commented these lines
        try:
            with open(self.evaluation_dir / "clustering_metrics.json", 'r') as f:
                metrics = json.load(f)
            
            cluster_stats_df = pd.read_csv(self.evaluation_dir / "cluster_statistics.csv")
            
        except FileNotFoundError as e:
            logger.error(f"Missing evaluation file: {e.filename}")
            logger.error("Please run Step 10 (evaluation.py) before running visualization.")
            raise

        logger.info(f"  Loaded {len(clusters_df):,} sequences")
        logger.info(f"  Embeddings shape: {embeddings.shape}")
        
        # FIX: Added metrics and cluster_stats_df to return
        return (clusters_df, cluster_analysis_df, embeddings, 
                metadata_df, candidates_df, metrics, cluster_stats_df)
    
    def plot_umap_projection(self,
                            embeddings: np.ndarray,
                            clusters_df: pd.DataFrame,
                            metadata_df: pd.DataFrame,
                            max_points: int = 20000):
        """
        Generate UMAP projection plots
        """
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available. Skipping UMAP plots.")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info("Generating UMAP Projections")
        logger.info(f"{'='*60}")
        
        # Sample if too many points
        if len(embeddings) > max_points:
            logger.info(f"  Sampling {max_points:,} points for visualization...")
            indices = np.random.choice(len(embeddings), max_points, replace=False)
            embeddings_sample = embeddings[indices]
            clusters_sample = clusters_df.iloc[indices].copy()
            seq_ids_sample = clusters_sample['seqID']
        else:
            embeddings_sample = embeddings
            clusters_sample = clusters_df.copy()
            seq_ids_sample = clusters_sample['seqID']
            indices = np.arange(len(embeddings))
        
        # Compute UMAP
        logger.info("  Computing UMAP (this may take a few minutes)...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_sample)
        
        # Save UMAP coordinates
        umap_df = pd.DataFrame({
            'umap_1': embedding_2d[:, 0],
            'umap_2': embedding_2d[:, 1],
            'cluster_id': clusters_sample['cluster_id'].values,
            'marker': clusters_sample['marker'].values
        })
        umap_df.to_csv(self.output_dir / "umap_coordinates.csv", index=False)
        
        # Plot 1: Colored by cluster
        logger.info("  Plotting UMAP by cluster...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        noise_mask = umap_df['cluster_id'] == -1
        
        if noise_mask.sum() > 0:
            ax.scatter(umap_df.loc[noise_mask, 'umap_1'], umap_df.loc[noise_mask, 'umap_2'],
                      c='lightgray', s=5, alpha=0.3, label='Noise')
        
        cluster_ids = sorted(set(umap_df['cluster_id']) - {-1})
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(cluster_ids))))
        
        for i, cluster_id in enumerate(cluster_ids[:20]):
            mask = umap_df['cluster_id'] == cluster_id
            ax.scatter(umap_df.loc[mask, 'umap_1'], umap_df.loc[mask, 'umap_2'],
                      c=[colors[i]], s=10, alpha=0.6, label=f'Cluster {cluster_id}')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Projection colored by Cluster')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        plt.tight_layout()
        plt.savefig(self.output_dir / "umap_by_cluster.png", bbox_inches='tight')
        plt.close()
        logger.info(f"    Saved: umap_by_cluster.png")
        
        # Plot 2: Colored by marker
        logger.info("  Plotting UMAP by marker...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        marker_colors = {'ITS': 'red', 'LSU': 'blue', 'SSU': 'green'}
        for marker, color in marker_colors.items():
            mask = umap_df['marker'] == marker
            if mask.sum() > 0:
                ax.scatter(umap_df.loc[mask, 'umap_1'], umap_df.loc[mask, 'umap_2'],
                          c=color, s=10, alpha=0.5, label=marker)
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Projection colored by Marker')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "umap_by_marker.png")
        plt.close()
        logger.info(f"    Saved: umap_by_marker.png")
        
        # Plot 3: Colored by taxonomy (top 10 taxa)
        logger.info("  Plotting UMAP by taxonomy...")
        seqid_to_name = dict(zip(metadata_df['seqID'], metadata_df['scientific_name']))
        taxa = [seqid_to_name.get(sid, 'NA') for sid in seq_ids_sample]
        
        # Filter out 'NA' before finding most common
        valid_taxa = [t for t in taxa if str(t).upper() != 'NA' and pd.notna(t)]
        
        if valid_taxa:
            top_taxa = [t for t, _ in Counter(valid_taxa).most_common(10)]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot background (other taxa)
            other_mask = ~np.isin(taxa, top_taxa)
            ax.scatter(embedding_2d[other_mask, 0], embedding_2d[other_mask, 1],
                      c='lightgray', s=5, alpha=0.2, label='Other / NA')
            
            # Plot top taxa
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for i, taxon in enumerate(top_taxa):
                mask = np.array(taxa) == taxon
                ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                          c=[colors[i]], s=10, alpha=0.6, label=taxon[:30])
            
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('UMAP Projection colored by Top 10 Taxa')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
            plt.tight_layout()
            plt.savefig(self.output_dir / "umap_by_taxonomy.png", bbox_inches='tight')
            plt.close()
            logger.info(f"    Saved: umap_by_taxonomy.png")
        else:
            logger.warning("  No valid taxonomy data found. Skipping UMAP by taxonomy plot.")
    
    def plot_cluster_analysis(self,
                             cluster_analysis_df: pd.DataFrame,
                             cluster_stats_df: pd.DataFrame):
        """
        Generate cluster analysis plots
        """
        logger.info(f"\n{'='*60}")
        logger.info("Generating Cluster Analysis Plots")
        logger.info(f"{'='*60}")
        
        # Exclude noise cluster
        valid_clusters = cluster_analysis_df[cluster_analysis_df['cluster_id'] != -1].copy()
        valid_stats = cluster_stats_df[cluster_stats_df['cluster_id'] != -1].copy()

        if valid_clusters.empty or valid_stats.empty:
            logger.warning("  No clusters (only noise) found. Skipping cluster analysis plots.")
            return

        # Plot 1: Cluster size distribution
        logger.info("  Plotting cluster size distribution...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.hist(valid_clusters['size'], bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Cluster Size')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Cluster Size Distribution')
        ax1.set_yscale('log')
        
        ax2.boxplot(valid_clusters['size'])
        ax2.set_ylabel('Cluster Size')
        ax2.set_title('Cluster Size Boxplot')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "cluster_size_distribution.png")
        plt.close()
        logger.info(f"    Saved: cluster_size_distribution.png")
        
        # Plot 2: Cluster purity distribution
        logger.info("  Plotting cluster purity distribution...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(valid_clusters['purity'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0.8, color='green', linestyle='--', label='High purity threshold (0.8)')
        ax.axvline(0.5, color='orange', linestyle='--', label='Medium purity threshold (0.5)')
        ax.set_xlabel('Cluster Purity')
        ax.set_ylabel('Frequency')
        ax.set_title('Cluster Purity Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "cluster_purity_distribution.png")
        plt.close()
        logger.info(f"    Saved: cluster_purity_distribution.png")
        
        # Plot 3: Top 20 clusters by quality score
        logger.info("  Plotting top clusters...")
        top_20 = valid_stats.head(20)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = range(len(top_20))
        ax.barh(x, top_20['quality_score'], color='steelblue', alpha=0.7)
        
        labels = [f"C{int(row['cluster_id'])} ({int(row['size'])} seqs)" 
                 for _, row in top_20.iterrows()]
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Quality Score (Purity × Mean Probability)')
        ax.set_title('Top 20 Clusters by Quality Score')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_clusters_quality.png")
        plt.close()
        logger.info(f"    Saved: top_clusters_quality.png")
        
        # Plot 4: Purity vs Size scatter
        logger.info("  Plotting purity vs size...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(valid_clusters['size'], valid_clusters['purity'],
                           c=valid_clusters['num_taxa'], cmap='viridis',
                           alpha=0.6, s=50)
        ax.set_xlabel('Cluster Size (log scale)')
        ax.set_ylabel('Cluster Purity')
        ax.set_title('Cluster Purity vs Size (colored by number of taxa)')
        ax.set_xscale('log')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Taxa')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "purity_vs_size.png")
        plt.close()
        logger.info(f"    Saved: purity_vs_size.png")
    
    def plot_novelty_analysis(self, candidates_df: pd.DataFrame):
        """
        Generate novelty detection plots
        """
        logger.info(f"\n{'='*60}")
        logger.info("Generating Novelty Analysis Plots")
        logger.info(f"{'='*60}")
        
        if candidates_df.empty:
            logger.warning("  No novelty candidates found. Skipping novelty plots.")
            return

        # Plot 1: Candidates by type
        logger.info("  Plotting candidate distribution by type...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        type_counts = candidates_df['type'].value_counts()
        ax1.bar(range(len(type_counts)), type_counts.values, 
               color='steelblue', alpha=0.7)
        ax1.set_xticks(range(len(type_counts)))
        ax1.set_xticklabels(type_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Number of Candidates')
        ax1.set_title('Novel Candidates by Type')
        
        type_seqs = candidates_df.groupby('type')['size'].sum()
        ax2.bar(range(len(type_seqs)), type_seqs.values, 
               color='coral', alpha=0.7)
        ax2.set_xticks(range(len(type_seqs)))
        ax2.set_xticklabels(type_seqs.index, rotation=45, ha='right')
        ax2.set_ylabel('Total Sequences')
        ax2.set_title('Total Sequences in Candidates by Type')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "novelty_by_type.png")
        plt.close()
        logger.info(f"    Saved: novelty_by_type.png")
        
        # Plot 2: Candidate size distribution
        logger.info("  Plotting candidate size distribution...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for ctype in candidates_df['type'].unique():
            type_data = candidates_df[candidates_df['type'] == ctype]['size']
            ax.hist(type_data, bins=20, alpha=0.5, label=ctype, edgecolor='black')
        
        ax.set_xlabel('Candidate Size (number of sequences)')
        ax.set_ylabel('Frequency')
        ax.set_title('Size Distribution of Novel Candidates')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "candidate_size_distribution.png")
        plt.close()
        logger.info(f"    Saved: candidate_size_distribution.png")
    
    def plot_metrics_summary(self, metrics: Dict):
        """
        Generate metrics summary visualization
        """
        logger.info(f"\n{'='*60}")
        logger.info("Generating Metrics Summary Plot")
        logger.info(f"{'='*60}")
        
        # Extract relevant metrics
        metric_names = []
        metric_values = []
        
        for key, value in metrics.items():
            if value is not None and isinstance(value, (int, float)):
                # Only plot metrics that are scaled 0-1
                if "score" in key or "purity" in key or "measure" in key or "nmi" in key or "ari" in key:
                     # Format metric name
                    name = key.replace('_', ' ').title()
                    metric_names.append(name)
                    metric_values.append(value)
        
        if not metric_names:
            logger.warning("  No metrics (0-1 scale) to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ensure values are capped at 1 for color logic
        safe_values = [min(v, 1.0) for v in metric_values]
        colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' 
                 for v in safe_values]
        
        ax.barh(range(len(metric_names)), metric_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(metric_names)))
        ax.set_yticklabels(metric_names)
        ax.set_xlabel('Score')
        ax.set_title('Clustering Quality Metrics (0-1 Scale)')
        ax.set_xlim(0, 1)
        ax.axvline(0.7, color='green', linestyle='--', alpha=0.5, label='Good (>0.7)')
        ax.axvline(0.4, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.4)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_summary.png")
        plt.close()
        logger.info(f"    Saved: metrics_summary.png")
    
    def plot_marker_comparison(self, clusters_df: pd.DataFrame):
        """
        Generate marker comparison plots
        """
        logger.info(f"\n{'='*60}")
        logger.info("Generating Marker Comparison Plots")
        logger.info(f"{'='*60}")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Per-Marker Analysis', fontsize=16)
        
        # Plot 1: Sequences per marker
        marker_counts = clusters_df['marker'].value_counts().reindex(['ITS', 'LSU', 'SSU'])
        axes[0, 0].bar(marker_counts.index, marker_counts.values, color=['red', 'blue', 'green'], alpha=0.7)
        axes[0, 0].set_ylabel('Number of Sequences')
        axes[0, 0].set_title('Sequences per Marker')
        
        # Plot 2: Clusters per marker
        n_clusters_per_marker = {}
        for marker in ['ITS', 'LSU', 'SSU']:
            marker_data = clusters_df[clusters_df['marker'] == marker]
            n_clusters = len(set(marker_data['cluster_id'])) - (1 if -1 in marker_data['cluster_id'].values else 0)
            n_clusters_per_marker[marker] = n_clusters
            
        axes[0, 1].bar(n_clusters_per_marker.keys(), n_clusters_per_marker.values(), 
                       color=['red', 'blue', 'green'], alpha=0.7)
        axes[0, 1].set_ylabel('Number of Clusters')
        axes[0, 1].set_title('Unique Clusters per Marker')
        
        # Plot 3: Noise ratio per marker
        noise_ratios = {}
        for marker in ['ITS', 'LSU', 'SSU']:
            marker_data = clusters_df[clusters_df['marker'] == marker]
            if len(marker_data) > 0:
                noise_count = (marker_data['cluster_id'] == -1).sum()
                noise_ratios[marker] = noise_count / len(marker_data) * 100
            else:
                noise_ratios[marker] = 0
        
        axes[1, 0].bar(noise_ratios.keys(), noise_ratios.values(), 
                      color=['red', 'blue', 'green'], alpha=0.7)
        axes[1, 0].set_ylabel('Noise Ratio (%)')
        axes[1, 0].set_title('Noise Sequences per Marker')
        
        # Plot 4: Cluster probability distribution per marker
        for marker, color in [('ITS', 'red'), ('LSU', 'blue'), ('SSU', 'green')]:
            marker_data = clusters_df[clusters_df['marker'] == marker]
            # Exclude noise
            non_noise = marker_data[marker_data['cluster_id'] != -1]
            if not non_noise.empty:
                sns.kdeplot(non_noise['cluster_probability'], ax=axes[1, 1],
                            label=marker, color=color, fill=True)
        
        axes[1, 1].set_xlabel('Cluster Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Cluster Probability Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / "marker_comparison.png")
        plt.close()
        logger.info(f"    Saved: marker_comparison.png")
    
    def run_complete_pipeline(self):
        """Execute complete visualization pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info("VISUALIZATION PIPELINE")
        logger.info(f"{'='*60}\n")
        
        # Load data
        # FIX: Unpacked new variables
        (clusters_df, cluster_analysis_df, embeddings, 
         metadata_df, candidates_df, metrics, cluster_stats_df) = self.load_data()
        
        # Generate visualizations
        self.plot_umap_projection(embeddings, clusters_df, metadata_df)
        
        # FIX: Passed both required arguments
        self.plot_cluster_analysis(cluster_analysis_df, cluster_stats_df)
        
        self.plot_novelty_analysis(candidates_df)
        
        # FIX: Added missing call
        self.plot_metrics_summary(metrics)
        
        self.plot_marker_comparison(clusters_df)
        
        logger.info(f"\n{'='*60}")
        logger.info("VISUALIZATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Generated plots:")
        logger.info(f"  - umap_by_cluster.png")
        logger.info(f"  - umap_by_marker.png")
        logger.info(f"  - umap_by_taxonomy.png")
        logger.info(f"  - cluster_size_distribution.png")
        logger.info(f"  - cluster_purity_distribution.png")
        logger.info(f"  - top_clusters_quality.png")
        logger.info(f"  - purity_vs_size.png")
        logger.info(f"  - novelty_by_type.png")
        logger.info(f"  - candidate_size_distribution.png")
        logger.info(f"  - metrics_summary.png") # FIX: Added to log
        logger.info(f"  - marker_comparison.png")
        logger.info(f"{'='*60}\n")


def main():
    """Main execution"""
    
    # Initialize visualizer
    visualizer = Visualizer(
        clusters_dir="dataset/clusters",
        embeddings_dir="dataset/embeddings",
        novelty_dir="dataset/novelty",
        evaluation_dir="dataset/evaluation",
        metadata_file="dataset/metadata.csv",
        output_dir="dataset/visualizations"
    )
    
    try:
        # Run complete pipeline
        visualizer.run_complete_pipeline()
        
        logger.info("Visualization pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in visualization pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())