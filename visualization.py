"""
Step 11: Enhanced Visualization
Comprehensive visualizations including training curves, confusion matrices, and performance metrics
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
from tqdm import tqdm
import yaml
from typing import Dict, Tuple

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not installed. UMAP plots will be skipped.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class EnhancedVisualizer:
    """Generate comprehensive visualizations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.clusters_dir = Path(config['paths']['clusters_dir'])
        self.embeddings_dir = Path(config['paths']['embeddings_dir'])
        self.novelty_dir = Path(config['paths']['novelty_dir'])
        self.evaluation_dir = Path(config['paths']['evaluation_dir'])
        self.models_dir = Path(config['paths']['models_dir'])
        self.metadata_file = Path(config['paths']['processed_dir']).parent / "metadata.csv"
        self.output_dir = Path(config['paths']['visualization_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_umap_points = config['visualization']['max_umap_points']
        self.top_clusters_display = config['visualization']['top_clusters_display']
    
    def load_data(self):
        """Load all required data"""
        logger.info("\nLoading data for visualization...")
        
        clusters_df = pd.read_csv(self.clusters_dir / "clusters.csv")
        cluster_analysis_df = pd.read_csv(self.clusters_dir / "cluster_analysis.csv")
        embeddings = np.load(self.embeddings_dir / "embeddings.npy")
        metadata_df = pd.read_csv(self.metadata_file, dtype={'taxID': str})
        
        candidates_file = self.novelty_dir / "novel_candidates.csv"
        candidates_df = pd.read_csv(candidates_file) if candidates_file.exists() else pd.DataFrame()
        
        with open(self.evaluation_dir / "clustering_metrics.json", 'r') as f:
            metrics = json.load(f)
        
        cluster_stats_df = pd.read_csv(self.evaluation_dir / "cluster_statistics.csv")
        
        logger.info(f"  Loaded {len(clusters_df):,} sequences")
        logger.info(f"  Embeddings shape: {embeddings.shape}")
        
        return (clusters_df, cluster_analysis_df, embeddings, 
                metadata_df, candidates_df, metrics, cluster_stats_df)
    
    def plot_training_curves(self):
        """Plot training loss and validation curves"""
        logger.info("\nPlotting training curves...")
        
        history_file = self.models_dir / "training_history.json"
        if not history_file.exists():
            logger.warning("Training history not found, skipping training curves")
            return
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        min_val_loss_epoch = np.argmin(history['val_loss']) + 1
        min_val_loss = np.min(history['val_loss'])
        ax.axvline(min_val_loss_epoch, color='green', linestyle='--', 
                   label=f'Best Epoch ({min_val_loss_epoch})', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png")
        plt.close()
        logger.info("  Saved: training_curves.png")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix heatmap"""
        logger.info("\nPlotting confusion matrix...")
        
        cm_file = self.evaluation_dir / "confusion_matrix.npy"
        labels_file = self.evaluation_dir / "confusion_matrix_labels.json"
        
        if not cm_file.exists() or not labels_file.exists():
            logger.warning("Confusion matrix data not found, skipping")
            return
        
        cm = np.load(cm_file)
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        im = ax.imshow(cm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        ax.set_xticks(np.arange(len(labels['cluster_labels'])))
        ax.set_yticks(np.arange(len(labels['taxa_labels'])))
        ax.set_xticklabels([f"C{c}" for c in labels['cluster_labels']], rotation=45, ha='right')
        ax.set_yticklabels([t[:30] for t in labels['taxa_labels']])
        
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Taxonomic Class (Genus)')
        ax.set_title('Confusion Matrix: Top 20 Taxa vs Top 20 Clusters')
        
        plt.colorbar(im, ax=ax, label='Number of Sequences')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png")
        plt.close()
        logger.info("  Saved: confusion_matrix.png")
    
    def plot_external_validation_metrics(self, metrics: Dict):
        """Plot external validation metrics comparison"""
        logger.info("\nPlotting external validation metrics...")
        
        if 'external_validation' not in metrics:
            logger.warning("External validation metrics not found, skipping")
            return
        
        genus_metrics = metrics['external_validation'].get('genus_level', {})
        family_metrics = metrics['external_validation'].get('family_level', {})
        
        metric_names = ['adjusted_rand_index', 'normalized_mutual_info', 'v_measure', 
                       'homogeneity_score', 'completeness_score', 'fowlkes_mallows_index']
        display_names = ['ARI', 'NMI', 'V-Measure', 'Homogeneity', 'Completeness', 'FMI']
        
        genus_values = [genus_metrics.get(m, 0) for m in metric_names]
        family_values = [family_metrics.get(m, 0) for m in metric_names]
        
        x = np.arange(len(display_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width/2, genus_values, width, label='Genus Level', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, family_values, width, label='Family Level', color='coral', alpha=0.8)
        
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (>0.7)')
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.4)')
        
        ax.set_ylabel('Score')
        ax.set_title('External Validation Metrics (Clustering vs Ground Truth)')
        ax.set_xticks(x)
        ax.set_xticklabels(display_names)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "external_validation_metrics.png")
        plt.close()
        logger.info("  Saved: external_validation_metrics.png")
    
    def plot_novelty_detection_performance(self, metrics: Dict):
        """Plot novelty detection precision, recall, F1"""
        logger.info("\nPlotting novelty detection performance...")
        
        if 'novelty_detection' not in metrics or not metrics['novelty_detection']:
            logger.warning("Novelty detection metrics not found, skipping")
            return
        
        nov_metrics = metrics['novelty_detection']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        metric_values = [
            nov_metrics.get('precision', 0),
            nov_metrics.get('recall', 0),
            nov_metrics.get('f1_score', 0),
            nov_metrics.get('accuracy', 0)
        ]
        
        colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in metric_values]
        
        ax1.barh(metric_names, metric_values, color=colors, alpha=0.7)
        ax1.set_xlabel('Score')
        ax1.set_title('Novelty Detection Performance Metrics')
        ax1.set_xlim(0, 1)
        ax1.axvline(0.7, color='green', linestyle='--', alpha=0.3)
        ax1.axvline(0.4, color='orange', linestyle='--', alpha=0.3)
        
        confusion = [
            nov_metrics.get('true_positives', 0),
            nov_metrics.get('false_positives', 0),
            nov_metrics.get('true_negatives', 0),
            nov_metrics.get('false_negatives', 0)
        ]
        labels = ['TP', 'FP', 'TN', 'FN']
        colors_cm = ['green', 'red', 'lightgreen', 'lightcoral']
        
        ax2.bar(labels, confusion, color=colors_cm, alpha=0.7)
        ax2.set_ylabel('Count')
        ax2.set_title('Novelty Detection Confusion Matrix')
        ax2.set_yscale('log')
        
        for i, v in enumerate(confusion):
            ax2.text(i, v, f'{v:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "novelty_detection_performance.png")
        plt.close()
        logger.info("  Saved: novelty_detection_performance.png")
    
    def plot_cluster_quality_metrics(self, cluster_stats_df: pd.DataFrame):
        """Plot cluster quality metrics distribution"""
        logger.info("\nPlotting cluster quality metrics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cluster Quality Metrics Distribution', fontsize=16)
        
        axes[0, 0].hist(cluster_stats_df['mean_probability'], bins=30, 
                       color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0.8, color='green', linestyle='--', label='High (>0.8)')
        axes[0, 0].axvline(0.5, color='orange', linestyle='--', label='Medium (>0.5)')
        axes[0, 0].set_xlabel('Mean Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Cluster Coherence (Mean Probability)')
        axes[0, 0].legend()
        
        axes[0, 1].scatter(cluster_stats_df['size'], cluster_stats_df['mean_probability'],
                          c=cluster_stats_df['taxonomy_coverage'], cmap='viridis',
                          alpha=0.6, s=50)
        axes[0, 1].set_xlabel('Cluster Size (log scale)')
        axes[0, 1].set_ylabel('Mean Probability')
        axes[0, 1].set_title('Size vs Coherence (colored by taxonomy coverage)')
        axes[0, 1].set_xscale('log')
        
        axes[1, 0].hist(cluster_stats_df['taxonomy_coverage'], bins=30,
                       color='coral', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Taxonomy Coverage Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Taxonomy Annotation Coverage')
        
        top_20 = cluster_stats_df.head(20)
        quality_score = top_20['mean_probability'] * top_20['taxonomy_coverage']
        
        y_pos = np.arange(len(top_20))
        axes[1, 1].barh(y_pos, quality_score, color='steelblue', alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([f"C{int(cid)}" for cid in top_20['cluster_id']], fontsize=8)
        axes[1, 1].set_xlabel('Quality Score (Prob × Coverage)')
        axes[1, 1].set_title('Top 20 Clusters by Quality')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "cluster_quality_metrics.png")
        plt.close()
        logger.info("  Saved: cluster_quality_metrics.png")
    
    def plot_metrics_summary(self, metrics: Dict):
        """Enhanced metrics summary with multiple categories"""
        logger.info("\nPlotting comprehensive metrics summary...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comprehensive Clustering Quality Metrics', fontsize=16)
        
        if 'external_validation' in metrics and 'genus_level' in metrics['external_validation']:
            ext = metrics['external_validation']['genus_level']
            metric_names = ['ARI', 'NMI', 'V-Measure', 'Homogeneity', 'Completeness']
            metric_values = [
                ext.get('adjusted_rand_index', 0),
                ext.get('normalized_mutual_info', 0),
                ext.get('v_measure', 0),
                ext.get('homogeneity_score', 0),
                ext.get('completeness_score', 0)
            ]
            
            colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in metric_values]
            
            axes[0, 0].barh(metric_names, metric_values, color=colors, alpha=0.7)
            axes[0, 0].set_xlabel('Score')
            axes[0, 0].set_title('External Validation (vs Ground Truth)')
            axes[0, 0].set_xlim(0, 1)
            axes[0, 0].axvline(0.7, color='green', linestyle='--', alpha=0.3)
        
        if 'cluster_purity' in metrics and 'genus_level' in metrics['cluster_purity']:
            pur = metrics['cluster_purity']['genus_level']
            purity_metrics = ['Mean Purity', 'Median Purity']
            purity_values = [pur.get('mean_purity', 0), pur.get('median_purity', 0)]
            
            axes[0, 1].bar(purity_metrics, purity_values, color='steelblue', alpha=0.7)
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Cluster Purity')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].axhline(0.8, color='green', linestyle='--', alpha=0.3)
            
            for i, v in enumerate(purity_values):
                axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        if 'summary' in metrics:
            summ = metrics['summary']
            labels = ['Clustered', 'Noise']
            sizes = [summ['total_sequences'] - summ['noise_sequences'], summ['noise_sequences']]
            colors_pie = ['steelblue', 'lightgray']
            
            axes[1, 0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title(f'Sequence Distribution\n({summ["total_clusters"]} clusters)')
        
        if 'novelty_detection' in metrics and metrics['novelty_detection']:
            nov = metrics['novelty_detection']
            nov_metrics = ['Precision', 'Recall', 'F1']
            nov_values = [nov.get('precision', 0), nov.get('recall', 0), nov.get('f1_score', 0)]
            
            colors_nov = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in nov_values]
            
            axes[1, 1].bar(nov_metrics, nov_values, color=colors_nov, alpha=0.7)
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Novelty Detection')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axhline(0.7, color='green', linestyle='--', alpha=0.3)
            
            for i, v in enumerate(nov_values):
                axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comprehensive_metrics_summary.png")
        plt.close()
        logger.info("  Saved: comprehensive_metrics_summary.png")
    
    def plot_umap_projection(self, embeddings, clusters_df, metadata_df):
        """UMAP projection (same as before)"""
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available. Skipping UMAP plots.")
            return
        
        logger.info("\nGenerating UMAP projections...")
        
        if len(embeddings) > self.max_umap_points:
            indices = np.random.choice(len(embeddings), self.max_umap_points, replace=False)
            embeddings_sample = embeddings[indices]
            clusters_sample = clusters_df.iloc[indices].copy()
        else:
            embeddings_sample = embeddings
            clusters_sample = clusters_df.copy()
        
        logger.info("  Computing UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_sample)
        
        umap_df = pd.DataFrame({
            'umap_1': embedding_2d[:, 0],
            'umap_2': embedding_2d[:, 1],
            'cluster_id': clusters_sample['cluster_id'].values,
            'marker': clusters_sample['marker'].values
        })
        
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
                      c=[colors[i]], s=10, alpha=0.6, label=f'C{cluster_id}')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Projection (Colored by Cluster)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        plt.tight_layout()
        plt.savefig(self.output_dir / "umap_by_cluster.png", bbox_inches='tight')
        plt.close()
        logger.info("  Saved: umap_by_cluster.png")
    
    def run_complete_pipeline(self):
        """Execute complete visualization pipeline"""
        logger.info("\nVISUALIZATION PIPELINE")
        
        (clusters_df, cluster_analysis_df, embeddings, 
         metadata_df, candidates_df, metrics, cluster_stats_df) = self.load_data()
        
        self.plot_training_curves()
        self.plot_confusion_matrix()
        self.plot_external_validation_metrics(metrics)
        self.plot_novelty_detection_performance(metrics)
        self.plot_cluster_quality_metrics(cluster_stats_df)
        self.plot_metrics_summary(metrics)
        self.plot_umap_projection(embeddings, clusters_df, metadata_df)
        
        logger.info("\nVisualization complete")
        logger.info(f"Output directory: {self.output_dir}")


def main():
    visualizer = EnhancedVisualizer(config_path="config.yaml")
    
    try:
        visualizer.run_complete_pipeline()
        logger.info("Visualization pipeline completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())