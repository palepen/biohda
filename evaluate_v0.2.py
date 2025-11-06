import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plotting will be skipped.")


class TaxonomicDatabase:
    """Handler for taxonomic database with hierarchy support."""
    
    def __init__(self, taxa_file):
        """
        Load taxonomic database from file.
        
        Args:
            taxa_file: Path to tab-separated file with format:
                       SeqID\tTaxID\tScientificName\tFullTaxonomy
        """
        self.df = pd.read_csv(taxa_file, sep='\t', header=None,
                              names=['seq_id', 'tax_id', 'sci_name', 'full_taxonomy'])
        
        # Parse taxonomic hierarchy
        self.df['taxonomy_list'] = self.df['full_taxonomy'].apply(
            lambda x: x.split(';') if pd.notna(x) else []
        )
        
        # Extract taxonomic ranks (assuming standard hierarchy)
        self._extract_taxonomic_ranks()
        
        print(f"Loaded {len(self.df)} sequences from taxonomic database")
        print(f"Known taxa: {self.df['tax_id'].notna().sum()}")
        print(f"Unknown/N/A: {self.df['tax_id'].isna().sum()}")
    
    def _extract_taxonomic_ranks(self):
        """Extract common taxonomic ranks from full taxonomy."""
        ranks = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
        for rank in ranks:
            self.df[rank] = None
        
        for idx, row in self.df.iterrows():
            tax_list = row['taxonomy_list']
            if len(tax_list) > 0:
                # Map based on position (adjust based on your taxonomy format)
                if len(tax_list) >= 1:
                    self.df.at[idx, 'domain'] = tax_list[0] if len(tax_list) > 0 else None
                if len(tax_list) >= 2:
                    self.df.at[idx, 'kingdom'] = tax_list[1] if len(tax_list) > 1 else None
                if len(tax_list) >= 3:
                    self.df.at[idx, 'phylum'] = tax_list[2] if len(tax_list) > 2 else None
                if len(tax_list) >= 4:
                    self.df.at[idx, 'class'] = tax_list[3] if len(tax_list) > 3 else None
                if len(tax_list) >= 5:
                    self.df.at[idx, 'order'] = tax_list[4] if len(tax_list) > 4 else None
                if len(tax_list) >= 6:
                    self.df.at[idx, 'family'] = tax_list[5] if len(tax_list) > 5 else None
                if len(tax_list) >= 7:
                    self.df.at[idx, 'genus'] = tax_list[6] if len(tax_list) > 6 else None
                if len(tax_list) >= 8:
                    self.df.at[idx, 'species'] = tax_list[7] if len(tax_list) > 7 else None
    
    def get_taxonomy(self, seq_id):
        """Get taxonomic information for a sequence ID."""
        result = self.df[self.df['seq_id'] == seq_id]
        if len(result) == 0:
            return None
        return result.iloc[0].to_dict()
    
    def is_known(self, seq_id):
        """Check if sequence has known taxonomy."""
        result = self.df[self.df['seq_id'] == seq_id]
        if len(result) == 0:
            return False
        return pd.notna(result.iloc[0]['tax_id']) and result.iloc[0]['tax_id'] != '0'


class HDBScanClusterEvaluator:
    """Comprehensive evaluation of HDBScan clustering results."""
    
    def __init__(self, embeddings, labels, seq_ids, taxa_db):
        """
        Initialize evaluator.
        
        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            labels: numpy array of cluster labels from HDBScan
            seq_ids: list of sequence identifiers
            taxa_db: TaxonomicDatabase instance
        """
        self.embeddings = embeddings
        self.labels = labels
        self.seq_ids = seq_ids
        self.taxa_db = taxa_db
        
        # Filter out noise points (label = -1) for some metrics
        self.valid_mask = labels != -1
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_noise = sum(labels == -1)
        
        print(f"\nClustering Summary:")
        print(f"Total sequences: {len(labels)}")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Noise points: {self.n_noise} ({100*self.n_noise/len(labels):.2f}%)")
    
    def compute_clustering_metrics(self):
        """Compute standard clustering evaluation metrics."""
        metrics = {}
        
        if self.n_clusters < 2:
            print("Warning: Less than 2 clusters found. Skipping clustering metrics.")
            return metrics
        
        # Silhouette Score (higher is better, range: [-1, 1])
        if sum(self.valid_mask) > self.n_clusters:
            try:
                metrics['silhouette_score'] = silhouette_score(
                    self.embeddings[self.valid_mask],
                    self.labels[self.valid_mask]
                )
            except:
                metrics['silhouette_score'] = None
        
        # Davies-Bouldin Index (lower is better, range: [0, inf])
        if sum(self.valid_mask) > self.n_clusters:
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(
                    self.embeddings[self.valid_mask],
                    self.labels[self.valid_mask]
                )
            except:
                metrics['davies_bouldin_score'] = None
        
        # Calinski-Harabasz Index (higher is better)
        if sum(self.valid_mask) > self.n_clusters:
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                    self.embeddings[self.valid_mask],
                    self.labels[self.valid_mask]
                )
            except:
                metrics['calinski_harabasz_score'] = None
        
        return metrics
    
    def compute_cluster_statistics(self):
        """Compute per-cluster statistics."""
        cluster_stats = []
        
        for cluster_id in sorted(set(self.labels)):
            if cluster_id == -1:
                continue
            
            mask = self.labels == cluster_id
            cluster_embeds = self.embeddings[mask]
            cluster_seqs = [self.seq_ids[i] for i in np.where(mask)[0]]
            
            # Compute centroid
            centroid = cluster_embeds.mean(axis=0)
            
            # Compute intra-cluster distances
            distances = cdist(cluster_embeds, [centroid], metric='euclidean').flatten()
            
            # Compute inter-cluster distance (to nearest cluster)
            other_clusters = [c for c in set(self.labels) if c != cluster_id and c != -1]
            min_inter_dist = float('inf')
            
            for other_id in other_clusters:
                other_mask = self.labels == other_id
                other_centroid = self.embeddings[other_mask].mean(axis=0)
                dist = np.linalg.norm(centroid - other_centroid)
                min_inter_dist = min(min_inter_dist, dist)
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_seqs),
                'mean_intra_dist': distances.mean(),
                'std_intra_dist': distances.std(),
                'max_intra_dist': distances.max(),
                'min_inter_dist': min_inter_dist if min_inter_dist != float('inf') else None,
                'compactness': distances.std() / (distances.mean() + 1e-10),
                'separation': min_inter_dist / (distances.mean() + 1e-10) if min_inter_dist != float('inf') else None
            }
            
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def validate_against_taxonomy(self):
        """Validate clusters against known taxonomy."""
        cluster_taxonomy = []
        
        for cluster_id in sorted(set(self.labels)):
            mask = self.labels == cluster_id
            cluster_seqs = [self.seq_ids[i] for i in np.where(mask)[0]]
            
            # Get taxonomic information for sequences in cluster
            taxa_info = []
            unknown_seqs = []
            
            for seq_id in cluster_seqs:
                if self.taxa_db.is_known(seq_id):
                    taxa_info.append(self.taxa_db.get_taxonomy(seq_id))
                else:
                    unknown_seqs.append(seq_id)
            
            # Analyze taxonomic composition
            result = {
                'cluster_id': cluster_id,
                'size': len(cluster_seqs),
                'known_taxa': len(taxa_info),
                'unknown_taxa': len(unknown_seqs),
                'pct_unknown': 100 * len(unknown_seqs) / len(cluster_seqs)
            }
            
            # Compute taxonomic purity at different ranks
            if len(taxa_info) > 0:
                for rank in ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                    rank_values = [t.get(rank) for t in taxa_info if t.get(rank) is not None]
                    if len(rank_values) > 0:
                        counter = Counter(rank_values)
                        most_common = counter.most_common(1)[0]
                        result[f'{rank}_purity'] = most_common[1] / len(rank_values)
                        result[f'{rank}_dominant'] = most_common[0]
                        result[f'{rank}_unique'] = len(counter)
                        
                        # Compute entropy (diversity measure)
                        probs = np.array(list(counter.values())) / sum(counter.values())
                        result[f'{rank}_entropy'] = entropy(probs)
            
            # Flag potential novel taxa
            result['potential_novel'] = self._assess_novelty(result)
            
            cluster_taxonomy.append(result)
        
        return pd.DataFrame(cluster_taxonomy)
    
    def _assess_novelty(self, cluster_result):
        """
        Assess if cluster might contain novel taxa.
        
        Criteria:
        1. High percentage of unknown sequences (>50%)
        2. Low taxonomic purity at species/genus level
        3. High taxonomic diversity
        """
        if cluster_result.get('pct_unknown', 0) > 50:
            return 'high_confidence_novel'
        
        if cluster_result.get('pct_unknown', 0) > 20:
            # Check genus purity
            genus_purity = cluster_result.get('genus_purity', 1.0)
            if genus_purity < 0.7:
                return 'moderate_confidence_novel'
        
        if cluster_result.get('genus_unique', 0) > 5:
            return 'diverse_cluster_potential_novel'
        
        return 'likely_known'
    
    def identify_novel_taxa_candidates(self, taxonomy_df):
        """Identify and report potential novel taxa."""
        novel_candidates = []
        
        for _, row in taxonomy_df.iterrows():
            if row['potential_novel'] in ['high_confidence_novel', 'moderate_confidence_novel']:
                mask = self.labels == row['cluster_id']
                cluster_seqs = [self.seq_ids[i] for i in np.where(mask)[0]]
                
                unknown_seqs = [
                    seq_id for seq_id in cluster_seqs 
                    if not self.taxa_db.is_known(seq_id)
                ]
                
                novel_candidates.append({
                    'cluster_id': row['cluster_id'],
                    'confidence': row['potential_novel'],
                    'cluster_size': row['size'],
                    'unknown_count': len(unknown_seqs),
                    'unknown_seqs': unknown_seqs,
                    'dominant_family': row.get('family_dominant', 'N/A'),
                    'family_purity': row.get('family_purity', 0),
                    'genus_diversity': row.get('genus_unique', 0)
                })
        
        return pd.DataFrame(novel_candidates)
    
    def generate_report(self, output_file='cluster_evaluation_report.txt'):
        """Generate comprehensive evaluation report."""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HDBScan Cluster Evaluation and Taxonomic Validation Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Clustering metrics
            f.write("1. CLUSTERING QUALITY METRICS\n")
            f.write("-" * 80 + "\n")
            metrics = self.compute_clustering_metrics()
            for metric, value in metrics.items():
                if value is not None:
                    f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # Cluster statistics
            f.write("2. CLUSTER STATISTICS\n")
            f.write("-" * 80 + "\n")
            cluster_stats = self.compute_cluster_statistics()
            f.write(cluster_stats.to_string(index=False))
            f.write("\n\n")
            
            # Taxonomic validation
            f.write("3. TAXONOMIC VALIDATION\n")
            f.write("-" * 80 + "\n")
            taxonomy_df = self.validate_against_taxonomy()
            f.write(taxonomy_df.to_string(index=False))
            f.write("\n\n")
            
            # Novel taxa candidates
            f.write("4. POTENTIAL NOVEL TAXA CANDIDATES\n")
            f.write("-" * 80 + "\n")
            novel_df = self.identify_novel_taxa_candidates(taxonomy_df)
            if len(novel_df) > 0:
                for _, row in novel_df.iterrows():
                    f.write(f"\nCluster {row['cluster_id']}:\n")
                    f.write(f"  Confidence: {row['confidence']}\n")
                    f.write(f"  Size: {row['cluster_size']}\n")
                    f.write(f"  Unknown sequences: {row['unknown_count']}\n")
                    f.write(f"  Dominant family: {row['dominant_family']}\n")
                    f.write(f"  Family purity: {row['family_purity']:.3f}\n")
                    f.write(f"  Genus diversity: {row['genus_diversity']}\n")
                    f.write(f"  Sample unknown seq IDs: {', '.join(row['unknown_seqs'][:5])}\n")
            else:
                f.write("No high-confidence novel taxa candidates identified.\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\nReport saved to: {output_file}")
        return taxonomy_df, novel_df
    
    def plot_results(self, output_prefix='cluster_eval'):
        """Generate visualization plots."""
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Skipping plots.")
            return
        
        # Plot 1: Cluster size distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        cluster_ids = [c for c in self.labels if c != -1]
        cluster_counts = Counter(cluster_ids)
        
        axes[0, 0].bar(cluster_counts.keys(), cluster_counts.values())
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Sequences')
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Taxonomic purity
        taxonomy_df = self.validate_against_taxonomy()
        if 'genus_purity' in taxonomy_df.columns:
            axes[0, 1].scatter(taxonomy_df['size'], taxonomy_df['genus_purity'])
            axes[0, 1].set_xlabel('Cluster Size')
            axes[0, 1].set_ylabel('Genus Purity')
            axes[0, 1].set_title('Taxonomic Purity vs Cluster Size')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Unknown sequences percentage
        axes[1, 0].bar(taxonomy_df['cluster_id'], taxonomy_df['pct_unknown'])
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('% Unknown Sequences')
        axes[1, 0].set_title('Percentage of Unknown Sequences per Cluster')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=50, color='r', linestyle='--', label='50% threshold')
        axes[1, 0].legend()
        
        # Plot 4: Novelty assessment
        novelty_counts = taxonomy_df['potential_novel'].value_counts()
        axes[1, 1].pie(novelty_counts.values, labels=novelty_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Novelty Assessment Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_summary.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {output_prefix}_summary.png")
        plt.close()


def main():
    """
    Main execution function.
    
    Expected inputs:
    1. embeddings.npy - numpy array of embeddings (n_samples, n_features)
    2. labels.npy - numpy array of HDBScan cluster labels
    3. seq_ids.txt - text file with sequence IDs (one per line)
    4. taxonomy_database.tsv - taxonomic database file
    """
    
    # Load data
    print("Loading data...")
    embeddings = np.load('embeddings.npy')
    labels = np.load('labels.npy')
    
    with open('seq_ids.txt', 'r') as f:
        seq_ids = [line.strip() for line in f]
    
    # Load taxonomic database
    taxa_db = TaxonomicDatabase('taxonomy_database.tsv')
    
    # Create evaluator
    evaluator = HDBScanClusterEvaluator(embeddings, labels, seq_ids, taxa_db)
    
    # Run evaluation
    print("\nRunning evaluation...")
    taxonomy_df, novel_df = evaluator.generate_report('cluster_evaluation_report.txt')
    
    # Save detailed results
    taxonomy_df.to_csv('cluster_taxonomy_details.csv', index=False)
    novel_df.to_csv('novel_taxa_candidates.csv', index=False)
    
    # Generate plots
    evaluator.plot_results('cluster_evaluation')
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    print(f"Summary: {evaluator.n_clusters} clusters identified")
    print(f"Novel taxa candidates: {len(novel_df)}")
    print("\nOutput files:")
    print("  - cluster_evaluation_report.txt")
    print("  - cluster_taxonomy_details.csv")
    print("  - novel_taxa_candidates.csv")
    print("  - cluster_evaluation_summary.png")


if __name__ == '__main__':
    main()