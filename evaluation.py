"""
Evaluation Module for HDBScan Clustering and Novelty Detection
Evaluates clustering quality and validates detected novel sequences
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
from scipy.stats import entropy

class ClusteringEvaluator:
    def __init__(self, 
                 cluster_file: str,
                 taxonomy_file: str,
                 novel_sequences_file: str = None):
        """
        Initialize the evaluator.
        
        Args:
            cluster_file: Path to HDBScan output CSV
            taxonomy_file: Path to taxonomy assignment file
            novel_sequences_file: Optional path to novel sequences detected
        """
        self.clusters = pd.read_csv(cluster_file)
        self.taxonomy = self._load_taxonomy(taxonomy_file)
        
        # Merge data
        self.data = self.clusters.merge(self.taxonomy, on='seqID', how='left')
        
        # Load novel sequences if provided
        self.novel_sequences = []
        if novel_sequences_file:
            with open(novel_sequences_file, 'r') as f:
                self.novel_sequences = [line.strip() for line in f if line.strip() and line.strip() != 'seqID']
        
        self.evaluation_results = {}
    
    def _load_taxonomy(self, taxonomy_file: str) -> pd.DataFrame:
        """Load and parse taxonomy TSV file."""
        # Use pandas to read TSV file with proper handling
        taxonomy_df = pd.read_csv(
            taxonomy_file, 
            sep='\t', 
            header=None,
            names=['seqID', 'tax_id', 'species', 'full_taxonomy'],
            dtype=str,
            keep_default_na=False  # Don't convert strings to NaN
        )
        
        # Fill empty taxonomy strings
        taxonomy_df['full_taxonomy'] = taxonomy_df['full_taxonomy'].replace('', pd.NA)
        
        return taxonomy_df
    
    def _parse_taxonomy(self, tax_string: str) -> Dict[str, str]:
        """Parse taxonomy string into taxonomic levels."""
        levels = ['domain', 'kingdom', 'phylum', 'class', 'order', 
                  'family', 'genus', 'species']
        parts = tax_string.split(';')
        taxonomy = {}
        for i, part in enumerate(parts):
            if i < len(levels):
                taxonomy[levels[i]] = part.strip()
        return taxonomy
    
    def evaluate_cluster_purity(self, level: str = 'genus') -> Dict:
        """
        Evaluate taxonomic purity of clusters at a given taxonomic level.
        
        Args:
            level: Taxonomic level to evaluate (genus, family, order, etc.)
        
        Returns:
            Dictionary with purity metrics
        """
        print(f"\nEvaluating cluster purity at {level} level...")
        
        cluster_purities = {}
        cluster_entropies = {}
        
        # Get non-noise clusters
        valid_clusters = self.data[self.data['cluster_id'] != -1]
        unique_clusters = valid_clusters['cluster_id'].unique()
        
        for cluster_id in unique_clusters:
            cluster_data = valid_clusters[valid_clusters['cluster_id'] == cluster_id]
            
            # Extract taxonomic labels at specified level
            taxa = []
            for tax_str in cluster_data['full_taxonomy'].dropna():
                parsed = self._parse_taxonomy(tax_str)
                if level in parsed and parsed[level]:
                    taxa.append(parsed[level])
                else:
                    taxa.append('Unassigned')
            
            if not taxa:
                continue
            
            # Calculate purity (fraction of most common taxon)
            taxon_counts = Counter(taxa)
            most_common_count = taxon_counts.most_common(1)[0][1]
            purity = most_common_count / len(taxa)
            cluster_purities[cluster_id] = {
                'purity': purity,
                'size': len(taxa),
                'taxa_distribution': dict(taxon_counts),
                'dominant_taxon': taxon_counts.most_common(1)[0][0]
            }
            
            # Calculate entropy (lower is better - less diversity)
            probs = np.array(list(taxon_counts.values())) / len(taxa)
            cluster_entropies[cluster_id] = entropy(probs)
        
        # Overall statistics
        purities = [v['purity'] for v in cluster_purities.values()]
        entropies = list(cluster_entropies.values())
        
        return {
            'level': level,
            'cluster_purities': cluster_purities,
            'cluster_entropies': cluster_entropies,
            'mean_purity': np.mean(purities) if purities else 0,
            'median_purity': np.median(purities) if purities else 0,
            'mean_entropy': np.mean(entropies) if entropies else 0,
            'high_purity_clusters': sum(1 for p in purities if p >= 0.8),
            'total_clusters': len(purities)
        }
    
    def evaluate_cluster_coherence(self) -> Dict:
        """
        Evaluate cluster coherence using probability scores.
        High coherence means sequences have high probability of belonging to cluster.
        """
        print("\nEvaluating cluster coherence...")
        
        valid_clusters = self.data[self.data['cluster_id'] != -1]
        unique_clusters = valid_clusters['cluster_id'].unique()
        
        cluster_coherence = {}
        for cluster_id in unique_clusters:
            cluster_data = valid_clusters[valid_clusters['cluster_id'] == cluster_id]
            probs = cluster_data['cluster_probability'].values
            
            cluster_coherence[cluster_id] = {
                'mean_probability': np.mean(probs),
                'median_probability': np.median(probs),
                'std_probability': np.std(probs),
                'min_probability': np.min(probs),
                'size': len(probs),
                'high_confidence_count': sum(probs >= 0.8),
                'low_confidence_count': sum(probs < 0.5)
            }
        
        all_means = [v['mean_probability'] for v in cluster_coherence.values()]
        
        return {
            'cluster_coherence': cluster_coherence,
            'overall_mean_probability': np.mean(all_means) if all_means else 0,
            'high_coherence_clusters': sum(1 for m in all_means if m >= 0.8),
            'total_clusters': len(all_means)
        }
    
    def evaluate_noise_cluster(self) -> Dict:
        """Evaluate sequences assigned to noise cluster (-1)."""
        print("\nEvaluating noise cluster...")
        
        noise_data = self.data[self.data['cluster_id'] == -1]
        
        if len(noise_data) == 0:
            return {'noise_count': 0}
        
        # Check how many noise sequences are actually unassigned taxonomically
        unassigned = noise_data['tax_id'].eq('0') | noise_data['species'].eq('N/A')
        
        # Get taxonomic diversity in noise
        taxa_counts = defaultdict(int)
        for tax_str in noise_data['full_taxonomy'].dropna():
            parsed = self._parse_taxonomy(tax_str)
            if 'genus' in parsed:
                taxa_counts[parsed['genus']] += 1
        
        return {
            'noise_count': len(noise_data),
            'noise_ratio': len(noise_data) / len(self.data),
            'unassigned_in_noise': unassigned.sum(),
            'assigned_in_noise': len(noise_data) - unassigned.sum(),
            'unique_genera_in_noise': len(taxa_counts),
            'mean_probability': noise_data['cluster_probability'].mean()
        }
    
    def evaluate_novel_sequences(self) -> Dict:
        """
        Evaluate detected novel sequences to validate if they are truly novel.
        """
        if not self.novel_sequences:
            print("\nNo novel sequences to evaluate.")
            return {'total_novel_detected': 0}
        
        print(f"\nEvaluating {len(self.novel_sequences)} novel sequences...")
        
        novel_data = self.data[self.data['seqID'].isin(self.novel_sequences)]
        
        # True positives: Novel sequences that are unassigned
        truly_unassigned = novel_data['tax_id'].eq('0') | novel_data['species'].eq('N/A')
        true_positive_count = truly_unassigned.sum()
        
        # False positives: Novel sequences that are actually assigned
        false_positive_count = len(novel_data) - true_positive_count
        
        # Get details of false positives
        false_positives = novel_data[~truly_unassigned]
        false_positive_taxa = []
        for _, row in false_positives.iterrows():
            if pd.notna(row['full_taxonomy']):
                parsed = self._parse_taxonomy(row['full_taxonomy'])
                false_positive_taxa.append({
                    'seqID': row['seqID'],
                    'species': row['species'],
                    'genus': parsed.get('genus', 'Unknown'),
                    'cluster_id': row['cluster_id']
                })
        
        # Check if novel sequences from same clusters
        cluster_distribution = Counter(novel_data['cluster_id'].values)
        
        # Calculate precision
        precision = true_positive_count / len(novel_data) if len(novel_data) > 0 else 0
        
        return {
            'total_novel_detected': len(self.novel_sequences),
            'true_positives': true_positive_count,
            'false_positives': false_positive_count,
            'precision': precision,
            'false_positive_details': false_positive_taxa,
            'cluster_distribution': dict(cluster_distribution),
            'in_noise_cluster': cluster_distribution.get(-1, 0)
        }
    
    def calculate_cluster_homogeneity(self, level: str = 'genus') -> float:
        """
        Calculate overall homogeneity score across all clusters.
        Based on how well clusters separate different taxonomic groups.
        """
        valid_clusters = self.data[self.data['cluster_id'] != -1]
        
        # Get taxonomic assignments
        taxon_to_clusters = defaultdict(set)
        cluster_to_taxa = defaultdict(set)
        
        for _, row in valid_clusters.iterrows():
            if pd.notna(row['full_taxonomy']):
                parsed = self._parse_taxonomy(row['full_taxonomy'])
                if level in parsed and parsed[level]:
                    taxon = parsed[level]
                    cluster_id = row['cluster_id']
                    taxon_to_clusters[taxon].add(cluster_id)
                    cluster_to_taxa[cluster_id].add(taxon)
        
        # Calculate homogeneity: average number of clusters per taxon (lower is better)
        # and average number of taxa per cluster (lower is better)
        avg_clusters_per_taxon = np.mean([len(clusters) for clusters in taxon_to_clusters.values()])
        avg_taxa_per_cluster = np.mean([len(taxa) for taxa in cluster_to_taxa.values()])
        
        # Normalize to 0-1 score (1 is perfect homogeneity)
        homogeneity_score = 1 / (1 + avg_taxa_per_cluster)
        
        return {
            'homogeneity_score': homogeneity_score,
            'avg_clusters_per_taxon': avg_clusters_per_taxon,
            'avg_taxa_per_cluster': avg_taxa_per_cluster,
            'unique_taxa': len(taxon_to_clusters),
            'unique_clusters': len(cluster_to_taxa)
        }
    
    def generate_evaluation_report(self, output_file: str = 'evaluation_report.txt'):
        """Generate comprehensive evaluation report."""
        print("\n" + "=" * 80)
        print("Running comprehensive evaluation...")
        print("=" * 80)
        
        # Run all evaluations
        purity_genus = self.evaluate_cluster_purity('genus')
        purity_family = self.evaluate_cluster_purity('family')
        coherence = self.evaluate_cluster_coherence()
        noise_eval = self.evaluate_noise_cluster()
        novel_eval = self.evaluate_novel_sequences()
        homogeneity = self.calculate_cluster_homogeneity('genus')
        
        # Write report
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLUSTERING AND NOVELTY DETECTION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total sequences: {len(self.data)}\n")
            f.write(f"Total clusters (excluding noise): {len(self.data[self.data['cluster_id'] != -1]['cluster_id'].unique())}\n")
            f.write(f"Sequences in noise cluster: {noise_eval['noise_count']} ({noise_eval.get('noise_ratio', 0):.1%})\n\n")
            
            # Clustering quality
            f.write("=" * 80 + "\n")
            f.write("1. CLUSTERING QUALITY EVALUATION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1.1 Taxonomic Purity\n")
            f.write("-" * 80 + "\n")
            f.write(f"Genus Level:\n")
            f.write(f"  Mean Purity: {purity_genus['mean_purity']:.3f}\n")
            f.write(f"  Median Purity: {purity_genus['median_purity']:.3f}\n")
            f.write(f"  Mean Entropy: {purity_genus['mean_entropy']:.3f}\n")
            f.write(f"  High Purity Clusters (≥0.8): {purity_genus['high_purity_clusters']}/{purity_genus['total_clusters']}\n\n")
            
            f.write(f"Family Level:\n")
            f.write(f"  Mean Purity: {purity_family['mean_purity']:.3f}\n")
            f.write(f"  Median Purity: {purity_family['median_purity']:.3f}\n")
            f.write(f"  High Purity Clusters (≥0.8): {purity_family['high_purity_clusters']}/{purity_family['total_clusters']}\n\n")
            
            # Show top impure clusters
            f.write("Clusters with Low Purity (<0.5):\n")
            low_purity = {k: v for k, v in purity_genus['cluster_purities'].items() 
                         if v['purity'] < 0.5}
            if low_purity:
                for cluster_id in sorted(low_purity.keys())[:10]:
                    info = low_purity[cluster_id]
                    f.write(f"  Cluster {cluster_id}: Purity={info['purity']:.3f}, Size={info['size']}\n")
                    f.write(f"    Taxa: {info['taxa_distribution']}\n")
            else:
                f.write("  None found\n")
            f.write("\n")
            
            f.write("1.2 Cluster Coherence (Probability Scores)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Overall Mean Probability: {coherence['overall_mean_probability']:.3f}\n")
            f.write(f"High Coherence Clusters (≥0.8): {coherence['high_coherence_clusters']}/{coherence['total_clusters']}\n\n")
            
            # Show problematic clusters
            f.write("Clusters with Low Coherence (<0.6):\n")
            low_coherence = {k: v for k, v in coherence['cluster_coherence'].items() 
                           if v['mean_probability'] < 0.6}
            if low_coherence:
                for cluster_id in sorted(low_coherence.keys())[:10]:
                    info = low_coherence[cluster_id]
                    f.write(f"  Cluster {cluster_id}: Mean Prob={info['mean_probability']:.3f}, "
                           f"Size={info['size']}, Low Conf={info['low_confidence_count']}\n")
            else:
                f.write("  None found\n")
            f.write("\n")
            
            f.write("1.3 Homogeneity Score\n")
            f.write("-" * 80 + "\n")
            f.write(f"Homogeneity Score: {homogeneity['homogeneity_score']:.3f}\n")
            f.write(f"Avg Taxa per Cluster: {homogeneity['avg_taxa_per_cluster']:.2f}\n")
            f.write(f"Avg Clusters per Taxon: {homogeneity['avg_clusters_per_taxon']:.2f}\n\n")
            
            f.write("1.4 Noise Cluster Analysis\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sequences in noise: {noise_eval['noise_count']}\n")
            f.write(f"Unassigned in noise: {noise_eval.get('unassigned_in_noise', 0)}\n")
            f.write(f"Assigned in noise: {noise_eval.get('assigned_in_noise', 0)}\n")
            f.write(f"Unique genera in noise: {noise_eval.get('unique_genera_in_noise', 0)}\n\n")
            
            # Novelty detection evaluation
            f.write("=" * 80 + "\n")
            f.write("2. NOVELTY DETECTION EVALUATION\n")
            f.write("=" * 80 + "\n\n")
            
            if novel_eval['total_novel_detected'] > 0:
                f.write(f"Total Novel Sequences Detected: {novel_eval['total_novel_detected']}\n")
                f.write(f"True Positives (actually unassigned): {novel_eval['true_positives']}\n")
                f.write(f"False Positives (have assignments): {novel_eval['false_positives']}\n")
                f.write(f"Precision: {novel_eval['precision']:.3f}\n\n")
                
                f.write(f"Novel sequences in noise cluster: {novel_eval['in_noise_cluster']}\n")
                f.write(f"Novel sequences in valid clusters: {novel_eval['total_novel_detected'] - novel_eval['in_noise_cluster']}\n\n")
                
                if novel_eval['false_positive_details']:
                    f.write("False Positive Examples:\n")
                    for fp in novel_eval['false_positive_details'][:20]:
                        f.write(f"  {fp['seqID']}: {fp['species']} (genus: {fp['genus']}, cluster: {fp['cluster_id']})\n")
                    f.write("\n")
            else:
                f.write("No novel sequences were detected or provided for evaluation.\n\n")
            
            # Overall assessment
            f.write("=" * 80 + "\n")
            f.write("3. OVERALL ASSESSMENT\n")
            f.write("=" * 80 + "\n\n")
            
            # Clustering quality score
            clustering_score = (purity_genus['mean_purity'] + 
                              coherence['overall_mean_probability'] + 
                              homogeneity['homogeneity_score']) / 3
            
            f.write(f"Overall Clustering Quality Score: {clustering_score:.3f}/1.0\n\n")
            
            f.write("Interpretation:\n")
            if clustering_score >= 0.8:
                f.write("  EXCELLENT: Clusters are highly pure and coherent\n")
            elif clustering_score >= 0.6:
                f.write("  GOOD: Clustering is generally reliable\n")
            elif clustering_score >= 0.4:
                f.write("  FAIR: Some clusters may need refinement\n")
            else:
                f.write("  POOR: Consider adjusting clustering parameters\n")
            
            f.write("\nNovelty Detection Performance:\n")
            if novel_eval['total_novel_detected'] > 0:
                if novel_eval['precision'] >= 0.8:
                    f.write("  EXCELLENT: Most detected novel sequences are truly unassigned\n")
                elif novel_eval['precision'] >= 0.6:
                    f.write("  GOOD: Majority of novel detections are accurate\n")
                elif novel_eval['precision'] >= 0.4:
                    f.write("  FAIR: Moderate rate of false positives\n")
                else:
                    f.write("  POOR: High false positive rate, consider adjusting threshold\n")
            
        print(f"\nEvaluation report saved to {output_file}")
        
        # Return summary for programmatic use
        return {
            'clustering_quality_score': clustering_score,
            'purity': purity_genus,
            'coherence': coherence,
            'homogeneity': homogeneity,
            'noise': noise_eval,
            'novelty': novel_eval
        }


def main():
    """Example usage."""
    evaluator = ClusteringEvaluator(
        cluster_file='./dataset/clusters/clusters.csv',
        taxonomy_file='./dataset/ssu_accession_full_lineage.tsv'
    )
    
    results = evaluator.generate_evaluation_report('dataset/evaluation/evaluation_report.txt')
    
    print("\n" + "=" * 80)
    print(f"Overall Clustering Quality: {results['clustering_quality_score']:.3f}/1.0")
    if results['novelty']['total_novel_detected'] > 0:
        print(f"Novelty Detection Precision: {results['novelty']['precision']:.3f}")
    print("=" * 80)
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()