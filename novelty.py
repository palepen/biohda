"""
Novelty Detection Module for HDBScan Clusters
Identifies novel sequences/clusters based on taxonomic composition and clustering characteristics
"""

import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from typing import Dict, List, Tuple, Set

class NoveltyDetector:
    def __init__(self, 
                 cluster_file: str, 
                 taxonomy_file: str,
                 novelty_threshold: float = 0.5,
                 min_cluster_size: int = 1):
        """
        Initialize the novelty detector.
        
        Args:
            cluster_file: Path to HDBScan output CSV
            taxonomy_file: Path to taxonomy assignment file (tab-separated)
            novelty_threshold: Threshold for considering a cluster novel (0-1)
            min_cluster_size: Minimum cluster size to consider
        """
        self.novelty_threshold = novelty_threshold
        self.min_cluster_size = min_cluster_size
        
        # Load data
        self.clusters = pd.read_csv(cluster_file)
        self.taxonomy = self._load_taxonomy(taxonomy_file)
        
        # Analysis results
        self.cluster_analysis = {}
        self.novel_clusters = []
        self.novel_sequences = []
        
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
    
    def analyze_cluster(self, cluster_id: int) -> Dict:
        """Analyze a single cluster for novelty indicators."""
        cluster_seqs = self.clusters[self.clusters['cluster_id'] == cluster_id]
        
        # Merge with taxonomy
        cluster_tax = cluster_seqs.merge(self.taxonomy, on='seqID', how='left')
        
        # Calculate metrics
        size = len(cluster_seqs)
        avg_prob = cluster_seqs['cluster_probability'].mean()
        
        # Check for unassigned sequences
        unassigned = cluster_tax['tax_id'].eq('0') | cluster_tax['species'].eq('N/A')
        unassigned_count = unassigned.sum()
        unassigned_ratio = unassigned_count / size if size > 0 else 0
        
        # Taxonomic diversity at different levels
        taxonomic_diversity = {}
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            taxa = []
            for tax_str in cluster_tax['full_taxonomy'].dropna():
                parsed = self._parse_taxonomy(tax_str)
                if level in parsed:
                    taxa.append(parsed[level])
            
            if taxa:
                unique_taxa = len(set(taxa))
                taxonomic_diversity[level] = {
                    'unique_count': unique_taxa,
                    'taxa': Counter(taxa)
                }
            else:
                taxonomic_diversity[level] = {'unique_count': 0, 'taxa': Counter()}
        
        # Calculate novelty score
        novelty_score = self._calculate_novelty_score(
            unassigned_ratio, taxonomic_diversity, avg_prob, size
        )
        
        return {
            'cluster_id': cluster_id,
            'size': size,
            'avg_probability': avg_prob,
            'unassigned_count': unassigned_count,
            'unassigned_ratio': unassigned_ratio,
            'taxonomic_diversity': taxonomic_diversity,
            'novelty_score': novelty_score,
            'is_novel': novelty_score >= self.novelty_threshold
        }
    
    def _calculate_novelty_score(self, 
                                   unassigned_ratio: float,
                                   tax_diversity: Dict,
                                   avg_prob: float,
                                   size: int) -> float:
        """
        Calculate novelty score based on multiple factors.
        
        Score components:
        - Unassigned sequences ratio (40%)
        - Low taxonomic diversity at genus level (30%)
        - High cluster probability (20%)
        - Sufficient cluster size (10%)
        """
        score = 0.0
        
        # Unassigned sequences strongly suggest novelty
        score += unassigned_ratio * 0.4
        
        # Low genus diversity with some sequences suggests novel genus/species
        genus_diversity = tax_diversity.get('genus', {}).get('unique_count', 0)
        if genus_diversity <= 2 and unassigned_ratio > 0:
            score += 0.3
        elif genus_diversity == 0:
            score += 0.3
        
        # High cluster probability suggests strong grouping
        if avg_prob > 0.8:
            score += 0.2
        elif avg_prob > 0.6:
            score += 0.1
        
        # Sufficient size
        if size >= self.min_cluster_size:
            score += 0.1
        
        return min(score, 1.0)
    
    def detect_novel_sequences(self) -> Tuple[List[int], List[str]]:
        """
        Detect novel clusters and sequences.
        
        Returns:
            Tuple of (novel_cluster_ids, novel_sequence_ids)
        """
        novel_clusters = []
        novel_sequences = []
        
        # Get unique clusters (excluding noise cluster -1)
        unique_clusters = self.clusters[self.clusters['cluster_id'] != -1]['cluster_id'].unique()
        
        print(f"Analyzing {len(unique_clusters)} clusters...")
        
        for cluster_id in unique_clusters:
            analysis = self.analyze_cluster(cluster_id)
            self.cluster_analysis[cluster_id] = analysis
            
            if analysis['is_novel']:
                novel_clusters.append(cluster_id)
                # Get sequences from novel cluster
                cluster_seqs = self.clusters[self.clusters['cluster_id'] == cluster_id]['seqID'].tolist()
                novel_sequences.extend(cluster_seqs)
        
        # Also check noise cluster (-1) for unassigned sequences
        noise_cluster = self.clusters[self.clusters['cluster_id'] == -1]
        if len(noise_cluster) > 0:
            noise_tax = noise_cluster.merge(self.taxonomy, on='seqID', how='left')
            unassigned_noise = noise_tax[
                noise_tax['tax_id'].eq('0') | noise_tax['species'].eq('N/A')
            ]['seqID'].tolist()
            novel_sequences.extend(unassigned_noise)
        
        self.novel_clusters = novel_clusters
        self.novel_sequences = list(set(novel_sequences))  # Remove duplicates
        
        return novel_clusters, self.novel_sequences
    
    def generate_report(self, output_file: str = 'novelty_report.txt'):
        """Generate a detailed novelty detection report."""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NOVELTY DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total clusters analyzed: {len(self.cluster_analysis)}\n")
            f.write(f"Novel clusters detected: {len(self.novel_clusters)}\n")
            f.write(f"Novel sequences detected: {len(self.novel_sequences)}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("NOVEL CLUSTERS DETAILS\n")
            f.write("=" * 80 + "\n\n")
            
            for cluster_id in sorted(self.novel_clusters):
                analysis = self.cluster_analysis[cluster_id]
                f.write(f"\nCluster ID: {cluster_id}\n")
                f.write(f"  Size: {analysis['size']}\n")
                f.write(f"  Novelty Score: {analysis['novelty_score']:.3f}\n")
                f.write(f"  Avg Probability: {analysis['avg_probability']:.3f}\n")
                f.write(f"  Unassigned Sequences: {analysis['unassigned_count']} ({analysis['unassigned_ratio']:.1%})\n")
                
                # Show taxonomic composition
                f.write(f"  Taxonomic Composition:\n")
                for level in ['genus', 'family', 'order']:
                    if level in analysis['taxonomic_diversity']:
                        div = analysis['taxonomic_diversity'][level]
                        f.write(f"    {level.capitalize()}: {div['unique_count']} unique\n")
                        if div['taxa']:
                            top_taxa = div['taxa'].most_common(3)
                            for taxon, count in top_taxa:
                                f.write(f"      - {taxon}: {count}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            if self.cluster_analysis:
                scores = [a['novelty_score'] for a in self.cluster_analysis.values()]
                f.write(f"Novelty Score Statistics:\n")
                f.write(f"  Mean: {np.mean(scores):.3f}\n")
                f.write(f"  Median: {np.median(scores):.3f}\n")
                f.write(f"  Max: {np.max(scores):.3f}\n")
                f.write(f"  Min: {np.min(scores):.3f}\n")
        
        print(f"Report saved to {output_file}")
    
    def save_novel_sequences(self, output_file: str = 'novel_sequences.txt'):
        """Save list of novel sequence IDs."""
        with open(output_file, 'w') as f:
            f.write("seqID\n")
            for seq_id in sorted(self.novel_sequences):
                f.write(f"{seq_id}\n")
        print(f"Novel sequences saved to {output_file}")


def main():
    """Example usage."""
    # Initialize detector
    detector = NoveltyDetector(
        cluster_file='./dataset/clusters/clusters.csv',
        taxonomy_file='./dataset/ssu_accession_full_lineage.tsv',
        novelty_threshold=0.5,
        min_cluster_size=1
    )
    
    # Detect novel clusters and sequences
    novel_clusters, novel_sequences = detector.detect_novel_sequences()
    
    print(f"\nDetected {len(novel_clusters)} novel clusters")
    print(f"Detected {len(novel_sequences)} novel sequences")
    
    # Generate reports
    detector.generate_report('novelty_report.txt')
    detector.save_novel_sequences('novel_sequences.txt')
    
    print("\nNovelty detection complete!")


if __name__ == '__main__':
    main()