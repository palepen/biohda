"""
Novelty Detection Module for HDBScan Clusters
Identifies novel sequences/clusters based on taxonomic composition and clustering characteristics
"""

import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from typing import Dict, List, Tuple, Set

# novelty_detector.py
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

DEFAULT_TAX_LEVELS = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

class NoveltyDetector:
    def __init__(self,
                 cluster_file: str,
                 taxonomy_file: str,
                 novelty_threshold: float = 0.5,
                 min_cluster_size: int = 5,
                 weights: Optional[Dict[str, float]] = None,
                 treat_noise_as_novel: bool = True):
        """
        Args:
            cluster_file: CSV output from HDBSCAN containing at least seqID, cluster_id
            taxonomy_file: TSV with seqID and a full_taxonomy column (or species etc.)
            novelty_threshold: composite score threshold [0,1] for marking cluster novel
            min_cluster_size: clusters smaller than this are considered 'small' (affects scoring)
            weights: dict of weights for score components (unassigned, low_genus_div, avg_prob, size)
            treat_noise_as_novel: include label -1 sequences as candidates
        """
        self.cluster_file = cluster_file
        self.taxonomy_file = taxonomy_file
        self.novelty_threshold = float(novelty_threshold)
        self.min_cluster_size = int(min_cluster_size)
        self.treat_noise_as_novel = bool(treat_noise_as_novel)

        # score weights: will be normalized to sum=1
        self.weights = weights or {
            'unassigned': 0.4,
            'low_genus_div': 0.3,
            'avg_prob': 0.2,
            'size': 0.1
        }

        # Load data
        self.clusters = pd.read_csv(self.cluster_file)
        self._validate_cluster_df(self.clusters)
        self.taxonomy = self._load_taxonomy(self.taxonomy_file)
        self._prepare_taxonomy_columns()

        # results
        self.cluster_analysis = {}
        self.novel_clusters = []
        self.novel_sequences = []

    def _validate_cluster_df(self, df: pd.DataFrame):
        required = {'seqID', 'cluster_id'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Cluster file missing columns: {missing}")
        # ensure cluster_id is numericable (may be -1)
        # no return; raises on missing

    def _load_taxonomy(self, taxonomy_file: str) -> pd.DataFrame:
        # Attempt flexible reading: if header present try reading with headers
        try:
            tax = pd.read_csv(taxonomy_file, sep='\t', dtype=str, keep_default_na=False)
        except Exception:
            # fallback to no header, assume columns: seqID, tax_id, species, full_taxonomy
            tax = pd.read_csv(taxonomy_file, sep='\t', header=None, dtype=str, keep_default_na=False)
            tax.columns = ['seqID', 'tax_id', 'species', 'full_taxonomy'][:tax.shape[1]]
        # Normalize empty strings to NaN for easier processing
        tax = tax.replace({'': pd.NA, 'NA': pd.NA, 'N/A': pd.NA, 'unknown': pd.NA})
        if 'full_taxonomy' not in tax.columns and 'species' in tax.columns:
            # prefer a concatenation if needed
            tax['full_taxonomy'] = tax['species'].astype(str).fillna('')
        # ensure seqID exists
        if 'seqID' not in tax.columns:
            raise ValueError("taxonomy file must contain a seqID column (name 'seqID')")
        return tax

    def _parse_full_taxonomy(self, tax_string: str) -> Dict[str, Optional[str]]:
        """
        Parse semi-colon delimited taxonomy string like:
        'k__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;...'
        Returns dict of rank->name (without prefixes)
        """
        if pd.isna(tax_string):
            return {lvl: None for lvl in DEFAULT_TAX_LEVELS}
        parts = [p.strip() for p in tax_string.split(';') if p.strip() != '']
        parsed = {}
        for i, part in enumerate(parts):
            # remove common prefixes like k__, p__, g__, or rank labels 'k:' etc
            name = part
            # handle "k__Bacteria", "p:Bacteroidetes", "d__Eukaryota"
            for sep in ['__', ':']:
                if sep in name:
                    name = name.split(sep, 1)[1]
                    break
            parsed[DEFAULT_TAX_LEVELS[i]] = name if name != '' else None
        # fill missing with None
        for lvl in DEFAULT_TAX_LEVELS:
            parsed.setdefault(lvl, None)
        return parsed

    def _prepare_taxonomy_columns(self):
        """Expand full_taxonomy into columns domain..species for faster vectorized operations."""
        if 'full_taxonomy' not in self.taxonomy.columns:
            # nothing to parse; create empty columns
            for lvl in DEFAULT_TAX_LEVELS:
                self.taxonomy[lvl] = pd.NA
            return

        # apply parsing vectorized-ish
        parsed = self.taxonomy['full_taxonomy'].fillna(pd.NA).map(self._parse_full_taxonomy)
        # parsed is a Series of dicts; transform to DataFrame
        parsed_df = pd.DataFrame(parsed.tolist(), index=self.taxonomy.index)
        # merge back
        for lvl in DEFAULT_TAX_LEVELS:
            self.taxonomy[lvl] = parsed_df[lvl].replace({np.nan: pd.NA})

        # normalize tax_id and species columns if exist
        if 'tax_id' in self.taxonomy.columns:
            self.taxonomy['tax_id'] = self.taxonomy['tax_id'].replace({'': pd.NA})
        if 'species' in self.taxonomy.columns:
            self.taxonomy['species'] = self.taxonomy['species'].replace({'': pd.NA})

    def analyze_cluster(self, cluster_id) -> Dict:
        """Analyze a cluster and return metrics + novelty score"""
        cluster_seqs = self.clusters[self.clusters['cluster_id'] == cluster_id]
        size = int(len(cluster_seqs))
        # merge taxonomy on seqID (left join to keep sequences without taxonomy)
        cluster_tax = cluster_seqs.merge(self.taxonomy, on='seqID', how='left')

        # cluster probability may be missing; handle gracefully
        if 'cluster_probability' in cluster_seqs.columns:
            avg_prob = float(cluster_seqs['cluster_probability'].dropna().mean()) if size > 0 else 0.0
        else:
            avg_prob = 0.0  # unknown; treat as neutral/low confidence

        # robust unassigned detection: tax_id is NA or species is NA or genus is NA
        unassigned_mask = (
            cluster_tax.get('tax_id').isna() |
            cluster_tax.get('species').isna() |
            cluster_tax.get('genus').isna()
        )
        unassigned_count = int(unassigned_mask.sum())
        unassigned_ratio = (unassigned_count / size) if size > 0 else 0.0

        # genus diversity
        genus_series = cluster_tax['genus'].dropna().astype(str)
        genus_unique_count = int(genus_series.nunique()) if not genus_series.empty else 0
        genus_counts = Counter(genus_series.tolist())

        taxonomic_diversity = {
            'genus': {'unique_count': genus_unique_count, 'taxa': genus_counts},
            # optionally compute other ranks
        }

        novelty_score = self._calculate_novelty_score(
            unassigned_ratio=unassigned_ratio,
            genus_unique_count=genus_unique_count,
            avg_prob=avg_prob,
            size=size
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
                                 genus_unique_count: int,
                                 avg_prob: float,
                                 size: int) -> float:
        """
        Normalize each component to [0,1] then apply weights (weights normalized).
        Components (higher -> more novel):
          - unassigned_ratio (0..1)
          - low_genus_divergence (we define as 1 - normalized genus count)
          - avg_prob (if high -> more likely a real cluster -> positive)
          - size (we prefer size >= min_cluster_size; small clusters get a boost if we want to flag small novel clusters)
        """
        # normalize weights
        w = dict(self.weights)
        total_w = sum(w.values())
        if total_w <= 0:
            raise ValueError("Sum of weights must be > 0")
        for k in w:
            w[k] /= total_w

        # comp1: unassigned_ratio already [0,1]
        c_unassigned = float(np.clip(unassigned_ratio, 0.0, 1.0))

        # comp2: low genus diversity -> if genus_unique_count is small => more novel
        # normalize genus count by a heuristic max (e.g., 10)
        max_genus_for_norm = max(10, genus_unique_count)  # avoid div by zero; but we want monotonic
        # Instead better: scale with min(genus_unique_count, 10)/10 then invert
        normalized_genus = min(genus_unique_count, 10) / 10.0
        c_low_genus_div = 1.0 - normalized_genus  # 1 means genus_unique_count=0 (strong novelty signal)

        # comp3: avg_prob -> higher is better cluster quality (0..1)
        c_avg_prob = float(np.clip(avg_prob, 0.0, 1.0))

        # comp4: size -> if size >= min_cluster_size -> 1, else small clusters get 1 (if you want to flag small)
        # We'll give small clusters a boost if size < min_cluster_size (they might represent rare novel taxa).
        if size >= self.min_cluster_size:
            c_size = 1.0
        else:
            # scale smallness: 1.0 if size==1 else size/min_cluster_size
            c_size = float(size) / float(self.min_cluster_size) if self.min_cluster_size > 0 else 1.0
            # invert so smaller sizes yield higher novelty if you prefer small clusters flagged:
            c_size = 1.0 - np.clip(c_size, 0.0, 1.0)

        # Combine (note: if you want small clusters to increase score, use c_size as defined; otherwise invert)
        # Here we design so higher c_size means MORE novel (smaller cluster -> more novel)
        score = (
            w['unassigned'] * c_unassigned +
            w['low_genus_div'] * c_low_genus_div +
            w['avg_prob'] * c_avg_prob +
            w['size'] * c_size
        )
        # clamp
        return float(np.clip(score, 0.0, 1.0))

    def detect_novel_sequences(self) -> Tuple[List, List[str]]:
        """
        Process all clusters and return (novel_cluster_ids, novel_sequence_ids).
        By default excludes cluster -1 unless treat_noise_as_novel=True.
        """
        novel_clusters = []
        novel_seq_ids = set()

        unique_clusters = self.clusters['cluster_id'].unique()
        # optionally exclude -1 from cluster loop (we will handle later)
        to_process = unique_clusters if self.treat_noise_as_novel else unique_clusters[unique_clusters != -1]

        for cid in to_process:
            analysis = self.analyze_cluster(cid)
            self.cluster_analysis[cid] = analysis
            if analysis['is_novel']:
                novel_clusters.append(cid)
                # add seqIDs
                seqs = self.clusters[self.clusters['cluster_id'] == cid]['seqID'].tolist()
                novel_seq_ids.update(seqs)

        # handle noise cluster: if treat_noise_as_novel, include sequences that lack taxonomy
        if -1 in unique_clusters and self.treat_noise_as_novel:
            noise = self.clusters[self.clusters['cluster_id'] == -1]
            noise_tax = noise.merge(self.taxonomy, on='seqID', how='left')
            unassigned_mask = (
                noise_tax.get('tax_id').isna() |
                noise_tax.get('species').isna() |
                noise_tax.get('genus').isna()
            )
            noise_unassigned_seqs = noise_tax[unassigned_mask]['seqID'].tolist()
            novel_seq_ids.update(noise_unassigned_seqs)

        self.novel_clusters = novel_clusters
        self.novel_sequences = sorted(list(novel_seq_ids))
        return self.novel_clusters, self.novel_sequences

    def generate_report(self, output_file: str = 'novelty_report.txt'):
        with open(output_file, 'w') as f:
            f.write("NOVELTY DETECTION REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Clusters analyzed: {len(self.cluster_analysis)}\n")
            f.write(f"Novel clusters: {len(self.novel_clusters)}\n")
            f.write(f"Novel sequences: {len(self.novel_sequences)}\n\n")
            for cid in sorted(self.novel_clusters):
                a = self.cluster_analysis[cid]
                f.write(f"Cluster {cid}: size={a['size']}, novelty_score={a['novelty_score']:.3f}\n")
                f.write(f"  avg_prob={a['avg_probability']:.3f}, unassigned={a['unassigned_count']} ({a['unassigned_ratio']:.2f})\n")
                g = a['taxonomic_diversity']['genus']
                f.write(f"  genus_unique={g['unique_count']}\n")
                top = g['taxa'].most_common(5)
                for name, c in top:
                    f.write(f"    {name}: {c}\n")
                f.write("\n")
        print(f"Report written to {output_file}")

    def save_novel_sequences(self, output_file: str = 'novel_sequences.txt'):
        df = pd.DataFrame({'seqID': self.novel_sequences})
        df.to_csv(output_file, index=False)
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
    detector.generate_report('dataset/novelty/novelty_report.txt')
    detector.save_novel_sequences('dataset/novelty/novel_sequences.txt')
    
    print("\nNovelty detection complete!")


if __name__ == '__main__':
    main()