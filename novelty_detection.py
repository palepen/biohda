"""
Step 8: Enhanced Novelty Detection
===================================
Improved isolation metrics and candidate scoring for novel taxa discovery.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NoveltyDetector:
    """Enhanced novelty detection with improved scoring"""
    def __init__(self, clusters_dir="dataset/clusters", embeddings_dir="dataset/embeddings",
                 metadata_file="dataset/metadata.csv", output_dir="dataset/novelty",
                 purity_threshold=0.3, min_novel_size=5, isolation_percentile=85):
        self.clusters_dir = Path(clusters_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.purity_threshold = purity_threshold
        self.min_novel_size = min_novel_size
        self.isolation_percentile = isolation_percentile
        
        logger.info(f"Novelty Detection Config:")
        logger.info(f"  Purity threshold: {purity_threshold}")
        logger.info(f"  Min size: {min_novel_size}")
        logger.info(f"  Isolation percentile: {isolation_percentile}")

    def load_data(self):
        logger.info("\nLoading data...")
        
        clusters = pd.read_csv(self.clusters_dir / "clusters.csv")
        analysis = pd.read_csv(self.clusters_dir / "cluster_analysis.csv")
        embeddings = np.load(self.embeddings_dir / "embeddings.npy")
        
        with open(self.embeddings_dir / "embedding_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"  Sequences: {len(clusters):,}")
        logger.info(f"  Embeddings: {embeddings.shape}")
        return clusters, analysis, embeddings, metadata

    def identify_noise(self, clusters_df):
        logger.info("\nIdentifying noise sequences...")
        noise = clusters_df[clusters_df['cluster_id'] == -1].copy()
        logger.info(f"  Noise: {len(noise):,} ({len(noise)/len(clusters_df):.2%})")
        return noise

    def identify_low_purity(self, analysis_df):
        logger.info("\nIdentifying low-purity clusters...")
        valid = analysis_df[analysis_df['cluster_id'] != -1].copy()
        low_purity = valid[(valid['purity'] < self.purity_threshold) & 
                           (valid['size'] >= self.min_novel_size)]
        
        logger.info(f"  Low-purity: {len(low_purity)} clusters")
        if len(low_purity) > 0:
            logger.info(f"    Size: {low_purity['size'].min():,} - {low_purity['size'].max():,}")
        return low_purity

    def identify_unknown_taxonomy(self, analysis_df):
        logger.info("\nIdentifying unknown taxonomy clusters...")
        valid = analysis_df[analysis_df['cluster_id'] != -1].copy()
        unknown = valid[(~valid['has_taxonomy']) & (valid['size'] >= self.min_novel_size)]
        
        logger.info(f"  Unknown: {len(unknown)} clusters")
        if len(unknown) > 0:
            logger.info(f"    Sequences: {unknown['size'].sum():,}")
        return unknown

    def compute_isolation_advanced(self, embeddings, clusters_df, k=30):
        logger.info(f"\nComputing advanced isolation scores (k={k})...")
        
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1)
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Mean distance to k neighbors
        mean_dist = distances[:, 1:].mean(axis=1)
        
        # Std of distances (consistency)
        std_dist = distances[:, 1:].std(axis=1)
        
        # Combined isolation score
        isolation_scores = mean_dist * (1 + std_dist)
        
        logger.info(f"  Score range: {isolation_scores.min():.4f} - {isolation_scores.max():.4f}")
        logger.info(f"  Mean: {isolation_scores.mean():.4f}, Median: {np.median(isolation_scores):.4f}")
        
        return isolation_scores

    def identify_isolated(self, isolation_scores, clusters_df):
        logger.info("\nIdentifying isolated sequences...")
        
        threshold = np.percentile(isolation_scores, self.isolation_percentile)
        isolated_mask = isolation_scores > threshold
        
        isolated = clusters_df[isolated_mask].copy()
        isolated['isolation_score'] = isolation_scores[isolated_mask]
        
        logger.info(f"  Threshold: {threshold:.4f}")
        logger.info(f"  Isolated: {len(isolated):,} ({len(isolated)/len(clusters_df):.2%})")
        return isolated

    def extract_representatives(self, cluster_id, clusters_df, embeddings, n_rep=5):
        mask = clusters_df['cluster_id'] == cluster_id
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return []
        
        cluster_emb = embeddings[indices]
        centroid = cluster_emb.mean(axis=0)
        distances = np.linalg.norm(cluster_emb - centroid, axis=1)
        
        n = min(n_rep, len(indices))
        closest = np.argsort(distances)[:n]
        
        return clusters_df.iloc[indices[closest]]['seqID'].tolist()

    def compile_candidates(self, noise_df, low_purity_df, unknown_df, isolated_df, 
                          clusters_df, analysis_df, embeddings):
        logger.info("\nCompiling candidates...")
        
        candidates = []
        cid = 1
        
        # Noise by marker
        if len(noise_df) > 0:
            for marker in noise_df['marker'].unique():
                marker_noise = noise_df[noise_df['marker'] == marker]
                if len(marker_noise) >= self.min_novel_size:
                    reps = marker_noise['seqID'].head(5).tolist()
                    candidates.append({
                        'candidate_id': cid,
                        'type': 'noise',
                        'cluster_id': -1,
                        'marker': marker,
                        'size': len(marker_noise),
                        'purity': 0.0,
                        'majority_taxid': 'NOISE',
                        'majority_name': 'NOISE',
                        'representative_seqids': ','.join(reps),
                        'description': f'HDBSCAN noise ({marker})'
                    })
                    cid += 1
            logger.info(f"  Noise: {cid - 1} groups")
        
        # Low purity
        if len(low_purity_df) > 0:
            for _, row in low_purity_df.iterrows():
                reps = self.extract_representatives(row['cluster_id'], clusters_df, embeddings)
                candidates.append({
                    'candidate_id': cid,
                    'type': 'low_purity',
                    'cluster_id': row['cluster_id'],
                    'marker': 'mixed',
                    'size': row['size'],
                    'purity': row['purity'],
                    'majority_taxid': row['majority_taxid'],
                    'majority_name': row['majority_name'],
                    'representative_seqids': ','.join(reps),
                    'description': f'Low purity (p={row["purity"]:.3f})'
                })
                cid += 1
            logger.info(f"  Low-purity: {len(low_purity_df)} clusters")
        
        # Unknown taxonomy
        if len(unknown_df) > 0:
            for _, row in unknown_df.iterrows():
                reps = self.extract_representatives(row['cluster_id'], clusters_df, embeddings)
                candidates.append({
                    'candidate_id': cid,
                    'type': 'unknown_taxonomy',
                    'cluster_id': row['cluster_id'],
                    'marker': 'mixed',
                    'size': row['size'],
                    'purity': 0.0,
                    'majority_taxid': 'UNKNOWN',
                    'majority_name': 'UNKNOWN',
                    'representative_seqids': ','.join(reps),
                    'description': 'No known taxonomy'
                })
                cid += 1
            logger.info(f"  Unknown: {len(unknown_df)} clusters")
        
        # Isolated by marker
        if len(isolated_df) > 0:
            for marker in isolated_df['marker'].unique():
                marker_iso = isolated_df[isolated_df['marker'] == marker]
                if len(marker_iso) >= self.min_novel_size:
                    top = marker_iso.nlargest(5, 'isolation_score')
                    reps = top['seqID'].tolist()
                    candidates.append({
                        'candidate_id': cid,
                        'type': 'isolated',
                        'cluster_id': -1,
                        'marker': marker,
                        'size': len(marker_iso),
                        'purity': 0.0,
                        'majority_taxid': 'ISOLATED',
                        'majority_name': 'ISOLATED',
                        'representative_seqids': ','.join(reps),
                        'description': f'Highly isolated ({marker})'
                    })
                    cid += 1
            logger.info(f"  Isolated: {cid - len(candidates) - 1} groups")
        
        df = pd.DataFrame(candidates)
        logger.info(f"\nTotal candidates: {len(df)}")
        
        if len(df) > 0:
            for t in df['type'].unique():
                count = (df['type'] == t).sum()
                seqs = df[df['type'] == t]['size'].sum()
                logger.info(f"  {t}: {count} ({seqs:,} seqs)")
        
        return df

    def save_results(self, candidates_df, noise_df, isolated_df):
        logger.info("\nSaving results...")
        
        candidates_df.to_csv(self.output_dir / "novel_candidates.csv", index=False)
        logger.info(f"Saved: novel_candidates.csv ({len(candidates_df)} records)")
        
        noise_df.to_csv(self.output_dir / "noise_sequences.csv", index=False)
        logger.info(f"Saved: noise_sequences.csv ({len(noise_df):,} records)")
        
        isolated_df.to_csv(self.output_dir / "isolated_sequences.csv", index=False)
        logger.info(f"Saved: isolated_sequences.csv ({len(isolated_df):,} records)")
        
        summary = {
            'total_candidates': len(candidates_df),
            'total_sequences': int(candidates_df['size'].sum()),
            'by_type': candidates_df['type'].value_counts().to_dict(),
            'parameters': {
                'purity_threshold': self.purity_threshold,
                'min_novel_size': self.min_novel_size,
                'isolation_percentile': self.isolation_percentile
            }
        }
        
        with open(self.output_dir / "novelty_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved: novelty_summary.json")

    def run(self):
        logger.info("\nNOVELTY DETECTION PIPELINE")
        
        clusters_df, analysis_df, embeddings, metadata = self.load_data()
        
        noise_df = self.identify_noise(clusters_df)
        low_purity_df = self.identify_low_purity(analysis_df)
        unknown_df = self.identify_unknown_taxonomy(analysis_df)
        
        isolation_scores = self.compute_isolation_advanced(embeddings, clusters_df, k=30)
        isolated_df = self.identify_isolated(isolation_scores, clusters_df)
        
        candidates_df = self.compile_candidates(
            noise_df, low_purity_df, unknown_df, isolated_df,
            clusters_df, analysis_df, embeddings
        )
        
        self.save_results(candidates_df, noise_df, isolated_df)
        logger.info("\nNovelty detection complete!")


def main():
    detector = NoveltyDetector(
        clusters_dir="dataset/clusters",
        embeddings_dir="dataset/embeddings",
        metadata_file="dataset/metadata.csv",
        output_dir="dataset/novelty",
        purity_threshold=0.3,
        min_novel_size=5,
        isolation_percentile=85
    )
    
    try:
        detector.run()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())