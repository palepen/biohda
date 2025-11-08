"""
Step 7: HDBSCAN Clustering with cuML (GPU Accelerated)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import json
from collections import Counter
from tqdm import tqdm
import yaml

try:
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logging.warning("cuML not available. Using CPU HDBSCAN.")
    import hdbscan

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HDBSCANClustering:
    """GPU-accelerated HDBSCAN clustering"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.embeddings_dir = Path(config['paths']['embeddings_dir'])

        self.output_dir = Path(config['paths']['clusters_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.taxonomy_files = config['paths']['taxonomy_files']
        
        cluster_cfg = config['clustering']
        self.min_cluster_size = cluster_cfg['min_cluster_size']
        self.min_samples = cluster_cfg['min_samples']
        self.cluster_selection_epsilon = cluster_cfg['cluster_selection_epsilon']
        self.metric = cluster_cfg['metric']
        self.normalize = cluster_cfg['normalize']
        
        logger.info("HDBSCAN Configuration:")
        logger.info(f"  min_cluster_size: {self.min_cluster_size}")
        logger.info(f"  min_samples: {self.min_samples}")
        logger.info(f"  cluster_selection_epsilon: {self.cluster_selection_epsilon}")
        logger.info(f"  metric: {self.metric}")
        logger.info(f"  GPU: {CUML_AVAILABLE}")

    def load_embeddings(self):
        logger.info("\nLoading embeddings...")
        emb_file = self.embeddings_dir / "embeddings.npy"
        embeddings = np.load(emb_file)
        
        with open(self.embeddings_dir / "embedding_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"Shape: {embeddings.shape}, Sequences: {len(metadata['seq_ids']):,}")
        return embeddings, metadata

    def load_taxonomy(self):
        logger.info("Loading taxonomy from config file list...")
        
        if not self.taxonomy_files:
            logger.error("No files found in 'paths.taxonomy_files' list in config.")
            raise ValueError("Config 'paths.taxonomy_files' is empty.")

        logger.info(f"Found {len(self.taxonomy_files)} TSV files specified in config...")

        # --- MODIFICATION: Define column names based on your data snippet ---
        # Your data has 4 columns: accession, taxid, species_name, full_lineage
        col_names = ['accession', 'taxid', 'species_name', 'full_lineage']
        # --- End of MODIFICATION ---

        all_dfs = []
        for file_path_str in self.taxonomy_files:
            file_path = Path(file_path_str)
            if not file_path.exists():
                logger.warning(f"File not found, skipping: {file_path_str}")
                continue
                
            try:
                # --- MODIFICATION: Add header=None and names=col_names ---
                df = pd.read_csv(
                    file_path, 
                    sep='\t',
                    header=None,     # Tell pandas there is NO header row
                    names=col_names, # Manually provide the column names
                    dtype={'taxid': str} 
                )
                # --- End of MODIFICATION ---

                all_dfs.append(df)
                logger.info(f"  Loaded {file_path.name} ({len(df):,} records)")
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
        
        if not all_dfs:
            logger.error("No data loaded. All specified TSV files failed to parse or were not found.")
            raise ValueError("No valid taxonomy data loaded from 'paths.taxonomy_files'.")
            
        # Combine all individual DataFrames into one
        metadata_df = pd.concat(all_dfs, ignore_index=True)

        # This rename block is still correct and NECESSARY
        logger.info("Normalizing taxonomy column names...")
        column_map = {
            'accession': 'seqID',
            'taxid': 'taxID',
            'species_name': 'scientific_name'
            # 'full_lineage' is also loaded, but not renamed (which is fine)
        }
        
        cols_to_rename = {k: v for k, v in column_map.items() if k in metadata_df.columns}
        
        if cols_to_rename:
            metadata_df.rename(columns=cols_to_rename, inplace=True)
            logger.info(f"Renamed columns: {cols_to_rename}")

        logger.info(f"Total records loaded: {len(metadata_df):,}")
        
        # This check should now pass
        expected_cols = ['seqID', 'taxID', 'scientific_name']
        if not all(col in metadata_df.columns for col in expected_cols):
            logger.warning("Loaded data is still missing one or more expected columns: "
                           "'seqID', 'taxID', 'scientific_name'. Check TSV format.")
        else:
            logger.info("Taxonomy columns 'seqID', 'taxID', 'scientific_name' loaded successfully.")
                           
        return metadata_df

    def normalize_embeddings(self, embeddings):
        logger.info("Normalizing embeddings...")
        scaler = StandardScaler()
        emb_norm = scaler.fit_transform(embeddings)
        logger.info(f"Mean: {emb_norm.mean():.4f}, Std: {emb_norm.std():.4f}")
        return emb_norm, scaler

    def perform_clustering(self, embeddings):
        logger.info("\nRunning HDBSCAN...")
        
        if CUML_AVAILABLE:
            logger.info("Using cuML GPU-accelerated HDBSCAN")
            clusterer = cuHDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric=self.metric,
                cluster_selection_method='leaf',
                prediction_data=True
            )
        else:
            logger.info("Using CPU HDBSCAN")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric=self.metric,
                cluster_selection_method='leaf',
                core_dist_n_jobs=-1
            )
        
        clusterer.fit(embeddings)
        
        if CUML_AVAILABLE:
            labels = clusterer.labels_.to_numpy() if hasattr(clusterer.labels_, 'to_numpy') else clusterer.labels_
            probs = clusterer.probabilities_.to_numpy() if hasattr(clusterer.probabilities_, 'to_numpy') else clusterer.probabilities_
        else:
            labels = clusterer.labels_
            probs = clusterer.probabilities_
        
        unique = set(labels)
        n_clusters = len(unique) - (1 if -1 in unique else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"\nResults:")
        logger.info(f"  Clusters: {n_clusters}")
        logger.info(f"  Noise: {n_noise:,} ({n_noise/len(labels):.2%})")
        logger.info(f"  Clustered: {len(labels) - n_noise:,}")
        
        cluster_sizes = Counter(labels[labels != -1])
        if cluster_sizes:
            sizes = list(cluster_sizes.values())
            logger.info(f"  Size: min={min(sizes):,}, max={max(sizes):,}, mean={np.mean(sizes):.1f}")
        
        return clusterer, labels, probs

    def compute_metrics(self, embeddings, labels):
        logger.info("\nComputing metrics...")
        metrics = {}
        
        mask = labels != -1
        emb_valid = embeddings[mask]
        labels_valid = labels[mask]
        
        if len(np.unique(labels_valid)) > 1:
            sil = silhouette_score(emb_valid, labels_valid, sample_size=min(10000, len(labels_valid)))
            metrics['silhouette_score'] = float(sil)
            logger.info(f"  Silhouette: {sil:.4f}")
            
            ch = calinski_harabasz_score(emb_valid, labels_valid)
            metrics['calinski_harabasz_score'] = float(ch)
            logger.info(f"  Calinski-Harabasz: {ch:.2f}")
            
            db = davies_bouldin_score(emb_valid, labels_valid)
            metrics['davies_bouldin_score'] = float(db)
            logger.info(f"  Davies-Bouldin: {db:.4f}")
        else:
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz_score'] = None
            metrics['davies_bouldin_score'] = None
        
        return metrics

    def analyze_taxonomy(self, labels, seq_ids, metadata_df):
        logger.info("\nAnalyzing cluster taxonomy...")
        
        seqid_to_taxid = dict(zip(metadata_df['seqID'], metadata_df['taxID']))
        seqid_to_name = dict(zip(metadata_df['seqID'], metadata_df['scientific_name']))
        
        analysis = []
        unique_clusters = sorted(set(labels))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        for cid in tqdm(unique_clusters, desc="Analyzing"):
            mask = labels == cid
            cluster_seqs = [seq_ids[i] for i in range(len(seq_ids)) if mask[i]]
            
            taxids = [seqid_to_taxid.get(s, 'NA') for s in cluster_seqs]
            names = [seqid_to_name.get(s, 'NA') for s in cluster_seqs]
            
            valid_taxids = [t for t in taxids if t != 'NA']
            valid_names = [n for n in names if n != 'NA']
            
            if valid_taxids:
                most_common_taxid, count = Counter(valid_taxids).most_common(1)[0]
                purity = count / len(valid_taxids)
                majority_name = Counter(valid_names).most_common(1)[0][0]
            else:
                purity = 0.0
                most_common_taxid = 'NA'
                majority_name = 'NA'
                count = 0
            
            analysis.append({
                'cluster_id': cid,
                'size': len(cluster_seqs),
                'num_taxa': len(set(valid_taxids)),
                'purity': purity,
                'majority_taxid': most_common_taxid,
                'majority_name': majority_name,
                'majority_count': count,
                'has_taxonomy': len(valid_taxids) > 0
            })
        
        noise_mask = labels == -1
        n_noise = noise_mask.sum()
        if n_noise > 0:
            noise_seqs = [seq_ids[i] for i in range(len(seq_ids)) if noise_mask[i]]
            noise_taxids = [seqid_to_taxid.get(s, 'NA') for s in noise_seqs]
            valid_noise = [t for t in noise_taxids if t != 'NA']
            
            analysis.append({
                'cluster_id': -1,
                'size': n_noise,
                'num_taxa': len(set(valid_noise)),
                'purity': 0.0,
                'majority_taxid': 'NOISE',
                'majority_name': 'NOISE',
                'majority_count': 0,
                'has_taxonomy': len(valid_noise) > 0
            })
        
        df = pd.DataFrame(analysis)
        logger.info(f"Analyzed {len(df)-1} clusters + noise")
        return df

    def save_results(self, clusterer, labels, probs, metadata, cluster_analysis, metrics):
        logger.info("\nSaving results...")
        
        results_df = pd.DataFrame({
            'seqID': metadata['seq_ids'],
            'marker': metadata['markers'],
            'cluster_id': labels,
            'cluster_probability': probs
        })
        
        results_df.to_csv(self.output_dir / "clusters.csv", index=False)
        logger.info(f"Saved: clusters.csv ({len(results_df):,} records)")
        
        cluster_analysis.to_csv(self.output_dir / "cluster_analysis.csv", index=False)
        logger.info(f"Saved: cluster_analysis.csv")
        
        with open(self.output_dir / "hdbscan_model.pkl", 'wb') as f:
            pickle.dump(clusterer, f)
        logger.info(f"Saved: hdbscan_model.pkl")
        
        config = {
            'parameters': {
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'metric': self.metric,
                'normalize': self.normalize,
                'gpu': CUML_AVAILABLE
            },
            'results': {
                'n_clusters': int(len(set(labels)) - (1 if -1 in labels else 0)),
                'n_noise': int(list(labels).count(-1)),
                'n_sequences': len(labels)
            },
            'metrics': metrics
        }
        
        with open(self.output_dir / "clustering_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved: clustering_config.json")
        
        valid = cluster_analysis[cluster_analysis['cluster_id'] != -1]
        if len(valid) > 0:
            logger.info(f"\nPurity: high(>0.8)={sum(valid['purity'] > 0.8)}, "
                       f"med(0.5-0.8)={sum((valid['purity'] > 0.5) & (valid['purity'] <= 0.8))}, "
                       f"low(<0.5)={sum(valid['purity'] <= 0.5)}")

    def run(self):
        logger.info("\nHDBSCAN CLUSTERING PIPELINE")
        
        embeddings, emb_metadata = self.load_embeddings()
        tax_metadata = self.load_taxonomy()
        
        if self.normalize:
            embeddings, scaler = self.normalize_embeddings(embeddings)
            with open(self.output_dir / "scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
        
        clusterer, labels, probs = self.perform_clustering(embeddings)
        metrics = self.compute_metrics(embeddings, labels)
        cluster_analysis = self.analyze_taxonomy(labels, emb_metadata['seq_ids'], tax_metadata)
        
        self.save_results(clusterer, labels, probs, emb_metadata, cluster_analysis, metrics)
        
        logger.info("\nClustering complete")


def main():
    clustering = HDBSCANClustering(config_path="config.yaml")
    
    try:
        clustering.run()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())