"""
Step 7: Memory-Optimized HDBSCAN Clustering with cuML (GPU Accelerated)
Optimized for systems with limited GPU VRAM (4GB) and RAM (16GB)
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
import gc

try:
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    import cupy as cp
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logging.warning("cuML not available. Using CPU HDBSCAN.")
    import hdbscan

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HDBSCANClustering:
    """Memory-optimized GPU-accelerated HDBSCAN clustering"""
    
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
        
        # Memory optimization settings
        self.batch_size = 10000  # Process embeddings in batches
        self.max_metric_samples = 5000  # Limit samples for metric computation
        
        logger.info("HDBSCAN Configuration:")
        logger.info(f"  min_cluster_size: {self.min_cluster_size}")
        logger.info(f"  min_samples: {self.min_samples}")
        logger.info(f"  cluster_selection_epsilon: {self.cluster_selection_epsilon}")
        logger.info(f"  metric: {self.metric}")
        logger.info(f"  GPU: {CUML_AVAILABLE}")
        logger.info(f"  Batch size: {self.batch_size}")

    def load_embeddings(self):
        """Load embeddings with memory mapping to avoid loading all into RAM"""
        logger.info("\nLoading embeddings...")
        emb_file = self.embeddings_dir / "embeddings.npy"
        
        # Use memory mapping to avoid loading entire array
        embeddings = np.load(emb_file, mmap_mode='r')
        
        with open(self.embeddings_dir / "embedding_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"Shape: {embeddings.shape}, Sequences: {len(metadata['seq_ids']):,}")
        logger.info(f"Memory-mapped embeddings (not loaded into RAM yet)")
        return embeddings, metadata

    def load_taxonomy(self):
        """Load taxonomy with chunked reading for large files"""
        logger.info("Loading taxonomy from config file list...")
        
        if not self.taxonomy_files:
            logger.error("No files found in 'paths.taxonomy_files' list in config.")
            raise ValueError("Config 'paths.taxonomy_files' is empty.")

        logger.info(f"Found {len(self.taxonomy_files)} TSV files specified in config...")
        col_names = ['accession', 'taxid', 'species_name', 'full_lineage']

        all_dfs = []
        for file_path_str in self.taxonomy_files:
            file_path = Path(file_path_str)
            if not file_path.exists():
                logger.warning(f"File not found, skipping: {file_path_str}")
                continue
                
            try:
                # Read in chunks to reduce memory usage
                chunks = []
                for chunk in pd.read_csv(
                    file_path, 
                    sep='\t',
                    header=None,
                    names=col_names,
                    dtype={'taxid': str},
                    chunksize=50000  # Process 50k rows at a time
                ):
                    # Keep only necessary columns
                    chunk = chunk[['accession', 'taxid', 'species_name']]
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
                all_dfs.append(df)
                logger.info(f"  Loaded {file_path.name} ({len(df):,} records)")
                
                # Clean up
                del chunks
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
        
        if not all_dfs:
            logger.error("No data loaded. All specified TSV files failed to parse or were not found.")
            raise ValueError("No valid taxonomy data loaded from 'paths.taxonomy_files'.")
            
        metadata_df = pd.concat(all_dfs, ignore_index=True)
        
        logger.info("Normalizing taxonomy column names...")
        column_map = {
            'accession': 'seqID',
            'taxid': 'taxID',
            'species_name': 'scientific_name'
        }
        
        cols_to_rename = {k: v for k, v in column_map.items() if k in metadata_df.columns}
        
        if cols_to_rename:
            metadata_df.rename(columns=cols_to_rename, inplace=True)
            logger.info(f"Renamed columns: {cols_to_rename}")

        logger.info(f"Total records loaded: {len(metadata_df):,}")
        
        # Remove duplicates to save memory
        metadata_df = metadata_df.drop_duplicates(subset=['seqID'])
        logger.info(f"After deduplication: {len(metadata_df):,}")
        
        return metadata_df

    def normalize_embeddings(self, embeddings):
        """Normalize embeddings in batches to save memory"""
        logger.info("Normalizing embeddings in batches...")
        
        # Fit scaler on a sample to save memory
        sample_size = min(50000, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample = np.array(embeddings[sample_indices])
        
        scaler = StandardScaler()
        scaler.fit(sample)
        
        del sample
        gc.collect()
        
        # Normalize in batches
        n_samples = len(embeddings)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        normalized = np.zeros_like(embeddings)
        
        for i in tqdm(range(n_batches), desc="Normalizing"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            
            batch = np.array(embeddings[start_idx:end_idx])
            normalized[start_idx:end_idx] = scaler.transform(batch)
            
            del batch
            if i % 10 == 0:
                gc.collect()
        
        logger.info(f"Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
        return normalized, scaler

    def perform_clustering(self, embeddings):
        """Perform clustering with memory-efficient settings"""
        logger.info("\nRunning HDBSCAN...")
        
        use_gpu = CUML_AVAILABLE
        
        if use_gpu:
            logger.info("Using cuML GPU-accelerated HDBSCAN")
            
            # Convert to float32 to save GPU memory
            if embeddings.dtype != np.float32:
                logger.info("Converting to float32 to save GPU memory...")
                embeddings = embeddings.astype(np.float32)
            
            try:
                clusterer = cuHDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                    metric=self.metric,
                    cluster_selection_method='leaf',
                    prediction_data=False  # Disable to save memory
                )
                
                clusterer.fit(embeddings)
                
                # Immediately extract results and clear GPU memory
                labels = clusterer.labels_.to_numpy() if hasattr(clusterer.labels_, 'to_numpy') else clusterer.labels_
                probs = clusterer.probabilities_.to_numpy() if hasattr(clusterer.probabilities_, 'to_numpy') else clusterer.probabilities_
                
                # Clear GPU memory
                if hasattr(cp, 'get_default_memory_pool'):
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                
            except Exception as e:
                logger.error(f"GPU clustering failed: {e}")
                logger.info("Falling back to CPU HDBSCAN...")
                use_gpu = False
        
        if not use_gpu:
            logger.info("Using CPU HDBSCAN")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric=self.metric,
                cluster_selection_method='leaf',
                core_dist_n_jobs=-1,
                memory=str(self.output_dir / 'hdbscan_cache')  # Use disk cache
            )
            
            clusterer.fit(embeddings)
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
        """Compute metrics on a sample to save memory"""
        logger.info("\nComputing metrics on sample...")
        metrics = {}
        
        mask = labels != -1
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 1:
            # Sample for metric computation
            sample_size = min(self.max_metric_samples, len(valid_indices))
            sample_indices = np.random.choice(valid_indices, sample_size, replace=False)
            
            emb_sample = np.array(embeddings[sample_indices])
            labels_sample = labels[sample_indices]
            
            if len(np.unique(labels_sample)) > 1:
                sil = silhouette_score(emb_sample, labels_sample)
                metrics['silhouette_score'] = float(sil)
                logger.info(f"  Silhouette (sampled): {sil:.4f}")
                
                ch = calinski_harabasz_score(emb_sample, labels_sample)
                metrics['calinski_harabasz_score'] = float(ch)
                logger.info(f"  Calinski-Harabasz (sampled): {ch:.2f}")
                
                db = davies_bouldin_score(emb_sample, labels_sample)
                metrics['davies_bouldin_score'] = float(db)
                logger.info(f"  Davies-Bouldin (sampled): {db:.4f}")
            
            del emb_sample, labels_sample
            gc.collect()
        else:
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz_score'] = None
            metrics['davies_bouldin_score'] = None
        
        return metrics

    def analyze_taxonomy(self, labels, seq_ids, metadata_df):
        """Analyze taxonomy with memory-efficient processing"""
        logger.info("\nAnalyzing cluster taxonomy...")
        
        # Create dictionaries for fast lookup
        seqid_to_taxid = dict(zip(metadata_df['seqID'], metadata_df['taxID']))
        seqid_to_name = dict(zip(metadata_df['seqID'], metadata_df['scientific_name']))
        
        # Clear the dataframe to save memory
        del metadata_df
        gc.collect()
        
        analysis = []
        unique_clusters = sorted(set(labels))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        # Process in batches
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
            
            # Periodic garbage collection
            if len(analysis) % 100 == 0:
                gc.collect()
        
        # Handle noise
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
        """Save results in chunks to manage memory"""
        logger.info("\nSaving results...")
        
        # Save in chunks
        chunk_size = 100000
        n_chunks = (len(labels) + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(labels))
            
            chunk_df = pd.DataFrame({
                'seqID': metadata['seq_ids'][start_idx:end_idx],
                'marker': metadata['markers'][start_idx:end_idx],
                'cluster_id': labels[start_idx:end_idx],
                'cluster_probability': probs[start_idx:end_idx]
            })
            
            mode = 'w' if i == 0 else 'a'
            header = i == 0
            chunk_df.to_csv(self.output_dir / "clusters.csv", mode=mode, header=header, index=False)
            
            del chunk_df
            gc.collect()
        
        logger.info(f"Saved: clusters.csv ({len(labels):,} records)")
        
        cluster_analysis.to_csv(self.output_dir / "cluster_analysis.csv", index=False)
        logger.info(f"Saved: cluster_analysis.csv")
        
        # Save only essential parts of the model
        with open(self.output_dir / "hdbscan_model.pkl", 'wb') as f:
            pickle.dump(clusterer, f, protocol=pickle.HIGHEST_PROTOCOL)
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
        logger.info("\nMEMORY-OPTIMIZED HDBSCAN CLUSTERING PIPELINE")
        logger.info("=" * 60)
        
        embeddings, emb_metadata = self.load_embeddings()
        tax_metadata = self.load_taxonomy()
        
        if self.normalize:
            embeddings, scaler = self.normalize_embeddings(embeddings)
            with open(self.output_dir / "scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            del scaler
            gc.collect()
        
        clusterer, labels, probs = self.perform_clustering(embeddings)
        metrics = self.compute_metrics(embeddings, labels)
        
        # Clear embeddings from memory after clustering
        del embeddings
        gc.collect()
        
        cluster_analysis = self.analyze_taxonomy(labels, emb_metadata['seq_ids'], tax_metadata)
        self.save_results(clusterer, labels, probs, emb_metadata, cluster_analysis, metrics)
        
        logger.info("\nClustering complete")
        logger.info("=" * 60)


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