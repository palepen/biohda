## HDA

---
dataset should be in dataset/raw
---

sample tree after running
```
├── blast_validation.py
├── cgr_transformation.py
├── clustering.py
├── cnn_encoder.py
├── cnn_encoder_super.py
├── dataset
│   ├── cgr
│   │   ├── combined_cgr.npy
│   │   ├── combined_metadata.pkl
│   │   ├── ITS_cgr.npy
│   │   ├── ITS_seqids.pkl
│   │   ├── ITS_stats.pkl
│   │   ├── LSU_cgr.npy
│   │   ├── LSU_seqids.pkl
│   │   ├── LSU_stats.pkl
│   │   ├── SSU_cgr.npy
│   │   ├── SSU_seqids.pkl
│   │   └── SSU_stats.pkl
│   ├── clusters
│   │   ├── cluster_analysis.csv
│   │   ├── clustering_config.json
│   │   ├── clusters.csv
│   │   ├── hdbscan_model.pkl
│   │   └── scaler.pkl
│   ├── embeddings
│   │   ├── embedding_metadata.pkl
│   │   ├── embeddings.npy
│   │   └── embedding_stats.pkl
│   ├── evaluation
│   │   ├── clustering_metrics.json
│   │   ├── cluster_statistics.csv
│   │   ├── evaluation_report.txt
│   │   ├── marker_metrics.json
│   │   └── novelty_metrics.json
│   ├── fasta
│   │   ├── ITS_eukaryote_sequences.fasta
│   │   ├── LSU_eukaryote_rRNA.fasta
│   │   └── SSU_eukaryote_rRNA.fasta
│   ├── metadata.csv
│   ├── novelty
│   │   ├── isolated_sequences.csv
│   │   ├── noise_sequences.csv
│   │   ├── novel_candidates.csv
│   │   └── novelty_summary.json
│   ├── processed
│   │   ├── ITS_clean.fasta
│   │   ├── LSU_clean.fasta
│   │   └── SSU_clean.fasta
│   ├── raw
│   │   ├── ITS_eukaryote_sequences
│   │   │   ├── ITS_eukaryote_sequences.ndb
│   │   │   ├── ITS_eukaryote_sequences.nhr
│   │   │   ├── ITS_eukaryote_sequences.nin
│   │   │   ├── ITS_eukaryote_sequences.nog
│   │   │   ├── ITS_eukaryote_sequences.nos
│   │   │   ├── ITS_eukaryote_sequences.not
│   │   │   ├── ITS_eukaryote_sequences.nsq
│   │   │   ├── ITS_eukaryote_sequences.ntf
│   │   │   ├── ITS_eukaryote_sequences.nto
│   │   │   ├── taxdb.btd
│   │   │   ├── taxdb.bti
│   │   │   └── taxonomy4blast.sqlite3
│   │   ├── LSU_eukaryote_rRNA
│   │   │   ├── LSU_eukaryote_rRNA.ndb
│   │   │   ├── LSU_eukaryote_rRNA.nhr
│   │   │   ├── LSU_eukaryote_rRNA.nin
│   │   │   ├── LSU_eukaryote_rRNA.nog
│   │   │   ├── LSU_eukaryote_rRNA.nos
│   │   │   ├── LSU_eukaryote_rRNA.not
│   │   │   ├── LSU_eukaryote_rRNA.nsq
│   │   │   ├── LSU_eukaryote_rRNA.ntf
│   │   │   ├── LSU_eukaryote_rRNA.nto
│   │   │   ├── taxdb.btd
│   │   │   ├── taxdb.bti
│   │   │   └── taxonomy4blast.sqlite3
│   │   └── SSU_eukaryote_rRNA
│   │       ├── SSU_eukaryote_rRNA.ndb
│   │       ├── SSU_eukaryote_rRNA.nhr
│   │       ├── SSU_eukaryote_rRNA.nin
│   │       ├── SSU_eukaryote_rRNA.nog
│   │       ├── SSU_eukaryote_rRNA.nos
│   │       ├── SSU_eukaryote_rRNA.not
│   │       ├── SSU_eukaryote_rRNA.nsq
│   │       ├── SSU_eukaryote_rRNA.ntf
│   │       ├── SSU_eukaryote_rRNA.nto
│   │       ├── taxdb.btd
│   │       ├── taxdb.bti
│   │       └── taxonomy4blast.sqlite3
│   ├── validation
│   │   ├── high_novelty_candidates.csv
│   │   ├── validated_candidates.csv
│   │   └── validation_summary.json
│   └── visualizations
│       ├── candidate_size_distribution.png
│       ├── cluster_purity_distribution.png
│       ├── cluster_size_distribution.png
│       ├── marker_comparison.png
│       ├── novelty_by_type.png
│       ├── purity_vs_size.png
│       ├── top_clusters_quality.png
│       ├── umap_by_cluster.png
│       ├── umap_by_marker.png
│       ├── umap_by_taxonomy.png
│       └── umap_coordinates.csv
├── evaluation.py
├── fasta_extraction.py
├── generate_embeddings.py
├── metadata_extraction.py
├── minimal_preprocessing.py
├── models
│   ├── cgr_autoencoder_encoder_best.pth
│   ├── cgr_autoencoder_encoder_final.pth
│   ├── cgr_encoder_best.pth
│   └── training_history.json
├── novelty_detection.py
├── readme.md
├── requirements.txt
└── visualization.py
```