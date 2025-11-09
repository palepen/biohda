# eDNA Analysis Pipeline for Novel Taxa Discovery

## 🧬 Overview

A comprehensive deep learning pipeline for analyzing environmental DNA (eDNA) sequences to discover novel taxa. The pipeline combines Chaos Game Representation (CGR), convolutional neural networks, contrastive learning, and GPU-accelerated clustering to identify potentially novel species, genera, and higher taxonomic groups.

## 🎯 Key Features

- **Multi-marker Support**: Processes ITS, LSU, and SSU rRNA sequences
- **Deep Learning**: CNN encoder with contrastive learning for robust embeddings
- **GPU Acceleration**: CUDA-enabled training and cuML HDBSCAN clustering
- **Local BLAST**: Multi-database support for offline annotation
- **Comprehensive Evaluation**: Multi-level taxonomic accuracy assessment
- **Novelty Detection**: Automated identification of potential novel taxa

## 📊 Pipeline Architecture

```
Raw FASTA → Preprocessing → CGR Transform → CNN Encoder → Embeddings
                                                              ↓
Novel Taxa ← Evaluation ← Annotation ← Clustering ← Embeddings
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- BLAST+ toolkit
- NCBI taxonomy database

### Setup

```bash
# Clone repository
git clone <repository-url>
cd edna-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NCBI taxonomy
cd taxdmp
wget https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
tar -xzf taxdump.tar.gz
cd ..

# Build taxonomy database
python genDb.py
python buildRank.py
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
biopython>=1.81
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.65.0
hdbscan>=0.8.33
cuml-cu11>=23.10.0  # Optional, for GPU clustering
```

## 📁 Project Structure

```
edna-pipeline/
├── dataset/
│   ├── raw/              # Raw BLAST databases
│   ├── processed/        # Cleaned FASTA files
│   ├── cgr/             # CGR images
│   ├── embeddings/      # CNN embeddings
│   ├── clusters/        # Clustering results
│   ├── annotation/      # Taxonomic annotations
│   ├── evaluation/      # Performance metrics
│   └── novelty/         # Novel taxa candidates
├── models/              # Trained models
├── logs/               # Execution logs
├── taxdmp/             # NCBI taxonomy dump
├── config.yaml         # Configuration file
└── *.py               # Pipeline scripts
```

## 🚀 Quick Start

### 1. Configure Pipeline

Edit `config.yaml`:

```yaml
paths:
  taxonomy_files: 
    - "dataset/ssu_accession_full_lineage.tsv"
  
annotation:
  databases:
    - "./dataset/raw/SSU_eukaryote_rRNA/SSU_eukaryote_rRNA"
  identity_threshold_novel: 95.0
  identity_threshold_species: 97.0

clustering:
  min_cluster_size: 2
  min_samples: 1
  cluster_selection_epsilon: 0.5
```

### 2. Run Full Pipeline

```bash
# Step 1: Extract sequences from BLAST databases
python fasta_extraction.py

# Step 2: Preprocess sequences
python minimal_preprocessing.py

# Step 3: Extract metadata
python metadata_extraction.py

# Step 4: Generate CGR images
python cgr_transformation.py

# Step 5: Train CNN encoder
python cnn_encoder.py

# Step 6: Generate embeddings
python generate_embeddings.py

# Step 7: Cluster sequences
python clustering.py

# Step 8: Annotate clusters
python cluster_annotation.py

# Step 9: Evaluate results
python evaluation.py

# Step 10: Detect novel taxa
python novelty.py
```

## 📖 Pipeline Steps Explained

### Step 1: FASTA Extraction
- Extracts sequences from NCBI BLAST databases
- Uses `blastdbcmd` utility
- Preserves full sequence headers

### Step 2: Minimal Preprocessing
- Removes sequences <100 bp
- Filters sequences with >10% ambiguous bases
- Converts to uppercase
- Preserves biological diversity (no deduplication)

### Step 3: Metadata Extraction
- Parses taxonomy from TSV files
- Maps sequence IDs to taxonomic lineages
- Handles missing taxonomy gracefully

### Step 4: CGR Transformation
- Converts DNA sequences to 2D images
- Image size: 64x64 or 128x128 pixels
- Log-transforms frequency counts
- Memory-efficient processing

### Step 5: CNN Encoder Training
- Contrastive learning with NT-Xent loss
- 4-layer CNN architecture
- Generates 128-dimensional embeddings
- GPU-accelerated training

### Step 6: Embedding Generation
- Batch inference on all sequences
- Memory-mapped data loading
- Outputs: embeddings.npy, metadata.pkl

### Step 7: HDBSCAN Clustering
- GPU-accelerated (cuML) or CPU fallback
- Density-based clustering
- Automatic cluster number detection
- Identifies noise sequences

### Step 8: Cluster Annotation
- Multi-database local BLAST search
- Taxonomy lookup from local files
- Identity-based novelty classification
- No API calls (fully offline)

### Step 9: Evaluation
- Multi-level taxonomic accuracy
- Novelty detection metrics
- Cluster purity analysis
- Comprehensive reporting

### Step 10: Novelty Detection
- Score-based candidate ranking
- Considers taxonomy gaps and diversity
- Generates prioritized candidate lists

## 📊 Input Data Format

### Taxonomy TSV Format

```
accession   taxid   species_name            full_lineage
JF718645.1  1001139 Apokeronopsis ovalis    cellular organisms;Eukaryota;Sar;Alveolata;...
KX364317.1  1825157 Anteholosticha manca    cellular organisms;Eukaryota;Sar;Alveolata;...
```

### FASTA Format

```
>JF718645.1 Apokeronopsis ovalis 18S ribosomal RNA gene
ATCTGGTTGATCCTGCCAGTAGTCATATGCTTGTCTCAAAGATTAAGCCATGCATGTCTAAGTATAA...
```

## 🔬 Methods

### Chaos Game Representation (CGR)

Transforms DNA sequences into 2D images by recursively plotting positions in a unit square with corners representing nucleotides (A, T, G, C). Each nucleotide determines the direction of the next plot point, creating unique visual patterns.

### Contrastive Learning

Uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss to learn embeddings where:
- Similar sequences cluster together
- Dissimilar sequences are pushed apart
- Augmented views of same sequence are positive pairs

### HDBSCAN Clustering

Hierarchical Density-Based Spatial Clustering of Applications with Noise:
- Converts space into density graph
- Extracts stable clusters at multiple scales
- Identifies noise points (potential singletons or novel taxa)

### Taxonomic Annotation

1. **BLAST Search**: Queries multiple local databases
2. **Best Hit Selection**: Highest identity across all databases
3. **Taxonomy Lookup**: Maps accessions to full lineages
4. **Novelty Classification**:
   - <95% identity → Potentially novel genus/family
   - 95-97% identity → Potentially novel species
   - >97% identity → Known species

### Shared Taxonomy Parser

`taxonomy_parser.py` ensures consistent parsing across pipeline:
- Loads global rank map from SQLite database
- Parses NCBI-style lineages
- Handles 8 taxonomic levels: domain, kingdom, phylum, class, order, family, genus, species

## 📈 Performance Metrics

### Evaluation Metrics

1. **Taxonomic Accuracy** (per level)
   - Genus accuracy: ~52%
   - Species accuracy: ~38%

2. **Novelty Detection**
   - Precision: 0.070
   - Recall: 0.230
   - F1-score: 0.108

3. **Cluster Quality**
   - Mean purity (genus): 0.594
   - Median purity :0.500

### Processing Speed

- **CGR transformation**: ~1,000 sequences/second
- **CNN inference**: ~5,000 sequences/second (GPU)
- **Clustering**: ~10,000 sequences/second (cuML GPU)
- **BLAST annotation**: ~0.01 seconds/cluster (local)

### Resource Requirements

- **Memory**: 8-16 GB RAM (32 GB recommended for large datasets)
- **Storage**: 10-50 GB (depends on dataset size)
- **GPU**: NVIDIA GPU with 6+ GB VRAM (optional but recommended)


## 🔧 Configuration Reference

### Complete config.yaml

```yaml
paths:
  raw_dir: "dataset/raw"
  processed_dir: "dataset/processed"
  cgr_dir: "dataset/cgr"
  models_dir: "models"
  embeddings_dir: "dataset/embeddings"
  clusters_dir: "dataset/clusters"
  annotation_dir: "dataset/annotation"
  evaluation_dir: "dataset/evaluation"
  taxonomy_files: 
    - "dataset/ssu_accession_full_lineage.tsv"

cgr:
  image_size: 64
  log_transform: true

training:
  embedding_dim: 128
  batch_size: 128
  num_epochs: 50
  learning_rate: 0.001
  device: "cuda"
  weight_decay: 0.0001
  temperature: 0.5

clustering:
  min_cluster_size: 2
  min_samples: 1
  cluster_selection_epsilon: 0.5
  metric: "l2"
  normalize: true

annotation:
  identity_threshold_novel: 95.0
  identity_threshold_species: 97.0
  use_local_blast: true
  databases: 
    - "./dataset/raw/SSU_eukaryote_rRNA/SSU_eukaryote_rRNA"
  max_clusters_to_annotate: null
  batch_size: null
  resume_from_checkpoint: false
  min_alignment_length: 100
  max_evalue: 0.00001

resources:
  num_workers: 12
  use_gpu: true
  use_mmap: true

logging:
  level: "INFO"
  save_logs: true
  log_dir: "logs"
```


---
