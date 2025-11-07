"""
Step 3: Metadata Extraction
Extract taxonomic information directly from TSV file, not FASTA headers.
"""

from pathlib import Path
import logging
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract taxonomic metadata from TSV file"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.fasta_dir = Path(self.config['paths']['processed_dir'])
        self.output_dir = Path(self.config['paths']['processed_dir']).parent
        self.taxonomy_file = Path(self.config['paths']['taxonomy_file'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fasta_files = {
            'ITS': 'ITS_clean.fasta',
            'LSU': 'LSU_clean.fasta',
            'SSU': 'SSU_clean.fasta'
        }
    
    def load_taxonomy_tsv(self) -> pd.DataFrame:
        """Load taxonomy TSV file"""
        logger.info(f"Loading taxonomy from {self.taxonomy_file}...")
        
        tax_df = pd.read_csv(
            self.taxonomy_file,
            sep='\t',
            header=None,
            names=['seqID', 'taxID', 'scientific_name', 'lineage'],
            dtype=str,
            keep_default_na=False
        )
        
        tax_df['lineage'] = tax_df['lineage'].replace('', pd.NA)
        logger.info(f"Loaded {len(tax_df):,} taxonomy records")
        
        return tax_df
    
    def extract_seqids_from_fasta(self, fasta_file: Path, marker: str) -> list:
        """Extract seqIDs from FASTA file"""
        from Bio import SeqIO
        
        logger.info(f"Extracting seqIDs from {fasta_file.name}...")
        seqids = []
        
        for record in SeqIO.parse(fasta_file, 'fasta'):
            seqids.append(record.id)
        
        logger.info(f"Found {len(seqids):,} sequences")
        return seqids
    
    def build_metadata(self) -> pd.DataFrame:
        """Build metadata by joining FASTA seqIDs with taxonomy TSV"""
        
        taxonomy_df = self.load_taxonomy_tsv()
        all_records = []
        
        for marker, fasta_name in self.fasta_files.items():
            logger.info(f"\nProcessing marker: {marker}")
            
            fasta_file = self.fasta_dir / fasta_name
            if not fasta_file.exists():
                logger.warning(f"FASTA file not found: {fasta_file}")
                continue
            
            seqids = self.extract_seqids_from_fasta(fasta_file, marker)
            
            marker_df = pd.DataFrame({'seqID': seqids, 'marker': marker})
            marker_df = marker_df.merge(taxonomy_df, on='seqID', how='left')
            
            marker_df['taxID'] = marker_df['taxID'].fillna('NA')
            marker_df['scientific_name'] = marker_df['scientific_name'].fillna('NA')
            marker_df['lineage'] = marker_df['lineage'].fillna('NA')
            
            all_records.append(marker_df)
            
            matched = (marker_df['taxID'] != 'NA').sum()
            logger.info(f"Matched {matched:,} / {len(seqids):,} ({matched/len(seqids):.1%})")
        
        if not all_records:
            logger.error("No metadata extracted")
            return pd.DataFrame()
        
        return pd.concat(all_records, ignore_index=True)
    
    def save_metadata(self, df: pd.DataFrame):
        """Save metadata to CSV"""
        output_path = self.output_dir / "metadata.csv"
        
        df.to_csv(output_path, index=False)
        logger.info(f"\nSaved metadata to: {output_path}")
        logger.info(f"Total records: {len(df):,}")
        logger.info(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
        
        logger.info("\nMetadata Summary:")
        for marker in df['marker'].unique():
            count = len(df[df['marker'] == marker])
            logger.info(f"  {marker}: {count:,}")
        
        has_taxid = (df['taxID'] != 'NA').sum()
        logger.info(f"\nTaxonomic coverage: {has_taxid:,} / {len(df):,} ({has_taxid/len(df):.1%})")


def main():
    extractor = MetadataExtractor(config_path="config.yaml")
    metadata_df = extractor.build_metadata()
    
    if metadata_df.empty:
        logger.error("Failed to extract metadata")
        return 1
    
    extractor.save_metadata(metadata_df)
    logger.info("\nMetadata extraction complete")
    return 0


if __name__ == "__main__":
    exit(main())