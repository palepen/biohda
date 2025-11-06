"""
Step 3: Metadata Extraction (Regenerated)
=========================================
Extract taxonomic information from cleaned FASTA file HEADERS.

This version is fixed to work with BLAST databases that lack
a complete taxonomy.sqlite3 file. It extracts the
seqID, taxID, and scientific_name directly from the FASTA header.

It no longer attempts to connect to the .sqlite3 database.
"""

import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from Bio import SeqIO
import pandas as pd
import re
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract taxonomic metadata from FASTA headers"""
    
    def __init__(self, 
                 fasta_dir: str = "dataset/processed",
                 output_dir: str = "dataset"):
        """
        Initialize metadata extractor
        
        Args:
            fasta_dir: Directory with cleaned FASTA files
            output_dir: Directory to save metadata.csv
        """
        self.fasta_dir = Path(fasta_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # FASTA files
        self.fasta_files = {
            'ITS': 'ITS_clean.fasta',
            'LSU': 'LSU_clean.fasta',
            'SSU': 'SSU_clean.fasta'
        }
    
    def parse_fasta_header(self, header: str) -> Tuple[str, Optional[str], str]:
        """
        Parse FASTA header to extract seqID, taxID, and scientific_name
        
        Assumes NCBI FASTA header format like:
        >NC_001234.1 Homo sapiens [taxid:9606]
        or
        >gi|123456|ref|NC_001234.1| Homo sapiens [taxid:9606]
        
        Args:
            header: FASTA header line (without '>')
        
        Returns:
            (seqID, taxID, scientific_name) tuple
        """
        header = header.lstrip('>')
        
        # 1. Extract taxID
        taxid_match = re.search(r'\[taxid:(\d+)\]', header)
        taxid = taxid_match.group(1) if taxid_match else None
        
        # 2. Extract seqID
        # Remove the taxid part to avoid confusion
        header_no_taxid = re.sub(r'\[taxid:\d+\]', '', header).strip()
        
        seqid_parts = header_no_taxid.split()[0].split('|')
        if len(seqid_parts) > 1:
            # Handle gi|xxx|ref|ACCESSION format
            seqid = seqid_parts[-1]
        else:
            seqid = seqid_parts[0]
            
        # 3. Extract scientific_name
        # It's usually the text after the seqID and before the taxid
        name = 'NA'
        try:
            # Get everything after the first token (seqID)
            name_part = header_no_taxid.split(' ', 1)[1].strip()
            # Clean up common organism names
            name = re.sub(r'\[.*?\]', '', name_part).strip() # Remove any other [brackets]
            name = re.sub(r' mitochondrion.*', '', name, flags=re.IGNORECASE).strip()
            name = re.sub(r' complete genome.*', '', name, flags=re.IGNORECASE).strip()
            name = re.sub(r' SSU ITS1 5.8S ITS2 LSU.*', '', name, flags=re.IGNORECASE).strip()
            
            if not name:
                name = 'NA'
        except IndexError:
            # No scientific name found after seqID
            name = 'NA'
        
        return seqid, taxid, name
    
    def extract_seqids_from_fasta(self, fasta_file: Path, marker: str) -> List[Dict]:
        """
        Extract all seqIDs, taxIDs, and names from a FASTA file
        
        Args:
            fasta_file: Path to FASTA file
            marker: Marker name (ITS, LSU, SSU)
        
        Returns:
            List of metadata dictionaries
        """
        logger.info(f"Extracting metadata from {fasta_file.name}...")
        
        records = []
        count = 0
        taxid_found = 0
        name_found = 0
        
        try:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                seqid, taxid, name = self.parse_fasta_header(record.description)
                
                records.append({
                    'seqID': seqid,
                    'marker': marker,
                    'taxID': taxid if taxid else 'NA',
                    'scientific_name': name if name else 'NA',
                    'rank': 'NA',  # Cannot get rank from header
                    'lineage': 'NA' # Cannot get lineage from header
                })
                
                count += 1
                if taxid:
                    taxid_found += 1
                if name != 'NA':
                    name_found += 1
                
                if count % 100000 == 0:
                    logger.info(f"  Processed {count:,} sequences...")
            
            logger.info(f"  Total sequences: {count:,}")
            logger.info(f"  TaxIDs found in headers: {taxid_found:,} ({taxid_found/count:.1%})")
            logger.info(f"  Names found in headers: {name_found:,} ({name_found/count:.1%})")
            
        except FileNotFoundError:
            logger.error(f"File not found: {fasta_file}")
        except Exception as e:
            logger.error(f"Error reading {fasta_file}: {e}")
        
        return records
    
    def build_complete_metadata(self) -> pd.DataFrame:
        """
        Build complete metadata for all markers
        
        Returns:
            Combined DataFrame with all metadata
        """
        all_records = []
        
        for marker, fasta_name in self.fasta_files.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing marker: {marker}")
            logger.info(f"{'='*60}")
            
            fasta_file = self.fasta_dir / fasta_name
            if not fasta_file.exists():
                logger.warning(f"FASTA file not found, skipping: {fasta_file}")
                continue

            records = self.extract_seqids_from_fasta(fasta_file, marker)
            if records:
                all_records.extend(records)
        
        if not all_records:
            logger.error("No metadata extracted from any marker")
            return pd.DataFrame()
        
        # Combine all markers
        combined_df = pd.DataFrame(all_records)
        
        return combined_df
    
    def save_metadata(self, df: pd.DataFrame, output_file: str = "metadata.csv"):
        """
        Save metadata to CSV
        
        Args:
            df: Metadata DataFrame
            output_file: Output filename
        """
        output_path = self.output_dir / output_file
        
        try:
            # Ensure 'NA' is saved as a string, not left empty
            df = df.fillna('NA')
            df.to_csv(output_path, index=False)
            logger.info(f"\n✓ Metadata saved to: {output_path}")
            logger.info(f"  Total records: {len(df):,}")
            logger.info(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")
            
            # Summary statistics
            logger.info(f"\n{'='*60}")
            logger.info("METADATA SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Records by marker:")
            for marker in df['marker'].unique():
                count = len(df[df['marker'] == marker])
                logger.info(f"  {marker}: {count:,}")
            
            logger.info(f"\nTaxonomic coverage:")
            has_taxid = (df['taxID'] != 'NA').sum()
            has_name = (df['scientific_name'] != 'NA').sum()
            logger.info(f"  Has taxID: {has_taxid:,} ({has_taxid/len(df):.1%})")
            logger.info(f"  Has scientific name: {has_name:,} ({has_name/len(df):.1%})")
            logger.info(f"  (Rank and Lineage are expected to be 'NA')")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")


def main():
    """Main execution"""
    
    # Initialize extractor
    extractor = MetadataExtractor(
        fasta_dir="dataset/processed",
        output_dir="dataset"
    )
    
    # Build complete metadata
    logger.info("Starting metadata extraction from FASTA headers...")
    metadata_df = extractor.build_complete_metadata()
    
    if metadata_df.empty:
        logger.error("Failed to extract metadata")
        return 1
    
    # Save to CSV
    extractor.save_metadata(metadata_df)
    
    logger.info("\n✓ Metadata extraction complete!")
    return 0


if __name__ == "__main__":
    exit(main())