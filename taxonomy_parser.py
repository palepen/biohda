"""
taxonomy_parser.py
==================
Shared taxonomy parsing utilities for consistent lineage interpretation
across cluster_annotation.py and evaluation.py

This ensures Ground Truth and Predictions use IDENTICAL parsing logic.
"""

from typing import Dict, List
import pandas as pd


def parse_ncbi_lineage(lineage_string: str, species_name: str = None) -> Dict[str, str]:
    """
    Parse NCBI-style semicolon-delimited lineage into standard taxonomic levels.
    
    This is the SINGLE SOURCE OF TRUTH for parsing taxonomy in this pipeline.
    Both cluster_annotation.py and evaluation.py MUST use this function.
    
    Args:
        lineage_string: Semicolon-separated lineage from TSV
                       Example: "cellular organisms;Eukaryota;Sar;Alveolata;Ciliophora;..."
        species_name: Optional species name from TSV (column 3)
                     Example: "Apokeronopsis ovalis"
    
    Returns:
        Dictionary with keys: domain, kingdom, phylum, class, order, family, genus, species
    """
    tax_levels = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    parsed = {level: None for level in tax_levels}
    
    # Handle missing/invalid lineage
    if pd.isna(lineage_string) or lineage_string == '' or lineage_string == 'N/A':
        return parsed
    
    # Split and clean
    parts = [p.strip() for p in lineage_string.split(';') if p.strip()]
    
    if len(parts) == 0:
        return parsed
    
    # Remove "cellular organisms" if present (not informative)
    if parts[0] == 'cellular organisms':
        parts = parts[1:]
    
    if len(parts) == 0:
        return parsed
    
    # =========================================================================
    # RANK-AWARE MAPPING
    # =========================================================================
    # Create a comprehensive mapping of known taxa to their ranks
    # This is based on common NCBI taxonomy patterns
    
    rank_mapping = {
        # Domains
        'Eukaryota': 'domain',
        'Bacteria': 'domain',
        'Archaea': 'domain',
        'Viruses': 'domain',
        
        # Kingdoms (Eukaryotic)
        'Metazoa': 'kingdom',
        'Viridiplantae': 'kingdom',
        'Fungi': 'kingdom',
        'Chromista': 'kingdom',
        'Protozoa': 'kingdom',
        'Opisthokonta': 'kingdom',  # Sometimes used as kingdom
        
        # Major Phyla
        'Chordata': 'phylum',
        'Arthropoda': 'phylum',
        'Mollusca': 'phylum',
        'Annelida': 'phylum',
        'Platyhelminthes': 'phylum',
        'Nematoda': 'phylum',
        'Cnidaria': 'phylum',
        'Echinodermata': 'phylum',
        'Porifera': 'phylum',
        'Bryozoa': 'phylum',
        'Brachiopoda': 'phylum',
        'Ascomycota': 'phylum',
        'Basidiomycota': 'phylum',
        'Chlorophyta': 'phylum',
        'Rhodophyta': 'phylum',
        'Bacillariophyta': 'phylum',
        'Ciliophora': 'phylum',
        'Apicomplexa': 'phylum',
        'Streptophyta': 'phylum',
        'Sar': 'phylum',  # Supergroup, treat as phylum
        'Alveolata': 'phylum',
        'Stramenopiles': 'phylum',
        'Rhizaria': 'phylum',
        'Amoebozoa': 'phylum',
        'Excavata': 'phylum',
        
        # Major Classes
        'Mammalia': 'class',
        'Aves': 'class',
        'Reptilia': 'class',
        'Amphibia': 'class',
        'Actinopterygii': 'class',
        'Insecta': 'class',
        'Arachnida': 'class',
        'Malacostraca': 'class',
        'Gastropoda': 'class',
        'Bivalvia': 'class',
        'Cephalopoda': 'class',
        'Polychaeta': 'class',
        'Clitellata': 'class',
        'Turbellaria': 'class',
        'Trematoda': 'class',
        'Cestoda': 'class',
        'Agaricomycetes': 'class',
        'Sordariomycetes': 'class',
        'Dothideomycetes': 'class',
        'Ulvophyceae': 'class',
        'Florideophyceae': 'class',
        'Spirotrichea': 'class',
        'Oligohymenophorea': 'class',
        'Intramacronucleata': 'class',
        
        # Major Orders
        'Primates': 'order',
        'Carnivora': 'order',
        'Rodentia': 'order',
        'Artiodactyla': 'order',
        'Chiroptera': 'order',
        'Coleoptera': 'order',
        'Lepidoptera': 'order',
        'Diptera': 'order',
        'Hymenoptera': 'order',
        'Hemiptera': 'order',
        'Passeriformes': 'order',
        'Accipitriformes': 'order',
        'Decapoda': 'order',
        'Amphipoda': 'order',
        'Isopoda': 'order',
        'Neogastropoda': 'order',
        'Stylommatophora': 'order',
        'Veneroida': 'order',
        'Mytiloida': 'order',
        'Agaricales': 'order',
        'Polyporales': 'order',
        'Hypocreales': 'order',
        'Urostylida': 'order',
        'Euplotida': 'order',
        
        # Major Families (add common ones)
        'Hominidae': 'family',
        'Felidae': 'family',
        'Canidae': 'family',
        'Muridae': 'family',
        'Bovidae': 'family',
        'Formicidae': 'family',
        'Scarabaeidae': 'family',
        'Nymphalidae': 'family',
        'Culicidae': 'family',
        'Drosophilidae': 'family',
        'Apidae': 'family',
        'Corvidae': 'family',
        'Accipitridae': 'family',
        'Portunidae': 'family',
        'Gammaridae': 'family',
        'Littorinidae': 'family',
        'Mytilidae': 'family',
        'Veneridae': 'family',
        'Agaricaceae': 'family',
        'Polyporaceae': 'family',
        'Nectriaceae': 'family',
        'Pseudokeronopsidae': 'family',
        'Euplotidae': 'family',
    }
    
    # =========================================================================
    # STEP 1: Map known taxa using rank_mapping
    # =========================================================================
    for part in parts:
        if part in rank_mapping:
            rank = rank_mapping[part]
            if parsed[rank] is None:  # Don't overwrite if already set
                parsed[rank] = part
    
    # =========================================================================
    # STEP 2: Handle Genus and Species (always last 1-2 elements)
    # =========================================================================
    if len(parts) >= 2:
        # Check if last part looks like binomial species name
        last_part = parts[-1]
        second_last_part = parts[-2]
        
        # Pattern: "Genus species" or "Genus species subspecies"
        if ' ' in last_part:
            # Last part is binomial (e.g., "Homo sapiens")
            parsed['species'] = last_part
            genus_from_species = last_part.split()[0]
            
            # Set genus from species name
            if parsed['genus'] is None:
                parsed['genus'] = genus_from_species
        else:
            # Last part is likely genus, second last might be family
            if parsed['genus'] is None:
                parsed['genus'] = parts[-1]
            
            # If there's a species_name provided separately, use it
            if species_name and species_name != 'N/A' and pd.notna(species_name):
                parsed['species'] = species_name
                # Also extract genus from species_name if not set
                if parsed['genus'] is None and ' ' in species_name:
                    parsed['genus'] = species_name.split()[0]
    
    elif len(parts) == 1:
        # Only one part - could be genus or domain
        if parts[0] not in rank_mapping:
            # Assume it's genus if not in our known list
            parsed['genus'] = parts[0]
    
    # =========================================================================
    # STEP 3: Use species_name from TSV if provided (highest priority)
    # =========================================================================
    if species_name and species_name != 'N/A' and pd.notna(species_name):
        parsed['species'] = species_name
        # Extract genus from species name
        if ' ' in species_name:
            genus_from_species = species_name.split()[0]
            # Only overwrite genus if it wasn't already set or if it matches
            if parsed['genus'] is None or parsed['genus'] == genus_from_species:
                parsed['genus'] = genus_from_species
    
    # =========================================================================
    # STEP 4: Fallback - fill gaps using positional heuristics
    # =========================================================================
    # This only applies if rank_mapping didn't catch everything
    
    # If we have domain but no kingdom, try to infer
    if parsed['domain'] and not parsed['kingdom']:
        # For Eukaryota, kingdom is often the next major group
        if parsed['domain'] == 'Eukaryota' and len(parts) >= 2:
            # Check if second element is a known kingdom-level group
            candidate = parts[1] if len(parts) > 1 else None
            if candidate and candidate not in rank_mapping:
                # Only set if it's not already mapped to another rank
                if candidate not in parsed.values():
                    parsed['kingdom'] = candidate
    
    return parsed


def normalize_taxonomy_for_comparison(tax_string: str) -> str:
    """
    Normalize taxonomy string for case-insensitive comparison.
    
    Args:
        tax_string: Raw taxonomy string
    
    Returns:
        Normalized string (lowercase, stripped)
    """
    if pd.isna(tax_string) or tax_string == '':
        return None
    return str(tax_string).lower().strip()


def get_taxonomy_levels() -> List[str]:
    """Return standard taxonomic levels used in this pipeline"""
    return ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']