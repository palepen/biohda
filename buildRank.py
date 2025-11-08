import sqlite3
import sys
from pathlib import Path
from typing import Dict

# --- CONFIGURATION ---
# This must match the DB name from the *creator* script
DB_NAME = "dataset/ncbi_taxonomy.db"
# --- END CONFIGURATION ---

def build_rank_map_from_db(db_path: Path) -> Dict[str, str]:
    """
    Generates a comprehensive {name: rank} map from the local NCBI database.
    This is the function from your prompt.
    """
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}", file=sys.stderr)
        print("Please run the `create_ncbi_db.py` script first.", file=sys.stderr)
        sys.exit(1)
        
    # These are the 8 ranks your original solution cared about
    target_ranks = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    # We only want to map official scientific names
    # This query joins the two tables we created
    # Using '?' placeholders makes the query safe and efficient
    query = f"""
    SELECT
        n.name_txt,
        t.rank
    FROM
        taxa t
    JOIN
        names n ON t.tax_id = n.tax_id
    WHERE
        t.rank IN ({','.join('?' for _ in target_ranks)})
        AND n.name_class = 'scientific name'
    """
    
    con = None
    rank_mapping = {}
    
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        print(f"Querying database for {len(target_ranks)} ranks...")
        
        for row in cur.execute(query, target_ranks):
            name, rank = row
            # This creates the exact map your function needs
            rank_mapping[name] = rank
            
    except sqlite3.Error as e:
        print(f"An error occurred during query: {e}", file=sys.stderr)
    finally:
        if con:
            con.close()
            
    print(f"Successfully built rank map with {len(rank_mapping)} entries.")
    return rank_mapping

def main():
    """
    Loads the rank map and demonstrates its contents.
    """
    print(f"Loading rank map from '{DB_NAME}'...")
    global_rank_map = build_rank_map_from_db(Path(DB_NAME))
    
    if not global_rank_map:
        print("Rank map is empty. Did the database creation succeed?")
        return
        
    print("\n--- Rank Map Demonstration ---")
    
    # Test cases that would fail a manual map
    test_names = {
        "Mammalia": "class",
        "Agaricales": "order",
        "Eukaryota": "superkingdom", # Note: NCBI often uses 'superkingdom'
        "Bacteria": "superkingdom",
        "Homo": "genus",
        "Felis": "genus",
        "Rosaceae": "family",
        "Plantae": "kingdom"
    }
    
    print(f"Total names in map: {len(global_rank_map)}")
    print("\nChecking some common examples (Your ranks may vary slightly, e.g., 'domain' vs 'superkingdom'):")
    
    for name, expected_rank in test_names.items():
        if name in global_rank_map:
            found_rank = global_rank_map[name]
            # Check if found rank matches expected, or if (domain, superkingdom) are used
            if found_rank == expected_rank:
                print(f"  [PASS] '{name}' -> '{found_rank}'")
            elif expected_rank in ('domain', 'superkingdom') and found_rank in ('domain', 'superkingdom'):
                 print(f"  [INFO] '{name}' -> '{found_rank}' (Matches expected 'domain'/'superkingdom')")
            else:
                print(f"  [WARN] '{name}' -> '{found_rank}' (Expected '{expected_rank}')")
        else:
            print(f"  [FAIL] '{name}' -> Not found in map.")
            
    print("\nThis 'global_rank_map' dictionary can now be passed to your `parse_ncbi_lineage` function.")

if __name__ == "__main__":
    main()