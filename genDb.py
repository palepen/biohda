import sqlite3
import os
from pathlib import Path
import sys

# CONFIGURATION
# IMPORTANT:
# Set this to the folder path where you extracted taxdump
# Example: TAXON_DUMP_DIR = Path("C:/Users/YourUser/Downloads/taxdump")
TAXON_DUMP_DIR = Path("./taxdmp") 

# This will be the name of the output database
DB_NAME = "dataset/ncbi_taxonomy.db"

# We'll insert in batches for speed
BATCH_SIZE = 50000

def create_tables(con):
    """Creates the necessary tables in the SQLite database."""
    print("Creating tables: 'taxa' and 'names'...")
    cur = con.cursor()
    
    # Table from nodes.dmp
    # We add indexes for faster lookups
    cur.execute("""
    CREATE TABLE IF NOT EXISTS taxa (
        tax_id INTEGER PRIMARY KEY,
        parent_tax_id INTEGER,
        rank TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS taxa_parent_idx ON taxa (parent_tax_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS taxa_rank_idx ON taxa (rank);")

    # Table from names.dmp
    # We only store scientific names
    cur.execute("""
    CREATE TABLE IF NOT EXISTS names (
        tax_id INTEGER,
        name_txt TEXT,
        name_class TEXT,
        FOREIGN KEY (tax_id) REFERENCES taxa (tax_id)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS names_tax_id_idx ON names (tax_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS names_name_txt_idx ON names (name_txt);")
    
    print("Tables created successfully.")

def process_nodes_dmp(con, dump_dir):
    """Parses nodes.dmp and populates the 'taxa' table."""
    nodes_file = dump_dir / "nodes.dmp"
    if not nodes_file.exists():
        print(f"Error: {nodes_file} not found.", file=sys.stderr)
        print(f"Please check the TAXON_DUMP_DIR path in this script.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Processing {nodes_file}...")
    cur = con.cursor()
    batch = []
    total_count = 0
    
    insert_query = "INSERT INTO taxa (tax_id, parent_tax_id, rank) VALUES (?, ?, ?)"
    
    with open(nodes_file, 'r', encoding='utf-8') as f:
        for line in f:
            # File format: tax_id | parent_tax_id | rank | ...
            # Delimiter is '\t|\t'
            line = line.rstrip('\t|\n')
            cols = line.split('\t|\t')
            
            try:
                tax_id = int(cols[0])
                parent_tax_id = int(cols[1])
                rank = cols[2].strip()
                
                batch.append((tax_id, parent_tax_id, rank))
                
                if len(batch) >= BATCH_SIZE:
                    cur.executemany(insert_query, batch)
                    total_count += len(batch)
                    print(f"  ... inserted {total_count} nodes", end='\r')
                    batch = []
                    
            except (IndexError, ValueError) as e:
                print(f"\nWarning: Skipping malformed line in nodes.dmp: {line[:50]}... | Error: {e}")
        
        # Insert any remaining items
        if batch:
            cur.executemany(insert_query, batch)
            total_count += len(batch)
            print(f"  ... inserted {total_count} nodes", end='\r')

    con.commit()
    print(f"\nFinished processing nodes.dmp. Total nodes: {total_count}")

def process_names_dmp(con, dump_dir):
    """
    Parses names.dmp and populates the 'names' table.
    Only 'scientific name' entries are stored.
    """
    names_file = dump_dir / "names.dmp"
    if not names_file.exists():
        print(f"Error: {names_file} not found.", file=sys.stderr)
        print(f"Please check the TAXON_DUMP_DIR path in this script.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Processing {names_file}...")
    cur = con.cursor()
    batch = []
    total_count = 0
    
    insert_query = "INSERT INTO names (tax_id, name_txt, name_class) VALUES (?, ?, ?)"
    
    with open(names_file, 'r', encoding='utf-8') as f:
        for line in f:
            # File format: tax_id | name_txt | unique_name | name_class | ...
            line = line.rstrip('\t|\n')
            cols = line.split('\t|\t')
            
            try:
                name_class = cols[3].strip()
                
                # Only store the names we care about.
                if name_class == 'scientific name':
                    tax_id = int(cols[0])
                    name_txt = cols[1].strip()
                    
                    batch.append((tax_id, name_txt, name_class))
                    
                    if len(batch) >= BATCH_SIZE:
                        cur.executemany(insert_query, batch)
                        total_count += len(batch)
                        batch = []
                        print(f"  ... inserted {total_count} scientific names", end='\r')
                        
            except (IndexError, ValueError) as e:
                print(f"\nWarning: Skipping malformed line in names.dmp: {line[:50]}... | Error: {e}")
        
        # Insert any remaining items
        if batch:
            cur.executemany(insert_query, batch)
            total_count += len(batch)
            print(f"  ... inserted {total_count} scientific names", end='\r')

    con.commit()
    print(f"\nFinished processing names.dmp. Total scientific names: {total_count}")

def main():
    """Main function to create and populate the database."""
    
    # Check if taxdump dir exists
    if not TAXON_DUMP_DIR.is_dir():
        print(f"Error: Directory not found: {TAXON_DUMP_DIR}", file=sys.stderr)
        print("Please download and extract taxdump.tar.gz from NCBI", file=sys.stderr)
        print("and set the TAXON_DUMP_DIR variable in this script.", file=sys.stderr)
        sys.exit(1)

    db_path = Path(DB_NAME)
    # Delete old DB file if it exists, to start fresh
    if db_path.exists():
        print(f"Removing old database: {db_path}")
        os.remove(db_path)
        
    con = None
    try:
        con = sqlite3.connect(db_path)
        
        create_tables(con)
        process_nodes_dmp(con, TAXON_DUMP_DIR)
        process_names_dmp(con, TAXON_DUMP_DIR)
        
        print("\n---------------------------------")
        print(f"Success! Database '{DB_NAME}' created.")
        print("You can now use this file with the `build_rank_map_from_db` function.")
        print("---------------------------------")
        
    except sqlite3.Error as e:
        print(f"\nAn SQLite error occurred: {e}", file=sys.stderr)
        
    finally:
        if con:
            con.close()

if __name__ == "__main__":
    main()