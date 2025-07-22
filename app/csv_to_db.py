import pandas as pd
import sqlite3

def csv_to_sqlite(csv_file, sqlite_file, table_name="data"):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Connect to SQLite database (creates it if not exists)
    conn = sqlite3.connect(sqlite_file)
    
    # Write DataFrame to SQLite table
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    
    # Close the connection
    conn.close()
    print(f"CSV file '{csv_file}' successfully converted to SQLite '{sqlite_file}' with table '{table_name}'")

# Example usage
# csv_to_sqlite("./data/ai4i2020.csv", "./data/ai4i2020.db" , 'ai4i2020')