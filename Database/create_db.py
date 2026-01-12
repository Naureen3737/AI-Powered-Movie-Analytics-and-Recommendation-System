import pandas as pd
from sqlalchemy import create_engine
import sqlite3

# 1. Load Excel
df = pd.read_excel("Movie_Detail_filled.xlsx")

# 2. See actual column names
print("Original columns in dataset:")
print(df.columns.tolist())

# 3. Create SQLite database
engine = create_engine("sqlite:///movies.db")

# Save DataFrame to SQL using the same column names
df.to_sql("movies", engine, if_exists="replace", index=False)

print("\nDatabase 'movies.db' created successfully with table 'movies'!")

# 4. Optional: Check how the table looks
conn = sqlite3.connect("movies.db")
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("\nTables in database:", tables)

# Show columns in 'movies' table
cursor.execute("PRAGMA table_info(movies);")
columns = cursor.fetchall()
print("\nColumns in 'movies' table:")
for col in columns:
    print(col[1])  # col[1] is the column name

# Show first 5 rows of data
cursor.execute("SELECT * FROM movies LIMIT 5;")
rows = cursor.fetchall()
print("\nFirst 5 rows in 'movies' table:")
for row in rows:
    print(row)

conn.close()
