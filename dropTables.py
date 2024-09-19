import sqlite3

# Connexion à la base de données
conn = sqlite3.connect('chrom.sqlite3')
cursor = conn.cursor()

# Obtenir les noms de toutes les tables dans la base de données
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Supprimer chaque table
for table in tables:
    cursor.execute(f"DROP TABLE IF EXISTS {table[0]};")
    print(f"Dropped table {table[0]}")

# Valider les changements et fermer la connexion
conn.commit()
conn.close()

print("All tables dropped successfully.")
