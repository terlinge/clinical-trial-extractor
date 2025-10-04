"""
Database backup script for Clinical Trial Extractor
Run this regularly to backup your extracted data
"""
import os
from datetime import datetime
import subprocess

# Configuration
DB_USER = "postgres"
DB_NAME = "postgres"
BACKUP_DIR = "database_backups"

# Create backup directory if it doesn't exist
os.makedirs(BACKUP_DIR, exist_ok=True)

# Generate filename with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_file = os.path.join(BACKUP_DIR, f'clinical_trials_backup_{timestamp}.sql')

print(f"Starting database backup...")
print(f"Backup file: {backup_file}")

# Run pg_dump
try:
    # You'll be prompted for password
    subprocess.run([
        'pg_dump',
        '-U', DB_USER,
        '-d', DB_NAME,
        '-f', backup_file
    ], check=True)
    
    print(f"✓ Backup completed successfully!")
    print(f"  File size: {os.path.getsize(backup_file) / 1024:.2f} KB")
    
    # Keep only last 10 backups
    backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.endswith('.sql')])
    if len(backups) > 10:
        for old_backup in backups[:-10]:
            os.remove(os.path.join(BACKUP_DIR, old_backup))
            print(f"  Removed old backup: {old_backup}")
            
except subprocess.CalledProcessError as e:
    print(f"✗ Backup failed: {e}")
except Exception as e:
    print(f"✗ Error: {e}")