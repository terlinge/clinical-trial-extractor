"""
Fix column size limitations in outcome_timepoints table
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Get database connection string
db_url = os.getenv('DATABASE_URL', 'postgresql://localhost/clinical_trials')

print(f"Connecting to database: {db_url}")

try:
    # Use the full connection string
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("Fixing Column Size Limitations")
    print("="*60 + "\n")
    
    # Fix effect_measure column (50 -> 200)
    print("📝 Increasing effect_measure column size from 50 to 200...")
    cursor.execute("""
        ALTER TABLE outcome_timepoints 
        ALTER COLUMN effect_measure TYPE VARCHAR(200);
    """)
    print("✅ effect_measure updated")
    
    # Fix p_value_text column (50 -> 200)
    print("\n📝 Increasing p_value_text column size from 50 to 200...")
    cursor.execute("""
        ALTER TABLE outcome_timepoints 
        ALTER COLUMN p_value_text TYPE VARCHAR(200);
    """)
    print("✅ p_value_text updated")
    
    print("\n" + "="*60)
    print("✅ Column sizes updated successfully!")
    print("="*60 + "\n")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
