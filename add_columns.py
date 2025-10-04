from app import app, db
from sqlalchemy import text

with app.app_context():
    try:
        # Add pdf_blob column
        db.session.execute(text("ALTER TABLE studies ADD COLUMN IF NOT EXISTS pdf_blob BYTEA"))
        print("✓ Added pdf_blob column")
        
        # Add pdf_filename column
        db.session.execute(text("ALTER TABLE studies ADD COLUMN IF NOT EXISTS pdf_filename VARCHAR(500)"))
        print("✓ Added pdf_filename column")
        
        db.session.commit()
        print("\n✓ Migration complete!")
        
    except Exception as e:
        print(f"Error: {e}")