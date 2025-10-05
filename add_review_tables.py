from app import db, app
from sqlalchemy import text

with app.app_context():
    with db.engine.connect() as conn:
        # Create review_queue table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id SERIAL PRIMARY KEY,
                study_id INTEGER REFERENCES studies(id),
                status VARCHAR(50) DEFAULT 'pending',
                submitted_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewer1_email VARCHAR(200),
                reviewer1_claimed_date TIMESTAMP,
                reviewer1_completed_date TIMESTAMP,
                reviewer2_email VARCHAR(200),
                reviewer2_claimed_date TIMESTAMP,
                reviewer2_completed_date TIMESTAMP,
                UNIQUE(study_id)
            )
        """))
        
        # Create review_details table for field-level reviews
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS review_details (
                id SERIAL PRIMARY KEY,
                study_id INTEGER REFERENCES studies(id),
                reviewer_number INTEGER,
                field_name VARCHAR(200),
                original_value TEXT,
                action VARCHAR(20),
                new_value TEXT,
                reviewer_note TEXT,
                reviewed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.commit()
    print("Review tables created successfully!")