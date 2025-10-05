"""
Celery tasks for async processing
"""
from celery import Celery, Task
from flask import Flask
import os
import tempfile
from typing import Dict, Any
from datetime import datetime

def make_celery(app: Flask) -> Celery:
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    
    class ContextTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    celery.conf.update(app.config)
    return celery

# This will be initialized in app.py
celery = None

def init_celery(app: Flask):
    global celery
    celery = make_celery(app)
    return celery

@celery.task(bind=True, name='extract_trial_data_async')
def extract_trial_data_async(self, study_id: int, pdf_path: str) -> Dict[str, Any]:
    """
    Async task for extracting trial data
    """
    from extractors.ensemble_extractor import EnsembleExtractor
    from models.database import db, Study
    from services.validation_service import DataQualityValidator
    
    try:
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={
                'current': 10,
                'total': 100,
                'status': 'Initializing extraction...'
            }
        )
        
        # Initialize extractor
        extractor = EnsembleExtractor()
        
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={
                'current': 20,
                'total': 100,
                'status': 'Extracting text and tables from PDF...'
            }
        )
        
        # Extract data
        extracted_data, metadata = extractor.extract_comprehensive(pdf_path)
        
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={
                'current': 80,
                'total': 100,
                'status': 'Validating extracted data...'
            }
        )
        
        # Validate data
        validator = DataQualityValidator()
        validation_results = validator.validate_extraction(extracted_data, pdf_path)
        
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={
                'current': 90,
                'total': 100,
                'status': 'Saving to database...'
            }
        )
        
        # Update study in database
        study = Study.query.get(study_id)
        if study:
            # Update study with extracted data
            study.title = extracted_data.get('study_identification', {}).get('title')
            study.extraction_completed = True
            study.extraction_date = datetime.utcnow()
            study.extraction_metadata = metadata
            study.validation_results = validation_results
            study.confidence_scores = metadata.get('confidence_scores')
            
            # Save related data (interventions, outcomes, etc.)
            # ... (implement based on your save_to_database function)
            
            db.session.commit()
        
        # Clean up temp file if exists
        if os.path.exists(pdf_path) and 'tmp' in pdf_path:
            os.unlink(pdf_path)
        
        return {
            'status': 'success',
            'study_id': study_id,
            'validation_results': validation_results,
            'confidence_scores': metadata.get('confidence_scores', {})
        }
        
    except Exception as e:
        # Log error
        print(f"Extraction failed for study {study_id}: {str(e)}")
        
        # Update study status
        try:
            study = Study.query.get(study_id)
            if study:
                study.extraction_failed = True
                study.extraction_error = str(e)
                db.session.commit()
        except:
            pass
        
        # Clean up temp file if exists
        if os.path.exists(pdf_path) and 'tmp' in pdf_path:
            try:
                os.unlink(pdf_path)
            except:
                pass
        
        raise

@celery.task(name='cleanup_old_pdfs')
def cleanup_old_pdfs():
    """
    Periodic task to clean up old temporary PDFs
    """
    import glob
    from datetime import datetime, timedelta
    
    # Clean up temp files older than 24 hours
    temp_dir = tempfile.gettempdir()
    pattern = os.path.join(temp_dir, '*.pdf')
    
    for filepath in glob.glob(pattern):
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if datetime.now() - file_time > timedelta(hours=24):
                os.unlink(filepath)
                print(f"Cleaned up old temp file: {filepath}")
        except:
            pass