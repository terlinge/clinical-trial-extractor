#!/usr/bin/env python3
"""
Database Migration Script for Multi-Source Data Architecture
============================================================

This script creates the new tables needed for multi-source data storage
and user selection functionality.

Run this script to update your database schema to support the enhanced
clinical trial extractor v2.2 features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db, ExtractionSource, DataElement, UserPreferences, ExtractionConflict
from sqlalchemy import text

def create_tables():
    """Create the new multi-source data tables"""
    
    print("🔧 Creating multi-source data tables...")
    
    with app.app_context():
        try:
            # Create all new tables
            db.create_all()
            
            # Verify tables were created
            inspector = db.inspect(db.engine)
            existing_tables = inspector.get_table_names()
            
            new_tables = [
                'extraction_sources',
                'data_elements', 
                'user_preferences',
                'extraction_conflicts'
            ]
            
            created_tables = []
            for table in new_tables:
                if table in existing_tables:
                    created_tables.append(table)
                    print(f"✅ {table}")
                else:
                    print(f"❌ {table} - FAILED TO CREATE")
            
            print(f"\n📊 Summary: {len(created_tables)}/{len(new_tables)} tables created successfully")
            
            # Create default user preferences
            existing_prefs = UserPreferences.query.first()
            if not existing_prefs:
                default_prefs = UserPreferences(
                    user_id='default',
                    preferred_source_order=['pdfplumber', 'heuristic', 'llm', 'ocr'],
                    auto_select_rules={
                        'use_highest_confidence': True,
                        'require_minimum_confidence': 0.6,
                        'prefer_table_data': True,
                        'flag_conflicts_above_threshold': 0.3
                    },
                    confidence_threshold=0.7,
                    require_manual_review=True,
                    highlight_conflicts=True,
                    show_all_sources=True
                )
                db.session.add(default_prefs)
                db.session.commit()
                print("✅ Default user preferences created")
            else:
                print("ℹ️ User preferences already exist")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            import traceback
            traceback.print_exc()
            return False

def migrate_existing_studies():
    """Migrate existing studies to new multi-source format"""
    
    print("\n🔄 Migrating existing studies...")
    
    with app.app_context():
        try:
            from app import Study
            studies = Study.query.all()
            
            print(f"Found {len(studies)} existing studies")
            
            for study in studies:
                # Check if already migrated
                existing_sources = ExtractionSource.query.filter_by(study_id=study.id).first()
                if existing_sources:
                    continue
                
                # Create a basic extraction source entry for legacy data
                legacy_source = ExtractionSource(
                    study_id=study.id,
                    extraction_method='legacy',
                    source_version='1.0',
                    raw_data={
                        'migrated_from_legacy': True,
                        'extraction_date': study.extraction_date.isoformat() if study.extraction_date else None
                    },
                    confidence_score=0.8,
                    quality_metrics={'legacy_migration': True},
                    processing_time=0.0,
                    success_status=True
                )
                db.session.add(legacy_source)
                
                # Create basic data elements for key fields
                if study.title:
                    title_element = DataElement(
                        study_id=study.id,
                        element_type='identification',
                        element_name='title',
                        element_path='study_identification.title',
                        llm_value=study.title,
                        llm_confidence=0.8,
                        llm_source_location='Legacy extraction',
                        selected_source='llm',
                        selected_value=study.title,
                        is_validated=True,
                        needs_review=False,
                        conflicting_sources=False
                    )
                    db.session.add(title_element)
            
            db.session.commit()
            print(f"✅ Migrated {len(studies)} studies to new format")
            return True
            
        except Exception as e:
            print(f"❌ Error migrating studies: {e}")
            db.session.rollback()
            return False

def verify_migration():
    """Verify the migration was successful"""
    
    print("\n🔍 Verifying migration...")
    
    with app.app_context():
        try:
            # Check table existence and basic functionality
            source_count = ExtractionSource.query.count()
            element_count = DataElement.query.count()
            prefs_count = UserPreferences.query.count()
            conflict_count = ExtractionConflict.query.count()
            
            print(f"📊 Database Status:")
            print(f"   - Extraction Sources: {source_count}")
            print(f"   - Data Elements: {element_count}")
            print(f"   - User Preferences: {prefs_count}")
            print(f"   - Extraction Conflicts: {conflict_count}")
            
            # Test basic queries
            if source_count > 0:
                latest_source = ExtractionSource.query.order_by(ExtractionSource.id.desc()).first()
                print(f"   - Latest source: {latest_source.extraction_method} for study {latest_source.study_id}")
            
            print("✅ Migration verification complete")
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False

def main():
    """Run the complete migration process"""
    
    print("🚀 Starting Clinical Trial Extractor v2.2 Database Migration")
    print("=" * 60)
    
    # Step 1: Create new tables
    if not create_tables():
        print("❌ Migration failed at table creation")
        return False
    
    # Step 2: Migrate existing data
    if not migrate_existing_studies():
        print("❌ Migration failed at data migration")
        return False
    
    # Step 3: Verify migration
    if not verify_migration():
        print("❌ Migration failed at verification")
        return False
    
    print("\n🎉 Migration completed successfully!")
    print("\nNext steps:")
    print("1. Restart your application")
    print("2. Upload a PDF to test the new multi-source interface")
    print("3. Use the 'Review & Select Data' tab to manage extraction sources")
    
    return True

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)