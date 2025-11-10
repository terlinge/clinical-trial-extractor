# Database Schema Enhancement for Multi-Source Data Storage and User Selection

"""
MULTI-SOURCE DATA ARCHITECTURE
===============================

Purpose: Store extraction results from all sources (PDFPlumber, OCR, LLM, Heuristic)
and allow users to select which values to use in final outputs.

Key Features:
1. Store raw extraction data from each method
2. Track confidence/quality scores for each extraction
3. Allow user preferences for data source selection
4. Maintain audit trail of extraction sources
5. Support systematic review workflows with source transparency
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

# NEW TABLES TO ADD TO app.py

class ExtractionSource(db.Model):
    """Store raw extraction data from each method"""
    __tablename__ = 'extraction_sources'
    
    id = db.Column(db.Integer, primary_key=True)
    study_id = db.Column(db.Integer, db.ForeignKey('studies.id'), nullable=False)
    
    # Source identification
    extraction_method = db.Column(db.String(50), nullable=False)  # 'pdfplumber', 'ocr', 'llm', 'heuristic'
    source_version = db.Column(db.String(20))  # Track version/model used
    extraction_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Raw extracted data
    raw_data = db.Column(db.JSON)  # Complete extraction result
    confidence_score = db.Column(db.Float)  # 0.0 - 1.0 confidence
    quality_metrics = db.Column(db.JSON)  # Method-specific quality indicators
    
    # Processing metadata
    processing_time = db.Column(db.Float)  # Seconds
    error_messages = db.Column(db.JSON)  # Any warnings/errors
    success_status = db.Column(db.Boolean, default=True)
    
    # Index for efficient querying
    __table_args__ = (
        db.Index('idx_study_method', 'study_id', 'extraction_method'),
    )

class DataElement(db.Model):
    """Individual data elements with source attribution"""
    __tablename__ = 'data_elements'
    
    id = db.Column(db.Integer, primary_key=True)
    study_id = db.Column(db.Integer, db.ForeignKey('studies.id'), nullable=False)
    
    # Data identification
    element_type = db.Column(db.String(50), nullable=False)  # 'outcome', 'intervention', 'demographic', 'design'
    element_name = db.Column(db.String(200))  # Specific field name
    element_path = db.Column(db.String(500))  # JSON path for nested data
    
    # Values from different sources
    pdfplumber_value = db.Column(db.Text)
    pdfplumber_confidence = db.Column(db.Float)
    pdfplumber_source_location = db.Column(db.String(200))  # "Table 2, page 5"
    
    ocr_value = db.Column(db.Text)
    ocr_confidence = db.Column(db.Float)
    ocr_source_location = db.Column(db.String(200))
    
    llm_value = db.Column(db.Text)
    llm_confidence = db.Column(db.Float)
    llm_source_location = db.Column(db.String(200))
    
    heuristic_value = db.Column(db.Text)
    heuristic_confidence = db.Column(db.Float)
    heuristic_source_location = db.Column(db.String(200))
    
    # User selection
    selected_source = db.Column(db.String(50))  # Which source user chose
    selected_value = db.Column(db.Text)  # Final value after user selection
    user_notes = db.Column(db.Text)  # User annotations
    selection_timestamp = db.Column(db.DateTime)
    
    # Validation flags
    is_validated = db.Column(db.Boolean, default=False)
    needs_review = db.Column(db.Boolean, default=False)
    conflicting_sources = db.Column(db.Boolean, default=False)
    
    # Indexes
    __table_args__ = (
        db.Index('idx_study_element_type', 'study_id', 'element_type'),
        db.Index('idx_needs_review', 'needs_review'),
        db.Index('idx_conflicting', 'conflicting_sources'),
    )

class UserPreferences(db.Model):
    """Store user preferences for data source selection"""
    __tablename__ = 'user_preferences'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100))  # Future user system integration
    
    # Global preferences
    preferred_source_order = db.Column(db.JSON)  # ["pdfplumber", "heuristic", "llm", "ocr"]
    auto_select_rules = db.Column(db.JSON)  # Automatic selection rules
    
    # Element-specific preferences
    element_type = db.Column(db.String(50))  # Optional: specific to element type
    confidence_threshold = db.Column(db.Float, default=0.7)  # Minimum confidence for auto-selection
    
    # Systematic review settings
    require_manual_review = db.Column(db.Boolean, default=True)  # Always require user validation
    highlight_conflicts = db.Column(db.Boolean, default=True)  # Show conflicting values
    show_all_sources = db.Column(db.Boolean, default=True)  # Display all extraction results
    
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, default=datetime.utcnow)

class ExtractionConflict(db.Model):
    """Track conflicts between different extraction methods"""
    __tablename__ = 'extraction_conflicts'
    
    id = db.Column(db.Integer, primary_key=True)
    study_id = db.Column(db.Integer, db.ForeignKey('studies.id'), nullable=False)
    data_element_id = db.Column(db.Integer, db.ForeignKey('data_elements.id'), nullable=False)
    
    # Conflict details
    conflict_type = db.Column(db.String(50))  # 'value_mismatch', 'missing_data', 'quality_concern'
    severity = db.Column(db.String(20))  # 'low', 'medium', 'high'
    
    # Conflicting values
    source_a = db.Column(db.String(50))
    value_a = db.Column(db.Text)
    confidence_a = db.Column(db.Float)
    
    source_b = db.Column(db.String(50))
    value_b = db.Column(db.Text)
    confidence_b = db.Column(db.Float)
    
    # Resolution
    resolution_status = db.Column(db.String(50), default='pending')  # 'pending', 'resolved', 'escalated'
    resolution_method = db.Column(db.String(100))  # How conflict was resolved
    resolution_notes = db.Column(db.Text)
    resolved_by = db.Column(db.String(100))
    resolved_date = db.Column(db.DateTime)
    
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        db.Index('idx_study_conflicts', 'study_id', 'resolution_status'),
        db.Index('idx_severity', 'severity'),
    )

# ENHANCED SOURCE ATTRIBUTION ICONS
SOURCE_ICONS = {
    'pdfplumber': '📊',  # Table/structured data
    'ocr': '👁️',        # Visual/image extraction  
    'llm': '🤖',        # AI interpretation
    'heuristic': '🔍',  # Pattern matching
    'calculated': '🧮', # Derived/calculated
    'manual': '✏️',     # User input
    'validated': '✅',  # User validated
    'conflict': '⚠️',   # Conflicting sources
    'missing': '❌',    # Not found
    'uncertain': '❓'   # Low confidence
}

SOURCE_PRIORITY_DEFAULT = [
    'pdfplumber',  # Structured table data (highest priority)
    'heuristic',   # Pattern matching from text
    'ocr',         # Visual extraction
    'llm'          # AI interpretation (lowest priority for conflicts)
]

# QUALITY THRESHOLDS
CONFIDENCE_THRESHOLDS = {
    'high': 0.8,
    'medium': 0.6,
    'low': 0.4
}

AUTO_SELECT_RULES = {
    'use_highest_confidence': True,
    'require_minimum_confidence': 0.6,
    'prefer_table_data': True,
    'flag_conflicts_above_threshold': 0.3  # Flag if sources differ significantly
}