"""
Clinical Trial Data Extractor - Production Backend
Comprehensive PDF extraction using multiple methods: OCR, LLM, Tables, Heuristics
"""
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib
import tempfile

# PDF Processing Libraries
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import camelot

# LLM
import openai
from openai import OpenAI

# Data Processing
import numpy as np
from collections import defaultdict
import io
import pandas as pd

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://localhost/clinical_trials')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

db = SQLAlchemy(app)
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== DATABASE MODELS ====================

class Study(db.Model):
    __tablename__ = 'studies'
    
    id = db.Column(db.Integer, primary_key=True)
    pdf_hash = db.Column(db.String(64), unique=True, nullable=False)
    pdf_blob = db.Column(db.LargeBinary)
    pdf_filename = db.Column(db.String(500))
    
    title = db.Column(db.Text)
    authors = db.Column(db.JSON)
    journal = db.Column(db.String(500))
    year = db.Column(db.Integer)
    doi = db.Column(db.String(200))
    trial_registration = db.Column(db.String(100))
    
    study_type = db.Column(db.Text)  # Enhanced to store source citations
    blinding = db.Column(db.Text)    # Enhanced to store source citations
    randomization = db.Column(db.Text)
    duration = db.Column(db.Text)    # Enhanced to store source citations
    
    population_data = db.Column(db.JSON)
    baseline_characteristics = db.Column(db.JSON)
    
    extraction_metadata = db.Column(db.JSON)
    confidence_scores = db.Column(db.JSON)
    source_tracking = db.Column(db.JSON)  # New field for detailed source citations
    extraction_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    interventions = db.relationship('Intervention', backref='study', lazy=True, cascade='all, delete-orphan')
    outcomes = db.relationship('Outcome', backref='study', lazy=True, cascade='all, delete-orphan')
    subgroups = db.relationship('SubgroupAnalysis', backref='study', lazy=True, cascade='all, delete-orphan')
    adverse_events = db.relationship('AdverseEvent', backref='study', lazy=True, cascade='all, delete-orphan')

class Intervention(db.Model):
    __tablename__ = 'interventions'
    
    id = db.Column(db.Integer, primary_key=True)
    study_id = db.Column(db.Integer, db.ForeignKey('studies.id'), nullable=False)
    arm_name = db.Column(db.String(200))
    n_randomized = db.Column(db.Integer)
    n_analyzed = db.Column(db.Integer)
    dose = db.Column(db.String(200))
    frequency = db.Column(db.String(200))
    duration = db.Column(db.String(200))

class Outcome(db.Model):
    __tablename__ = 'outcomes'
    
    id = db.Column(db.Integer, primary_key=True)
    study_id = db.Column(db.Integer, db.ForeignKey('studies.id'), nullable=False)
    outcome_name = db.Column(db.Text)
    outcome_type = db.Column(db.String(50))  # primary, secondary
    outcome_category = db.Column(db.String(100))  # continuous, dichotomous, time_to_event, count
    planned_timepoints = db.Column(db.Text)  # All planned measurement times
    primary_timepoint_id = db.Column(db.Integer)  # Reference to primary timepoint
    
    # JSON field for complex outcome data that doesn't need individual querying
    additional_data = db.Column(db.JSON)  # For complex nested data
    data_sources = db.Column(db.JSON)  # Source tracking
    
    # Relationships to timepoint-specific data
    timepoints = db.relationship('OutcomeTimepoint', backref='outcome', lazy=True, cascade='all, delete-orphan')

class OutcomeTimepoint(db.Model):
    """Separate table for each timepoint measurement of an outcome"""
    __tablename__ = 'outcome_timepoints'
    
    id = db.Column(db.Integer, primary_key=True)
    outcome_id = db.Column(db.Integer, db.ForeignKey('outcomes.id'), nullable=False)
    study_id = db.Column(db.Integer, db.ForeignKey('studies.id'), nullable=False)  # For direct querying
    
    # Timepoint identification
    timepoint_name = db.Column(db.Text)  # "12 weeks", "end of treatment", "6-month follow-up"
    timepoint_value = db.Column(db.Float)  # Numeric value: 12, 6, 24
    timepoint_unit = db.Column(db.String(50))  # "weeks", "months", "days", "years"
    timepoint_type = db.Column(db.String(50))  # "primary", "secondary", "interim", "follow_up", "post_hoc"
    
    # Results that can be efficiently queried
    n_analyzed = db.Column(db.Integer)  # Sample size at this timepoint
    
    # Statistical results (for continuous outcomes)
    mean_value = db.Column(db.Float)
    sd_value = db.Column(db.Float)
    median_value = db.Column(db.Float)
    iqr_lower = db.Column(db.Float)
    iqr_upper = db.Column(db.Float)
    ci_95_lower = db.Column(db.Float)
    ci_95_upper = db.Column(db.Float)
    
    # Results for dichotomous outcomes
    events = db.Column(db.Integer)
    total_participants = db.Column(db.Integer)
    
    # Between-group comparison results
    effect_measure = db.Column(db.String(200))  # "MD", "SMD", "OR", "RR", "HR", or longer descriptions
    effect_estimate = db.Column(db.Float)
    effect_ci_lower = db.Column(db.Float)
    effect_ci_upper = db.Column(db.Float)
    p_value = db.Column(db.Float)
    p_value_text = db.Column(db.String(200))  # For "<0.001", "NS", etc.
    
    # Source tracking
    data_source = db.Column(db.Text)  # "Table 2 page 5", "Figure 1 page 3"
    source_confidence = db.Column(db.String(20))  # "high", "medium", "low"
    
    # JSON for arm-specific results and complex data
    results_by_arm = db.Column(db.JSON)  # Detailed arm-specific results
    additional_statistics = db.Column(db.JSON)  # Any other statistical data
    
    # Indexes for efficient querying
    __table_args__ = (
        db.Index('idx_timepoint_value_unit', 'timepoint_value', 'timepoint_unit'),
        db.Index('idx_timepoint_type', 'timepoint_type'),
        db.Index('idx_study_outcome', 'study_id', 'outcome_id'),
    )

class SubgroupAnalysis(db.Model):
    __tablename__ = 'subgroup_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    study_id = db.Column(db.Integer, db.ForeignKey('studies.id'), nullable=False)
    subgroup_variable = db.Column(db.String(200))
    subgroups = db.Column(db.JSON)
    p_interaction = db.Column(db.String(50))

class AdverseEvent(db.Model):
    __tablename__ = 'adverse_events'
    
    id = db.Column(db.Integer, primary_key=True)
    study_id = db.Column(db.Integer, db.ForeignKey('studies.id'), nullable=False)
    event_name = db.Column(db.Text)
    severity = db.Column(db.String(50))
    results_by_arm = db.Column(db.JSON)

# ==================== MULTI-SOURCE DATA MODELS ====================

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

# ==================== SOURCE ATTRIBUTION CONSTANTS ====================

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

CONFIDENCE_THRESHOLDS = {
    'high': 0.8,
    'medium': 0.6,
    'low': 0.4
}

# ==================== PDF EXTRACTION METHODS ====================

class PDFExtractor:
    """Multi-method PDF extraction with ensemble approach"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.methods_used = []
        self.page_count = 0
        self.pages_text = {}  # Store text by page number
        self.page_metadata = {}  # Store metadata by page
        
    def extract_text_pdfplumber(self) -> str:
        """Extract text using pdfplumber - best for modern PDFs"""
        try:
            text = ""
            with pdfplumber.open(self.pdf_path) as pdf:
                self.page_count = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    text += page_text
                    
                    # Store page-specific data for traceability
                    self.pages_text[page_num] = page_text
                    self.page_metadata[page_num] = {
                        'char_count': len(page_text),
                        'word_count': len(page_text.split()),
                        'has_tables': len(page.find_tables()) > 0 if hasattr(page, 'find_tables') else False,
                        'method': 'pdfplumber_text'
                    }
                    
            self.methods_used.append('pdfplumber_text')
            print(f"✅ PDFPlumber extracted {len(text)} characters from {self.page_count} pages")
            print(f"📄 Page breakdown: {[(p, meta['word_count']) for p, meta in self.page_metadata.items()]}")
            return text
        except Exception as e:
            print(f"❌ PDFPlumber text extraction failed: {e}")
            return ""
    
    def extract_tables_pdfplumber(self) -> List[Dict]:
        """Extract tables using pdfplumber"""
        try:
            tables = []
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Must have headers and at least one row
                            # Convert table to list of dicts with headers
                            headers = table[0] if table else []
                            rows = table[1:] if len(table) > 1 else []
                            
                            table_data = []
                            for row in rows:
                                if row and any(cell for cell in row if cell):  # Skip empty rows
                                    row_dict = {}
                                    for i, cell in enumerate(row):
                                        header = headers[i] if i < len(headers) and headers[i] else f"Column_{i}"
                                        row_dict[header] = cell if cell else ""
                                    table_data.append(row_dict)
                            
                            if table_data:
                                tables.append({
                                    'table_number': len(tables) + 1,
                                    'page': page_num,
                                    'data': table_data,
                                    'method': 'pdfplumber',
                                    'rows': len(table_data),
                                    'cols': len(headers)
                                })
            
            if tables:
                self.methods_used.append('pdfplumber_tables')
            print(f"PDFPlumber extracted {len(tables)} tables")
            return tables
        except Exception as e:
            print(f"PDFPlumber table extraction failed: {e}")
            return []
    
    def extract_tables_camelot(self) -> List[Dict]:
        """Extract tables using Camelot - excellent for structured tables"""
        try:
            # Try lattice mode first (for tables with clear borders)
            tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='lattice')
            extracted_tables = []
            
            for i, table in enumerate(tables):
                if table.accuracy > 50:  # Only use tables with >50% accuracy
                    extracted_tables.append({
                        'table_number': len(extracted_tables) + 1,
                        'page': table.page,
                        'data': table.df.to_dict('records'),
                        'accuracy': table.accuracy,
                        'method': 'camelot_lattice',
                        'rows': len(table.df),
                        'cols': len(table.df.columns)
                    })
            
            # Try stream mode for tables without clear borders (if we got few tables)
            if len(extracted_tables) < 3:
                stream_tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='stream')
                for i, table in enumerate(stream_tables):
                    if table.accuracy > 40:  # Lower threshold for stream
                        extracted_tables.append({
                            'table_number': len(extracted_tables) + 1,
                            'page': table.page,
                            'data': table.df.to_dict('records'),
                            'accuracy': table.accuracy,
                            'method': 'camelot_stream',
                            'rows': len(table.df),
                            'cols': len(table.df.columns)
                        })
            
            if extracted_tables:
                self.methods_used.append('camelot')
            print(f"Camelot extracted {len(extracted_tables)} tables")
            return extracted_tables
            
        except Exception as e:
            print(f"Camelot table extraction failed: {e}")
            return []
    
    def extract_with_ocr(self) -> str:
        """OCR extraction for scanned PDFs or images - always run for clinical trials"""
        try:
            print("🔍 Converting PDF pages to images for OCR...")
            # Specify Poppler path explicitly
            poppler_path = r"C:\poppler\poppler-25.07.0\Library\bin"
            images = convert_from_path(self.pdf_path, dpi=300, poppler_path=poppler_path)
            text = ""
            
            for i, image in enumerate(images):
                print(f"📷 OCR processing page {i+1}/{len(images)}...")
                custom_config = r'--oem 3 --psm 6'
                page_text = pytesseract.image_to_string(image, config=custom_config)
                text += f"\n--- OCR Page {i+1} ---\n{page_text}"
                
                # Store OCR-specific page data for traceability
                if hasattr(self, 'page_metadata'):
                    if i+1 not in self.page_metadata:
                        self.page_metadata[i+1] = {}
                    self.page_metadata[i+1]['ocr_char_count'] = len(page_text)
                    self.page_metadata[i+1]['ocr_method'] = 'tesseract'
            
            self.methods_used.append('tesseract_ocr')
            print(f"✅ OCR extracted {len(text)} characters from {len(images)} pages")
            return text
        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            return ""
    
    def comprehensive_extract(self) -> Dict:
        """Run all extraction methods and combine results"""
        results = {
            'text': '',
            'tables': [],
            'methods_used': [],
            'page_count': 0
        }
        
        # Primary text extraction
        text = self.extract_text_pdfplumber()
        
        # Always run OCR for clinical trials to catch figure text and image content
        print("🔍 Running OCR extraction to capture figure text and image content...")
        ocr_text = self.extract_with_ocr()
        
        # Combine text sources
        if text and ocr_text:
            # Use primary text but append unique OCR content
            combined_text = text + "\n\n=== OCR EXTRACTED CONTENT ===\n" + ocr_text
            print(f"📄 Combined text: {len(text)} chars (pdfplumber) + {len(ocr_text)} chars (OCR)")
        elif ocr_text and not text:
            combined_text = ocr_text
            print("📄 Using OCR text only (pdfplumber failed)")
        elif text and not ocr_text:
            combined_text = text
            print("📄 Using pdfplumber text only (OCR failed)")
        else:
            combined_text = ""
            print("❌ Both text extraction methods failed")
        
        results['text'] = combined_text
        results['page_count'] = self.page_count
        
        # Table extraction - try both methods
        tables = []
        
        # Try pdfplumber first (fast)
        pdfplumber_tables = self.extract_tables_pdfplumber()
        tables.extend(pdfplumber_tables)
        
        # Try Camelot (better for older PDFs with image tables)
        camelot_tables = self.extract_tables_camelot()
        tables.extend(camelot_tables)
        
        results['tables'] = tables
        results['methods_used'] = self.methods_used
        
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Methods used: {', '.join(self.methods_used)}")
        print(f"Text extracted: {len(text)} characters")
        print(f"Tables found: {len(tables)} (pdfplumber: {len(pdfplumber_tables)}, camelot: {len(camelot_tables)})")
        print(f"Pages: {self.page_count}")
        print(f"========================\n")
        
        return results

# ==================== LLM EXTRACTION ====================

class LLMExtractor:
    """Advanced LLM-based extraction with GPT-4"""
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4-turbo-preview"
        self.tokens_used = 0
        self.finish_reason = None
    
    def extract_trial_data(self, text: str, tables: List[Dict] = None, pages_text: Dict = None) -> Dict:
        """Extract comprehensive trial data using GPT-4 with enhanced source tracking"""
        
        # Prepare context with tables if available
        table_context = ""
        if tables:
            table_context = "\n\n=== EXTRACTED TABLES ===\n"
            for i, table in enumerate(tables[:10]):
                table_context += f"\nTable {i+1} (Page {table.get('page', 'unknown')}, {table.get('rows', 0)} rows x {table.get('cols', 0)} cols):\n"
                table_context += json.dumps(table['data'][:20], indent=2)
        
        # Add page-by-page context for better source tracking
        page_context = ""
        if pages_text:
            page_context = "\n\n=== PAGE-BY-PAGE CONTENT ===\n"
            for page_num, page_text in pages_text.items():
                page_context += f"\n--- PAGE {page_num} ---\n"
                page_context += page_text[:3000]  # First 3000 chars per page
                if len(page_text) > 3000:
                    page_context += "\n[PAGE CONTENT TRUNCATED]\n"
        
        prompt = self._create_extraction_prompt(text, table_context, page_context)
        
        print(f"\n=== LLM EXTRACTION STARTING ===")
        print(f"Model: {self.model}")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Tables included: {len(tables) if tables else 0}")
        
        # Smart tiered prompt management based on OpenAI token limits
        estimated_tokens = len(prompt) // 4  # Rough estimate: 4 chars per token
        
        if estimated_tokens > 25000:  # Approaching 30k token limit
            print(f"⚠️ TOKEN LIMIT: Estimated {estimated_tokens} tokens exceeds limit. Creating optimized version...")
            prompt = self._create_focused_extraction_prompt(text, table_context, tables)
            print(f"📝 Optimized prompt length: {len(prompt)} characters (~{len(prompt)//4} tokens)")
        elif estimated_tokens > 20000:  # Yellow zone - mild optimization
            print(f"📊 LARGE PROMPT: {estimated_tokens} estimated tokens. Using selective optimization...")
            # Preserve key content but reduce redundancy
            if len(page_context) > 15000:
                page_context = page_context[:15000] + "\n[PAGE CONTEXT TRUNCATED]"
            prompt = self._create_extraction_prompt(text[:70000], table_context, page_context)
            print(f"📝 Selective optimization: {len(prompt)} characters (~{len(prompt)//4} tokens)")
        else:
            print(f"📊 Using full extraction prompt ({len(prompt)} chars, ~{estimated_tokens} tokens) - within limits")
        
        try:
            print("Making OpenAI API request...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert clinical trial data extractor with 20+ years of experience in systematic reviews and network meta-analyses. 
                        You extract data with extreme precision, never hallucinate, and always indicate when information is uncertain or missing.
                        You are particularly skilled at identifying all statistical parameters, subgroup analyses, and nuanced methodological details.
                        You search the ENTIRE paper thoroughly, especially the Methods section for study design details."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"},
                timeout=120  # 2 minute timeout
            )
            
            # Store metadata
            self.tokens_used = response.usage.total_tokens
            self.finish_reason = response.choices[0].finish_reason
            
            raw_content = response.choices[0].message.content
            
            print(f"=== LLM RESPONSE RECEIVED ===")
            print(f"Response length: {len(raw_content)} characters")
            print(f"Tokens used: {self.tokens_used}")
            print(f"Finish reason: {self.finish_reason}")
            
            if self.finish_reason == 'length':
                print("⚠️ WARNING: Response was truncated due to token limit!")
            
            extracted_data = json.loads(raw_content)
            
            # Debug: Check what was extracted
            print(f"=== EXTRACTION CHECK ===")
            print(f"Has study_identification: {bool(extracted_data.get('study_identification'))}")
            print(f"Has study_design: {bool(extracted_data.get('study_design'))}")
            print(f"Has interventions: {len(extracted_data.get('interventions', []))} arms")
            print(f"Has primary outcomes: {len(extracted_data.get('outcomes', {}).get('primary', []))}")
            print(f"Has secondary outcomes: {len(extracted_data.get('outcomes', {}).get('secondary', []))}")
            
            # Debug: Show sample of extracted data
            if extracted_data.get('outcomes', {}).get('primary'):
                primary = extracted_data['outcomes']['primary'][0]
                print(f"Primary outcome name: {primary.get('outcome_name', 'Not found')}")
                if primary.get('timepoints'):
                    print(f"Primary timepoints found: {len(primary['timepoints'])}")
                    print(f"First timepoint: {primary['timepoints'][0].get('timepoint_name', 'No name')}")
                elif primary.get('results_by_arm'):
                    print(f"Primary results by arm: {len(primary['results_by_arm'])} arms")
                else:
                    print("No results data found in primary outcome")
            print(f"========================\n")
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"❌ LLM JSON parsing error: {e}")
            print(f"First 500 chars of response: {raw_content[:500] if 'raw_content' in locals() else 'N/A'}")
            return {}
        except openai.RateLimitError as e:
            print(f"❌ LLM rate limit error: {e}")
            print(f"🔄 Attempting retry with smaller prompt...")
            # Fallback: Use much more aggressive focusing
            focused_prompt = self._create_focused_extraction_prompt(text[:30000], table_context[:5000], [])
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert clinical trial data extractor. Extract key data with source citations."},
                        {"role": "user", "content": focused_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=3000,
                    response_format={"type": "json_object"},
                    timeout=60
                )
                raw_content = response.choices[0].message.content
                extracted_data = json.loads(raw_content)
                print(f"✅ Retry successful with focused extraction")
                return extracted_data
            except Exception as retry_error:
                print(f"❌ Retry also failed: {retry_error}")
                return {}
        except openai.APIError as e:
            print(f"❌ LLM API error: {e}")
            return {}
        except Exception as e:
            print(f"❌ LLM extraction error: {e}")
            return {}
    
    def _create_extraction_prompt(self, text: str, table_context: str, page_context: str = "") -> str:
        """Create comprehensive extraction prompt with enhanced source tracking"""
        return f"""You are an expert clinical trial data extractor. Extract ALL data with MANDATORY source citations.

🔍 CRITICAL SOURCE TRACKING RULES:
1. For EVERY single data point, you MUST specify: "Source: page X" or "Source: Table Y on page Z" or "Source: Figure A on page B"
2. When you find the same data in multiple places, list ALL sources: "Source: page 2 methods section, Table 1 on page 4"
3. If you cannot find specific data, write "NOT_FOUND" - do not guess or approximate
4. Always search the ENTIRE document - abstract, methods, results, tables, figures, appendix
5. For tables, identify the exact table number and page
6. For statistical values, specify the exact location where you found each number

📊 DATA COMPLETENESS REQUIREMENTS:
1. Extract EVERY number, statistic, and parameter - NEVER write "NOT_SPECIFIED"
2. Search MULTIPLE locations for each piece of data: abstract, methods, results, tables, figures
3. For missing standard deviations, look for: SE (standard error), CI (confidence intervals), p-values
4. For missing sample sizes, check: CONSORT diagram, baseline table, results tables
5. For missing demographics, search: Table 1, baseline characteristics, participant flow
6. If truly not found after thorough search, write "NOT_FOUND_AFTER_COMPREHENSIVE_SEARCH"

🔍 MANDATORY SEARCH REQUIREMENTS:
- CONSORT diagram: Search for "Figure 1", "participant flow", "enrollment", "randomization"
- Baseline table: Search for "Table 1", "baseline characteristics", "demographics"  
- Results tables: Search for "Table 2", "Table 3", "primary outcomes", "secondary outcomes"
- Statistical parameters: Search for every instance of numbers, means, SDs, CIs, p-values

🎯 STATISTICAL DATA PRIORITY:
- Primary outcomes: Extract ALL means, SDs, CIs, p-values with exact sources
- Baseline characteristics: Age, sex, race, comorbidities with exact table references  
- Sample sizes: Screened, randomized, analyzed - specify exact source location
- Effect estimates: Always include confidence intervals and p-values with source

EXTRACT WITH MANDATORY SOURCE CITATIONS:
1. SEARCH THE ENTIRE METHODS SECTION for study design details
2. Look for these EXACT phrases: "randomized", "double-blind", "single-blind", "open-label", "placebo-controlled", "multi-center", "single-center", "multicenter", "multicentre"
3. Look for blinding information in: title, abstract, methods section, and anywhere else
4. Search for country mentions throughout the entire paper
5. COUNT ALL INSTITUTIONS/SITES mentioned in author affiliations and methods - list EVERY SINGLE ONE
6. Look for allocation ratio (e.g., "1:1", "2:1", "randomized in a 1:1 ratio")
7. Find study duration in methods (e.g., "12 weeks", "6 months")
8. Extract treatment duration and follow-up duration separately

STATISTICAL PARAMETERS - EXTRACT EVERYTHING:
1. Extract EVERY numerical value, statistical parameter, and data point
2. Include ALL measures of variation: SD, SE, IQR, Q1, Q3, median, min, max, confidence intervals
3. Include ALL subgroup analyses with complete statistics
4. Never hallucinate - if data is unclear or missing, mark as "NOT_REPORTED"
5. Extract exact values from tables when available
6. Include confidence intervals, p-values, standard deviations, medians, IQRs, event counts
7. Capture baseline characteristics for all arms
8. Extract adverse event frequencies

CRITICAL: Extract PRIMARY and SECONDARY OUTCOMES with ALL statistical details. This is ESSENTIAL for meta-analysis.

🎯 OUTCOME EXTRACTION REQUIREMENTS:
1. PRIMARY OUTCOMES: Look for "primary endpoint", "primary outcome", "main outcome" - extract ALL statistics
2. SECONDARY OUTCOMES: Look for "secondary endpoint", "secondary outcome", "additional outcomes" - extract EVERY ONE
   - Common secondary outcomes include: diastolic BP, heart rate, adverse events, QoL, laboratory values
   - Check EVERY results table (Table 2, Table 3, Table 4, etc.) for additional secondary outcomes
   - Look for outcomes mentioned in methods but reported separately in results
   - DO NOT LIMIT to just the first secondary outcome - extract ALL of them
3. **MULTIPLE TIMEPOINTS**: Extract the SAME outcome measured at DIFFERENT times:
   - Short-term effects (immediate, 1-7 days)
   - Medium-term effects (1-12 weeks)  
   - Long-term effects (3-12+ months)
   - Follow-up timepoints (post-treatment)
   - Interim analyses (planned looks during study)
4. Extract ALL outcomes from results tables - many secondary outcomes are only in tables
5. Look for Table 2, Table 3, etc. which often contain secondary outcomes
6. For each outcome, extract: means, SDs, confidence intervals, p-values, sample sizes
7. Common outcome types in clinical trials:
   - Clinical endpoints (symptoms, disease progression, events)
   - Laboratory values (biomarkers, vital signs, lab tests)
   - Patient-reported outcomes (quality of life, pain scores, functional scales)
   - Safety outcomes (adverse events, laboratory safety)
   - Composite endpoints (multiple outcomes combined)

⚠️ CRITICAL: If you find mention of secondary outcomes but incomplete data, note: "Additional secondary outcomes mentioned but require manual extraction from [specific location]"

🕐 TIMEPOINT EXTRACTION CRITICAL RULES:
- Look for phrases: "at 1 week", "at 12 weeks", "at 6 months", "at end of treatment", "at follow-up"
- Extract data from figures showing time curves (e.g., "week 4", "month 3", "day 30")
- Find interim results: "interim analysis", "planned interim look"
- Identify primary vs secondary timepoints: which timepoint is the main endpoint
- Look for post-hoc timepoint analyses: additional timepoints analyzed after study

Return JSON with this EXACT structure - EVERY field must include source citation:
{{
  "study_identification": {{
    "title": "complete title - Source: page X",
    "authors": ["list all authors - Source: page X"],
    "journal": "journal name - Source: page X",
    "year": "YYYY - Source: page X",
    "doi": "DOI - Source: page X", 
    "trial_registration": "NCT or ISRCTN number - Source: page X"
  }},
  "study_design": {{
    "type": "parallel-group RCT/crossover/cluster-randomized/factorial/etc - SEARCH for this in title, abstract, and methods",
    "blinding": "double-blind/single-blind/open-label/details - SEARCH entire paper for blinding information",
    "randomization_method": "exact description from methods section",
    "allocation_concealment": "method",
    "allocation_ratio": "e.g., 1:1 or 2:1 - LOOK for phrases like 'randomized in a X:Y ratio'",
    "sample_size_calculation": "complete power calculation",
    "duration_total": "total study duration - SEARCH for this",
    "duration_treatment": "treatment phase duration - SEARCH for this",
    "duration_followup": "followup duration - SEARCH for this",
    "number_of_sites": "COUNT institutions in affiliations and methods",
    "country": "countries involved - SEARCH entire paper",
    "sites": [
      {{
        "institution_name": "FULL institution name from affiliations or methods",
        "city": "city if available",
        "country": "country",
        "principal_investigator": "PI name if mentioned"
      }}
    ]
  }},
  "population": {{
    "total_screened": "FIND THIS: Search CONSORT diagram, Figure 1, enrollment numbers - Source: exact location",
    "total_randomized": "CALCULATE FROM ARMS: Add up all randomized numbers from interventions - Source: calculation or exact table", 
    "total_analyzed_itt": "FIND THIS: Search for 'intention-to-treat', 'ITT', 'analyzed population' - Source: exact location",
    "age_mean": "FIND THIS: Search Table 1, baseline characteristics, demographics section - Source: Table X page Y",
    "age_sd": "FIND THIS: Must be in baseline table - look for (SD), ±, standard deviation - Source: Table X page Y",
    "sex_male_percent": "FIND THIS: Search baseline table for 'male', 'sex', 'gender' percentages - Source: Table X page Y",
    "baseline_characteristics_source": "MANDATORY: Identify exact table/page where ALL demographics found",
    "age_range": "min-max ages if reported",
    "sex_male_n": "absolute number of males",
    "race_ethnicity": "race/ethnicity breakdown with percentages",
    "inclusion_criteria": ["FIND ALL inclusion criteria from methods section"],
    "exclusion_criteria": ["FIND ALL exclusion criteria from methods section"]
  }},
    "age_range": "min-max",
    "sex_male_n": "number male",
    "sex_male_percent": "percent male",
    "race_ethnicity": {{}},
    "baseline_disease_severity": {{}},
    "baseline_comorbidities": {{}},
    "prior_treatments": {{}}
  }},
  "interventions": [
    {{
      "arm_number": 1,
      "arm_name": "descriptive name",
      "intervention_type": "drug/device/behavioral/etc",
      "n_randomized": "number randomized to this arm",
      "n_started_treatment": "number who received intervention",
      "n_completed": "number who completed",
      "n_analyzed_primary": "number in primary analysis",
      "dropouts": "number and reasons",
      "drug_name": "generic and brand",
      "dose": "exact dosage",
      "frequency": "dosing schedule",
      "route": "oral/IV/SC/etc",
      "duration": "treatment duration",
      "concomitant_medications": "allowed medications",
      "baseline_characteristics": {{}}
    }}
  ],
  "outcomes": {{
    "primary": [
      {{
        "outcome_name": "exact outcome name - Source: page X",
        "outcome_type": "continuous/dichotomous/time_to_event/count - Source: page X",
        "timepoints": [
          {{
            "timepoint_name": "exact timepoint description - Source: page X",
            "timepoint_value": "numeric value - Source: page X", 
            "timepoint_unit": "days/weeks/months/years - Source: page X",
            "timepoint_type": "primary/interim/post_hoc/follow_up - Source: page X",
            "results_by_arm": [
              {{
                "arm": "arm name matching interventions",
                "n": "number analyzed at this timepoint - Source: Table X page Y",
                "mean": "mean value - Source: Table X page Y",
                "sd": "standard deviation - Source: Table X page Y", 
                "median": "median if reported - Source: Table X page Y",
                "iqr_lower": "IQR lower if reported - Source: Table X page Y",
                "iqr_upper": "IQR upper if reported - Source: Table X page Y",
                "ci_95_lower": "95% CI lower bound - Source: Table X page Y",
                "ci_95_upper": "95% CI upper bound - Source: Table X page Y",
                "events": "number of events for dichotomous outcomes - Source: Table X page Y",
                "total": "total participants for dichotomous outcomes - Source: Table X page Y"
              }}
            ],
            "between_group_comparison": {{
              "effect_measure": "MD/SMD/OR/RR/HR/difference - Source: page X",
              "effect_estimate": "point estimate - Source: Table X page Y",
              "ci_95_lower": "lower CI - Source: Table X page Y",
              "ci_95_upper": "upper CI - Source: Table X page Y", 
              "p_value": "exact p-value - Source: Table X page Y",
              "statistical_test": "t-test/ANOVA/chi-square/etc - Source: page X"
            }},
            "data_source": "exact location - page X, table Y, figure Z",
            "source_confidence": "high/medium/low"
          }}
        ],
        "planned_timepoints": "all planned measurement times - Source: methods section page X",
        "primary_timepoint": "which timepoint is primary endpoint - Source: page X"
      }}
    ],
    "secondary": [
      {{
        "outcome_name": "exact secondary outcome name - Source: page X",
        "outcome_type": "continuous/dichotomous/time_to_event/count - Source: page X",
        "timepoints": [
          {{
            "timepoint_name": "exact timepoint description - Source: page X",
            "timepoint_value": "numeric value - Source: page X", 
            "timepoint_unit": "days/weeks/months/years - Source: page X",
            "timepoint_type": "secondary/interim/post_hoc/follow_up - Source: page X",
            "results_by_arm": [
              {{
                "arm": "arm name matching interventions",
                "n": "number analyzed at this timepoint - Source: Table X page Y",
                "mean": "mean value - Source: Table X page Y",
                "sd": "standard deviation - Source: Table X page Y", 
                "median": "median if reported - Source: Table X page Y",
                "iqr_lower": "IQR lower if reported - Source: Table X page Y",
                "iqr_upper": "IQR upper if reported - Source: Table X page Y",
                "ci_95_lower": "95% CI lower bound - Source: Table X page Y",
                "ci_95_upper": "95% CI upper bound - Source: Table X page Y",
                "events": "number of events for dichotomous outcomes - Source: Table X page Y",
                "total": "total participants for dichotomous outcomes - Source: Table X page Y"
              }}
            ],
            "between_group_comparison": {{
              "effect_measure": "MD/SMD/OR/RR/HR/difference - Source: page X",
              "effect_estimate": "point estimate - Source: Table X page Y",
              "ci_95_lower": "lower CI - Source: Table X page Y",
              "ci_95_upper": "upper CI - Source: Table X page Y", 
              "p_value": "exact p-value - Source: Table X page Y",
              "statistical_test": "t-test/ANOVA/chi-square/etc - Source: page X"
            }},
            "data_source": "exact location - page X, table Y, figure Z",
            "source_confidence": "high/medium/low"
          }}
        ],
        "planned_timepoints": "all planned measurement times - Source: methods section page X"
      }}
    ],
        "source_confidence": "high/medium/low - how confident you are in this source"
      }}
    ]
  }},
  "subgroup_analyses": [
    {{
      "outcome": "which outcome",
      "subgroup_variable": "age/sex/baseline severity/region/etc",
      "prespecified": "yes/no/unclear",
      "subgroups": [
        {{
          "subgroup_name": "e.g., age <65",
          "subgroup_definition": "exact criteria",
          "results_by_arm": [
            {{
              "arm": "arm name",
              "n": "sample size in subgroup",
              "events": "events in subgroup",
              "total": "total in subgroup",
              "mean": "mean",
              "sd": "SD"
            }}
          ],
          "effect_estimate": {{
            "type": "OR/RR/MD/etc",
            "value": "point estimate",
            "ci_lower": "95% CI lower",
            "ci_upper": "95% CI upper",
            "p_value": "p-value"
          }}
        }}
      ],
      "test_for_interaction": "yes/no",
      "p_interaction": "p-value for interaction",
      "interpretation": "author's interpretation"
    }}
  ],
  "adverse_events": [
    {{
      "category": "serious/non-serious",
      "event_name": "specific AE",
      "severity_grade": "mild/moderate/severe or CTCAE grade",
      "results_by_arm": [
        {{
          "arm": "arm name",
          "events": "number of events",
          "participants_with_event": "number of participants",
          "total_exposed": "total in arm"
        }}
      ],
      "related_to_treatment": "yes/no/possibly",
      "led_to_discontinuation": "number discontinued due to this AE"
    }}
  ],
  "statistical_methods": {{
    "primary_analysis_method": "ANCOVA/t-test/logistic regression/etc",
    "adjustment_variables": ["covariates adjusted for"],
    "missing_data_method": "LOCF/multiple imputation/etc",
    "sensitivity_analyses": ["analysis 1", "analysis 2"],
    "multiplicity_adjustment": "method for multiple testing",
    "interim_analyses": "number and timing",
    "stopping_rules": "early stopping criteria"
  }},
  "risk_of_bias": {{
    "random_sequence_generation": "low/high/unclear with justification",
    "allocation_concealment": "low/high/unclear with justification",
    "blinding_participants_personnel": "low/high/unclear with justification",
    "blinding_outcome_assessment": "low/high/unclear with justification",
    "incomplete_outcome_data": "low/high/unclear with justification",
    "selective_reporting": "low/high/unclear with justification",
    "other_bias": "any other sources of bias"
  }},
  "funding_conflicts": {{
    "funding_source": "industry/government/nonprofit/mixed",
    "sponsor_name": "specific sponsor",
    "author_conflicts": "disclosed conflicts",
    "sponsor_role": "role in study design/analysis/writing"
  }},
  "data_extraction_notes": {{
    "confidence_overall": "high/medium/low",
    "uncertain_fields": ["list fields with uncertainty"],
    "requires_manual_review": ["fields needing expert review"],
    "figure_table_references": ["which figures/tables have key data"]
  }}
}}

CLINICAL TRIAL TEXT:
{text[:50000]}

{table_context}

{page_context}

🔍 SPECIAL SEARCH INSTRUCTIONS:
1. Look for "baseline characteristics" table - extract ALL demographics with exact sources
2. Look for primary outcome results table - extract means, SDs, CIs with exact table reference  
3. Search for "Table 1", "Table 2", etc. and specify which table contains which data
4. Look for CONSORT flow diagram for exact participant numbers
5. Search figures for any statistical data or effect estimates
6. Check appendix/supplementary materials mentioned in text

Extract now. MANDATORY: Every single data point must include its source location. Never write "NOT_SPECIFIED" - search harder or write "NOT_FOUND with exact source searched"."""

    def _create_focused_extraction_prompt(self, text: str, table_context: str, tables: List) -> str:
        """Ultra-compact prompt for efficient extraction - prioritizes tables"""
        
        # Extract only the most critical sections
        lines = text.split('\n')
        
        # Get results sections (compact)
        results = '\n'.join([l for l in lines if any(k in l.lower() for k in ['results', 'outcomes', 'table 2', 'table 3', 'endpoint', 'efficacy'])])[:15000]
        
        # Get baseline (compact)
        baseline = '\n'.join([l for l in lines[:400] if any(k in l.lower() for k in ['baseline', 'table 1', 'demographics', 'randomized', 'age', 'sex'])])[:5000]
        
        # Get methods (compact)
        methods = '\n'.join([l for l in lines[:600] if any(k in l.lower() for k in ['methods', 'design', 'participants', 'blinding', 'randomization'])])[:4000]
        
        # Prioritize table data - keep maximum amount
        table_data = table_context[:35000] if table_context else ""
        
        focused = f"""METHODS: {methods}\nBASELINE: {baseline}\nRESULTS: {results}\nTABLES: {table_data}"""
        
        return f"""Extract clinical trial data. For every value add "Source: page X" or "Source: Table Y page Z".

Extract ALL arms, ALL outcomes (primary+secondary), ALL timepoints, ALL statistics (mean, SD, CI, p-value).

Return JSON: {{"study_identification":{{"title":"...-Source:p.X","authors":["..."],"year":"YYYY","doi":"...","trial_registration":"..."}}, "study_design":{{"type":"...-Source:p.X","blinding":"...","randomization_method":"...","duration_total":"..."}}, "interventions":[{{"arm_name":"...-Source:p.X","n_randomized":"N-Source:TableX p.Y","n_analyzed":"N-Source:TableX p.Y","dose":"...","frequency":"..."}}], "outcomes":{{"primary":[{{"outcome_name":"...-Source:p.X","timepoint":"X weeks-Source:p.Y","results_by_arm":[{{"arm":"match intervention name","n":"N-Source:TblX p.Y","mean":"value-Source:TblX p.Y","sd":"value-Source:TblX p.Y","ci_95_lower":"value","ci_95_upper":"value"}}],"between_group_comparison":{{"effect_measure":"type","effect_estimate":"value-Source:TblX p.Y","ci_95_lower":"value","ci_95_upper":"value","p_value":"value-Source:TblX p.Y"}}}}],"secondary":[{{"outcome_name":"...-Source:p.X","timepoint":"X weeks","results_by_arm":[{{"arm":"...","n":"N","mean":"value","sd":"value"}}],"between_group_comparison":{{"effect_measure":"...","effect_estimate":"...","ci_95_lower":"...","ci_95_upper":"...","p_value":"..."}}}}]}}}}

CONTENT:
{focused}"""

# ==================== HEURISTIC EXTRACTION ====================

class HeuristicExtractor:
    """Rule-based extraction for common patterns"""
    
    @staticmethod
    def extract_statistical_values(text: str) -> Dict:
        """Extract p-values, CIs, ORs, RRs using regex patterns"""
        patterns = {
            'p_values': r'p\s*[=<>]\s*([0-9.]+|0\.0*1)',
            'confidence_intervals': r'95%?\s*CI[:\s]*\[?([0-9.]+)\s*[-–to]\s*([0-9.]+)\]?',
            'odds_ratios': r'OR[:\s]*([0-9.]+)',
            'relative_risks': r'RR[:\s]*([0-9.]+)',
            'hazard_ratios': r'HR[:\s]*([0-9.]+)',
            'mean_sd': r'([0-9.]+)\s*\(SD\s*[=:]?\s*([0-9.]+)\)',
            'nct_numbers': r'NCT\d{8}',
            # Enhanced demographic patterns
            'age_mean_sd': r'age[:\s]*([0-9.]+)\s*\(([0-9.]+)\)',
            'age_range': r'age[:\s]*([0-9]+)\s*[-–to]\s*([0-9]+)',
            'sample_sizes': r'[nN]\s*=\s*(\d+)',
            'percentages': r'(\d+\.?\d*)\s*%',
            'treatment_duration': r'(\d+)\s*(weeks?|months?|days?)',
            # Participant flow patterns
            'screened': r'(\d+)\s*(?:patients?|participants?|subjects?)?\s*(?:were\s*)?screened',
            'assessed_eligibility': r'(\d+)\s*(?:patients?|participants?|subjects?)?\s*(?:were\s*)?assessed\s*for\s*eligibility',
            'randomized_total': r'(\d+)\s*(?:patients?|participants?|subjects?)?\s*(?:were\s*)?randomized',
            'analyzed_total': r'(\d+)\s*(?:patients?|participants?|subjects?)?\s*(?:were\s*)?analyzed',
            'enrolled': r'(\d+)\s*(?:patients?|participants?|subjects?)?\s*(?:were\s*)?enrolled',
            'consort_flow': r'CONSORT|participant\s*flow|enrollment',
        }
        
        results = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            results[key] = matches
        
        return results
    
    @staticmethod
    def extract_sample_sizes(text: str) -> List[int]:
        """Extract sample size mentions"""
        patterns = [
            r'n\s*=\s*(\d+)',
            r'N\s*=\s*(\d+)',
            r'(\d+)\s+patients',
            r'(\d+)\s+subjects',
            r'(\d+)\s+participants'
        ]
        
        sizes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sizes.extend([int(m) for m in matches])
        
        return sizes
    
    @staticmethod
    def extract_demographics(text: str) -> Dict:
        """Extract demographic information using patterns"""
        demographics = {}
        
        # Age patterns
        age_patterns = [
            r'mean\s+age[:\s]*([0-9.]+)',
            r'age[:\s]*([0-9.]+)\s*years',
            r'aged\s+([0-9.]+)',
            r'age[:\s]*([0-9.]+)\s*\(([0-9.]+)\)',  # mean (sd)
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                    demographics['age_mean'] = float(matches[0][0])
                    demographics['age_sd'] = float(matches[0][1])
                else:
                    demographics['age_mean'] = float(matches[0])
                break
        
        # Gender patterns
        gender_text = text.lower()
        male_patterns = [
            r'(\d+)\s*\(\s*([0-9.]+)%\s*\)\s*male',
            r'male[:\s]*(\d+)',
            r'men[:\s]*(\d+)',
        ]
        
        female_patterns = [
            r'(\d+)\s*\(\s*([0-9.]+)%\s*\)\s*female',
            r'female[:\s]*(\d+)',
            r'women[:\s]*(\d+)',
        ]
        
        for pattern in male_patterns:
            matches = re.findall(pattern, gender_text)
            if matches:
                if isinstance(matches[0], tuple):
                    demographics['male_count'] = int(matches[0][0])
                    demographics['male_percentage'] = float(matches[0][1])
                else:
                    demographics['male_count'] = int(matches[0])
                break
        
        for pattern in female_patterns:
            matches = re.findall(pattern, gender_text)
            if matches:
                if isinstance(matches[0], tuple):
                    demographics['female_count'] = int(matches[0][0])
                    demographics['female_percentage'] = float(matches[0][1])
                else:
                    demographics['female_count'] = int(matches[0])
                break
        
        return demographics

# ==================== ENSEMBLE & VALIDATION ====================

class EnsembleExtractor:
    """Combine results from multiple extraction methods"""
    
    def __init__(self):
        self.pdf_extractor = None
        self.llm_extractor = LLMExtractor()
        self.heuristic_extractor = HeuristicExtractor()
    
    def extract_comprehensive(self, pdf_path: str) -> Tuple[Dict, Dict]:
        """Run all extraction methods and ensemble results"""
        
        # Step 1: PDF extraction
        self.pdf_extractor = PDFExtractor(pdf_path)
        pdf_data = self.pdf_extractor.comprehensive_extract()
        
        # Step 2: LLM extraction with enhanced source tracking
        llm_data = self.llm_extractor.extract_trial_data(
            pdf_data['text'], 
            pdf_data.get('tables', []),
            self.pdf_extractor.pages_text if hasattr(self.pdf_extractor, 'pages_text') else None
        )
        
        # Step 3: Heuristic extraction - ENHANCED
        heuristic_stats = self.heuristic_extractor.extract_statistical_values(pdf_data['text'])
        heuristic_demographics = self.heuristic_extractor.extract_demographics(pdf_data['text'])
        heuristic_data = {**heuristic_stats, 'demographics': heuristic_demographics}
        
        # Store for later access by multi-source storage
        self._last_llm_data = llm_data
        self._last_heuristic_data = heuristic_data
        self._last_pdf_data = pdf_data
        
        # Step 4: Ensemble and validation - NOW ACTUALLY USES ALL DATA
        final_data = self._ensemble_results(llm_data, heuristic_data, pdf_data)
        
        # Step 5: Calculate confidence scores
        confidence_scores = self._calculate_confidence(final_data, pdf_data, heuristic_data)
        
        metadata = {
            'extraction_methods': pdf_data['methods_used'] + ['llm', 'heuristic', 'demographics'],
            'tables_found': len(pdf_data.get('tables', [])),
            'pdf_pages': pdf_data.get('page_count', 0),
            'confidence_scores': confidence_scores,
            'llm_model': self.llm_extractor.model,
            'llm_tokens': self.llm_extractor.tokens_used,
            'llm_finish_reason': self.llm_extractor.finish_reason,
            'heuristic_findings': len(heuristic_stats),
            'demographics_extracted': bool(heuristic_demographics),
            # Enhanced process details
            'pdf_text_length': len(pdf_data.get('text', '')),
            'ocr_text_length': len(pdf_data.get('ocr_text', '')),
            'tables_extracted': len(pdf_data.get('tables', [])),
            'heuristic_patterns_found': {
                'p_values': len(heuristic_stats.get('p_values', [])),
                'confidence_intervals': len(heuristic_stats.get('confidence_intervals', [])),
                'sample_sizes': len(heuristic_stats.get('sample_sizes', [])),
                'demographics': len(heuristic_demographics) if heuristic_demographics else 0
            },
            'extraction_success': {
                'pdf_extraction': len(pdf_data.get('text', '')) > 0,
                'ocr_extraction': len(pdf_data.get('ocr_text', '')) > 0,
                'table_extraction': len(pdf_data.get('tables', [])) > 0,
                'llm_extraction': self.llm_extractor.tokens_used > 0,
                'heuristic_extraction': len(heuristic_stats) > 0,
                'demographics_extraction': bool(heuristic_demographics)
            }
        }
        
        return final_data, metadata
    
    def extract_comprehensive_with_sources(self, pdf_path: str, study_id: int = None) -> Tuple[Dict, Dict]:
        """Enhanced extraction that stores multi-source data"""
        final_data, metadata = self.extract_comprehensive(pdf_path)
        
        # Store multi-source data if study_id is provided
        if study_id:
            try:
                # Get individual source data for storage
                pdf_data = self.pdf_extractor.comprehensive_extract() if self.pdf_extractor else {}
                llm_data = getattr(self, '_last_llm_data', {})
                heuristic_data = getattr(self, '_last_heuristic_data', {})
                ocr_text = pdf_data.get('ocr_text', '')
                
                self.store_multi_source_data(study_id, llm_data, heuristic_data, pdf_data, ocr_text)
            except Exception as e:
                print(f"Warning: Could not store multi-source data: {e}")
                metadata['multi_source_storage_error'] = str(e)
        
        return final_data, metadata
    
    def _ensemble_results(self, llm_data: Dict, heuristic_data: Dict, pdf_data: Dict) -> Dict:
        """Combine and validate results from different methods"""
        ensembled = llm_data.copy()
        
        # Add extracted tables from PDF
        if pdf_data.get('tables'):
            ensembled['extracted_tables'] = pdf_data['tables']
        
        # CRITICAL FIX: Actually use heuristic extraction data!
        if heuristic_data:
            # Add statistical values found by pattern matching
            ensembled['heuristic_findings'] = heuristic_data
            
            # Enhance outcomes with heuristic statistical data
            if 'outcomes' in ensembled:
                self._enhance_outcomes_with_heuristics(ensembled['outcomes'], heuristic_data)
            
            # Add sample sizes if found
            if heuristic_data.get('mean_sd'):
                ensembled['population_statistics'] = {
                    'mean_sd_values': heuristic_data['mean_sd'],
                    'source': 'heuristic_extraction'
                }
            
            # Add demographics from heuristic extraction
            if heuristic_data.get('demographics'):
                if 'participants' not in ensembled:
                    ensembled['participants'] = {}
                ensembled['participants'].update(heuristic_data['demographics'])
                ensembled['participants']['heuristic_source'] = True
            
            # CRITICAL: Add participant flow data from heuristic patterns
            participant_flow = self._extract_participant_flow_from_heuristics(heuristic_data)
            if participant_flow:
                if 'participants' not in ensembled:
                    ensembled['participants'] = {}
                ensembled['participants'].update(participant_flow)
        
        # Extract demographics from tables if available
        if pdf_data.get('tables'):
            demographics = self._extract_demographics_from_tables(pdf_data['tables'])
            if demographics:
                if 'participants' not in ensembled:
                    ensembled['participants'] = {}
                ensembled['participants'].update(demographics)
            
            # Also check tables for participant flow data
            table_flow = self._extract_participant_flow_from_tables(pdf_data['tables'])
            if table_flow:
                if 'participants' not in ensembled:
                    ensembled['participants'] = {}
                ensembled['participants'].update(table_flow)
        
        # CRITICAL: Post-process and complete missing data
        ensembled = self._complete_missing_data(ensembled, pdf_data)
        
        return ensembled
    
    def store_multi_source_data(self, study_id: int, llm_data: Dict, heuristic_data: Dict, 
                               pdf_data: Dict, ocr_text: str = None) -> None:
        """Store extraction data from each source in the database"""
        import time
        start_time = time.time()
        
        # Store raw extraction sources
        sources_to_store = [
            ('llm', llm_data, 0.8, {'model': 'gpt-4', 'tokens_used': getattr(self.llm_extractor, 'tokens_used', 0)}),
            ('heuristic', heuristic_data, 0.7, {'patterns_matched': len(heuristic_data) if heuristic_data else 0}),
            ('pdfplumber', {'tables': pdf_data.get('tables', []), 'text_length': len(pdf_data.get('text', ''))}, 
             0.9, {'tables_found': len(pdf_data.get('tables', [])), 'pages': pdf_data.get('page_count', 0)})
        ]
        
        if ocr_text:
            sources_to_store.append(('ocr', {'text': ocr_text, 'text_length': len(ocr_text)}, 
                                   0.6, {'text_extracted': bool(ocr_text)}))
        
        # Store each source's raw data
        for method, raw_data, confidence, quality_metrics in sources_to_store:
            extraction_source = ExtractionSource(
                study_id=study_id,
                extraction_method=method,
                source_version='1.0',
                raw_data=raw_data,
                confidence_score=confidence,
                quality_metrics=quality_metrics,
                processing_time=time.time() - start_time,
                success_status=True
            )
            db.session.add(extraction_source)
        
        # Store individual data elements with source attribution
        self._store_data_elements(study_id, llm_data, heuristic_data, pdf_data, ocr_text)
        
        # Detect and store conflicts
        self._detect_and_store_conflicts(study_id)
        
        try:
            db.session.commit()
        except Exception as e:
            print(f"Warning: Could not store multi-source data: {e}")
            db.session.rollback()
    
    def _store_data_elements(self, study_id: int, llm_data: Dict, heuristic_data: Dict, 
                            pdf_data: Dict, ocr_text: str = None) -> None:
        """Store individual data elements with source attribution"""
        
        # Extract values from each source for key fields
        key_fields = [
            ('study_identification.title', 'identification'),
            ('study_identification.authors', 'identification'),
            ('study_identification.year', 'identification'),
            ('study_design.type', 'design'),
            ('study_design.blinding', 'design'),
            ('population.total_randomized', 'population'),
            ('population.total_screened', 'population'),
            ('population.total_analyzed', 'population')
        ]
        
        for field_path, element_type in key_fields:
            element_name = field_path.split('.')[-1]
            
            # Extract values from each source
            llm_value = self._extract_field_value(llm_data, field_path)
            heuristic_value = self._extract_field_value(heuristic_data, field_path)
            pdf_value = self._extract_field_value(pdf_data, field_path)
            ocr_value = None  # OCR doesn't typically extract structured data
            
            # Calculate confidence scores
            llm_confidence = 0.8 if llm_value else 0.0
            heuristic_confidence = 0.7 if heuristic_value else 0.0
            pdf_confidence = 0.9 if pdf_value else 0.0
            ocr_confidence = 0.0
            
            # Determine source locations
            llm_location = "AI interpretation"
            heuristic_location = "Pattern matching"
            pdf_location = self._find_pdf_source_location(pdf_data, field_path)
            ocr_location = "Not found"
            
            # Check if any sources have values
            if any([llm_value, heuristic_value, pdf_value]):
                # Detect conflicts
                sources_with_values = [
                    ('llm', llm_value),
                    ('heuristic', heuristic_value), 
                    ('pdfplumber', pdf_value)
                ]
                sources_with_values = [(source, value) for source, value in sources_with_values if value]
                
                conflicting = len(set(str(value) for _, value in sources_with_values)) > 1
                
                data_element = DataElement(
                    study_id=study_id,
                    element_type=element_type,
                    element_name=element_name,
                    element_path=field_path,
                    llm_value=str(llm_value) if llm_value else None,
                    llm_confidence=llm_confidence,
                    llm_source_location=llm_location,
                    heuristic_value=str(heuristic_value) if heuristic_value else None,
                    heuristic_confidence=heuristic_confidence,
                    heuristic_source_location=heuristic_location,
                    pdfplumber_value=str(pdf_value) if pdf_value else None,
                    pdfplumber_confidence=pdf_confidence,
                    pdfplumber_source_location=pdf_location,
                    ocr_value=str(ocr_value) if ocr_value else None,
                    ocr_confidence=ocr_confidence,
                    ocr_source_location=ocr_location,
                    conflicting_sources=conflicting,
                    needs_review=conflicting or len(sources_with_values) == 1  # Review if conflicting or only one source
                )
                
                db.session.add(data_element)
    
    def _extract_field_value(self, data: Dict, field_path: str):
        """Extract a field value from nested data using dot notation"""
        if not data:
            return None
        
        parts = field_path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current if current not in ['NOT_REPORTED', 'NOT_SPECIFIED', None, ''] else None
    
    def _find_pdf_source_location(self, pdf_data: Dict, field_path: str) -> str:
        """Find the source location in PDF for a field"""
        # This is a simplified implementation - in practice, you'd track this during extraction
        if pdf_data.get('tables') and 'population' in field_path:
            return f"Table data (found {len(pdf_data['tables'])} tables)"
        elif pdf_data.get('text'):
            return "PDF text extraction"
        else:
            return "PDF source unknown"
    
    def _detect_and_store_conflicts(self, study_id: int) -> None:
        """Detect and store conflicts between extraction sources"""
        conflicting_elements = DataElement.query.filter_by(
            study_id=study_id, 
            conflicting_sources=True
        ).all()
        
        for element in conflicting_elements:
            # Find the conflicting sources
            sources = [
                ('llm', element.llm_value, element.llm_confidence),
                ('heuristic', element.heuristic_value, element.heuristic_confidence),
                ('pdfplumber', element.pdfplumber_value, element.pdfplumber_confidence),
                ('ocr', element.ocr_value, element.ocr_confidence)
            ]
            
            sources_with_values = [(s, v, c) for s, v, c in sources if v]
            
            if len(sources_with_values) >= 2:
                # Create conflict record for the highest confidence conflicting pair
                sources_with_values.sort(key=lambda x: x[2], reverse=True)
                source_a, value_a, conf_a = sources_with_values[0]
                source_b, value_b, conf_b = sources_with_values[1]
                
                if str(value_a) != str(value_b):
                    conflict = ExtractionConflict(
                        study_id=study_id,
                        data_element_id=element.id,
                        conflict_type='value_mismatch',
                        severity='medium',
                        source_a=source_a,
                        value_a=str(value_a),
                        confidence_a=conf_a,
                        source_b=source_b,
                        value_b=str(value_b),
                        confidence_b=conf_b
                    )
                    db.session.add(conflict)
    
    def _complete_missing_data(self, data: Dict, pdf_data: Dict) -> Dict:
        """Complete missing data by calculating totals and extracting from tables"""
        
        # 1. Calculate total participant flow from individual arms if missing
        if data.get('interventions') and data.get('participants'):
            participants = data['participants']
            interventions = data['interventions']
            
            # Calculate total randomized if missing
            if not participants.get('total_randomized'):
                total_randomized = 0
                randomized_values = []
                
                for arm in interventions:
                    if arm.get('n_randomized'):
                        # Enhanced number extraction
                        n_val = self._extract_number_from_string(arm.get('n_randomized'))
                        if n_val > 0:
                            randomized_values.append(n_val)
                            total_randomized += n_val
                
                if total_randomized > 0:
                    participants['total_randomized'] = total_randomized
                    participants['randomized_source'] = f'calculated_from_arms ({"+".join(map(str, randomized_values))})'
                    print(f"📊 Calculated total randomized: {total_randomized} from arms: {randomized_values}")
            
            # Calculate total analyzed if missing
            if not participants.get('total_analyzed'):
                total_analyzed = 0
                analyzed_values = []
                
                for arm in interventions:
                    if arm.get('n_analyzed'):
                        # Enhanced number extraction
                        n_val = self._extract_number_from_string(arm.get('n_analyzed'))
                        if n_val > 0:
                            analyzed_values.append(n_val)
                            total_analyzed += n_val
                
                if total_analyzed > 0:
                    participants['total_analyzed'] = total_analyzed
                    participants['analyzed_source'] = f'calculated_from_arms ({"+".join(map(str, analyzed_values))})'
                    print(f"📊 Calculated total analyzed: {total_analyzed} from arms: {analyzed_values}")
        
        # 2. Extract missing standard deviations from tables
        if data.get('outcomes') and pdf_data.get('tables'):
            self._extract_missing_sds_from_tables(data['outcomes'], pdf_data['tables'])
        
        # 3. Validate and fix demographics
        if data.get('participants'):
            self._validate_demographics(data['participants'])
        
        # 4. Ensure all secondary outcomes are captured
        if data.get('outcomes') and pdf_data.get('tables'):
            self._extract_additional_secondary_outcomes(data['outcomes'], pdf_data['tables'])
        
        return data
    
    def _extract_number_from_string(self, value) -> int:
        """Extract number from strings like '264 patients', '266', 'NA', etc."""
        if not value or value in ['NA', 'N/A', 'NOT_REPORTED', 'NOT_SPECIFIED']:
            return 0
        
        # Convert to string and clean
        str_val = str(value).lower()
        
        # Remove common words and punctuation
        str_val = str_val.replace('patients', '').replace('subjects', '').replace('participants', '')
        str_val = str_val.replace(',', '').replace(' ', '').replace('n=', '')
        
        # Extract first number found
        import re
        numbers = re.findall(r'\d+', str_val)
        if numbers:
            return int(numbers[0])
        
        return 0
    
    def _extract_missing_sds_from_tables(self, outcomes: Dict, tables: List[Dict]):
        """Extract missing standard deviations from tables"""
        for outcome_type in ['primary', 'secondary']:
            if outcome_type not in outcomes:
                continue
                
            for outcome in outcomes[outcome_type]:
                if not outcome.get('timepoints'):
                    continue
                    
                for timepoint in outcome['timepoints']:
                    if not timepoint.get('results_by_arm'):
                        continue
                        
                    for arm_result in timepoint['results_by_arm']:
                        # If SD is missing but mean exists, try to find it in tables
                        if arm_result.get('mean') and not arm_result.get('sd'):
                            sd_value = self._find_sd_in_tables(arm_result.get('arm'), outcome.get('outcome_name'), tables)
                            if sd_value:
                                arm_result['sd'] = sd_value
                                arm_result['sd_source'] = 'table_extraction'
    
    def _find_sd_in_tables(self, arm_name: str, outcome_name: str, tables: List[Dict]) -> str:
        """Find standard deviation for specific arm and outcome in tables"""
        for table in tables:
            if not table.get('data'):
                continue
                
            table_text = str(table['data']).lower()
            
            # Look for patterns like "mean (SD)" or "mean ± SD"
            if arm_name and arm_name.lower() in table_text:
                # Common SD patterns
                sd_patterns = [
                    r'(\d+\.?\d*)\s*\(\s*(\d+\.?\d*)\s*\)',  # mean (sd)
                    r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)',        # mean ± sd
                    r'(\d+\.?\d*)\s*\+/-\s*(\d+\.?\d*)',     # mean +/- sd
                ]
                
                for pattern in sd_patterns:
                    matches = re.findall(pattern, table_text)
                    if matches:
                        # Return the SD part (second group)
                        return matches[0][1] if len(matches[0]) > 1 else None
        return None
    
    def _validate_demographics(self, participants: Dict):
        """Validate and fix unrealistic demographic values"""
        # Fix unrealistic age values
        if participants.get('age_mean'):
            age = float(participants['age_mean'])
            if age > 120 or age < 18:  # Unrealistic age
                # Try to find correct age in other sources
                participants['age_mean_error'] = f"Unrealistic age: {age}"
                # Could implement additional logic to find correct age
        
        # Validate gender percentages
        if participants.get('sex_male_percent'):
            male_pct = float(participants['sex_male_percent'])
            if male_pct > 100 or male_pct < 0:
                participants['sex_male_percent_error'] = f"Invalid percentage: {male_pct}"
    
    def _extract_additional_secondary_outcomes(self, outcomes: Dict, tables: List[Dict]):
        """Extract additional secondary outcomes that might have been missed"""
        # Look for common secondary outcomes in tables
        secondary_patterns = [
            'diastolic', 'dbp', 'diastolic blood pressure',
            'heart rate', 'pulse', 'adverse events',
            'quality of life', 'qol', 'side effects'
        ]
        
        for table in tables:
            if not table.get('data'):
                continue
                
            table_text = str(table['data']).lower()
            
            # Check if this table contains secondary outcome data
            for pattern in secondary_patterns:
                if pattern in table_text:
                    # This table likely contains secondary outcome data
                    # Add metadata about potential missing secondary outcomes
                    if 'secondary' not in outcomes:
                        outcomes['secondary'] = []
                    
                    # Add a placeholder for potentially missed secondary outcomes
                    found_pattern = {
                        'outcome_name': f'Potential secondary outcome: {pattern}',
                        'outcome_type': 'continuous',
                        'extraction_note': f'Found in table but not fully extracted: {pattern}',
                        'table_source': f"Table {table.get('page', 'unknown')}",
                        'requires_manual_review': True
                    }
                    
                    # Only add if not already present
                    if not any(outcome.get('outcome_name', '').lower().find(pattern) >= 0 
                             for outcome in outcomes['secondary']):
                        outcomes['secondary'].append(found_pattern)
    
    def _enhance_outcomes_with_heuristics(self, outcomes: Dict, heuristic_data: Dict):
        """Add statistical values from heuristic extraction to outcomes"""
        if not outcomes:
            return
            
        # Add p-values, CIs, effect estimates from pattern matching
        statistical_enhancements = {}
        
        if heuristic_data.get('p_values'):
            statistical_enhancements['p_values_found'] = heuristic_data['p_values']
        
        if heuristic_data.get('confidence_intervals'):
            statistical_enhancements['confidence_intervals'] = [
                f"{ci[0]}-{ci[1]}" for ci in heuristic_data['confidence_intervals']
            ]
        
        if heuristic_data.get('odds_ratios'):
            statistical_enhancements['odds_ratios'] = heuristic_data['odds_ratios']
            
        if heuristic_data.get('relative_risks'):
            statistical_enhancements['relative_risks'] = heuristic_data['relative_risks']
            
        if heuristic_data.get('hazard_ratios'):
            statistical_enhancements['hazard_ratios'] = heuristic_data['hazard_ratios']
        
        # Add to primary outcomes
        for outcome in outcomes.get('primary', []):
            if not outcome.get('statistical_analysis'):
                outcome['statistical_analysis'] = {}
            outcome['statistical_analysis'].update(statistical_enhancements)
    
    def _extract_demographics_from_tables(self, tables: List[Dict]) -> Dict:
        """Extract demographic data directly from tables"""
        demographics = {}
        
        for table in tables:
            if not table.get('data'):
                continue
                
            # Look for demographic/baseline tables
            table_text = str(table.get('data', '')).lower()
            
            # Common demographic indicators
            if any(term in table_text for term in ['age', 'gender', 'sex', 'demographics', 'baseline', 'characteristics']):
                # Extract age information
                age_patterns = [
                    r'age.*?(\d+\.?\d*)\s*\(\s*(\d+\.?\d*)\s*\)',  # Age: 65.5 (12.3)
                    r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)',  # 65.5 ± 12.3
                    r'mean.*?(\d+\.?\d*)',  # Mean age 65.5
                ]
                
                for pattern in age_patterns:
                    matches = re.findall(pattern, table_text, re.IGNORECASE)
                    if matches:
                        if len(matches[0]) == 2:  # Mean and SD
                            demographics['age_mean'] = float(matches[0][0])
                            demographics['age_sd'] = float(matches[0][1])
                        else:  # Just mean
                            demographics['age_mean'] = float(matches[0])
                        break
                
                # Extract gender distribution
                gender_patterns = [
                    r'male.*?(\d+)',
                    r'female.*?(\d+)',
                    r'men.*?(\d+)',
                    r'women.*?(\d+)'
                ]
                
                for pattern in gender_patterns:
                    matches = re.findall(pattern, table_text, re.IGNORECASE)
                    if matches:
                        if 'male' in pattern or 'men' in pattern:
                            demographics['male_count'] = int(matches[0])
                        else:
                            demographics['female_count'] = int(matches[0])
                
                # Add table source
                demographics['source'] = 'table_extraction'
                break
        
        return demographics
    
    def _extract_participant_flow_from_heuristics(self, heuristic_data: Dict) -> Dict:
        """Extract participant flow data from heuristic pattern matches"""
        flow_data = {}
        
        # Get the highest number for each flow category
        if heuristic_data.get('screened'):
            numbers = [int(x) for x in heuristic_data['screened'] if x.isdigit()]
            if numbers:
                flow_data['total_screened'] = max(numbers)
                flow_data['screened_source'] = 'heuristic_pattern'
        
        if heuristic_data.get('assessed_eligibility'):
            numbers = [int(x) for x in heuristic_data['assessed_eligibility'] if x.isdigit()]
            if numbers:
                # Use assessed for eligibility as screened if not already found
                if 'total_screened' not in flow_data:
                    flow_data['total_screened'] = max(numbers)
                    flow_data['screened_source'] = 'assessed_eligibility_pattern'
        
        if heuristic_data.get('randomized_total'):
            numbers = [int(x) for x in heuristic_data['randomized_total'] if x.isdigit()]
            if numbers:
                flow_data['total_randomized'] = max(numbers)
                flow_data['randomized_source'] = 'heuristic_pattern'
        
        if heuristic_data.get('analyzed_total'):
            numbers = [int(x) for x in heuristic_data['analyzed_total'] if x.isdigit()]
            if numbers:
                flow_data['total_analyzed'] = max(numbers)
                flow_data['analyzed_source'] = 'heuristic_pattern'
        
        if heuristic_data.get('enrolled'):
            numbers = [int(x) for x in heuristic_data['enrolled'] if x.isdigit()]
            if numbers:
                flow_data['total_enrolled'] = max(numbers)
                flow_data['enrolled_source'] = 'heuristic_pattern'
        
        return flow_data
    
    def _extract_participant_flow_from_tables(self, tables: List[Dict]) -> Dict:
        """Extract participant flow data from table content"""
        flow_data = {}
        
        for table in tables:
            table_text = str(table.get('content', '')).lower()
            
            # Look for CONSORT-style flow tables
            if any(word in table_text for word in ['screened', 'randomized', 'analyzed', 'enrolled', 'flow', 'consort']):
                # Extract numbers associated with flow terms
                flow_patterns = {
                    'total_screened': [r'screened[:\s]*(\d+)', r'assessed.*eligibility[:\s]*(\d+)'],
                    'total_randomized': [r'randomized[:\s]*(\d+)', r'allocated[:\s]*(\d+)'],
                    'total_analyzed': [r'analyzed[:\s]*(\d+)', r'completed[:\s]*(\d+)'],
                    'total_enrolled': [r'enrolled[:\s]*(\d+)', r'recruitment[:\s]*(\d+)']
                }
                
                for flow_key, patterns in flow_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, table_text, re.IGNORECASE)
                        if matches:
                            numbers = [int(x) for x in matches if x.isdigit()]
                            if numbers:
                                flow_data[flow_key] = max(numbers)
                                flow_data[f'{flow_key}_source'] = 'table_extraction'
                                break
        
        return flow_data
    
    def _calculate_confidence(self, final_data: Dict, pdf_data: Dict, heuristic_data: Dict) -> Dict:
        """Calculate meaningful confidence scores based on data completeness"""
        
        scores = {
            'overall': 0.0,
            'by_section': {}
        }
        
        # Study Identification (0-100)
        id_score = 0
        if final_data.get('study_identification', {}).get('title'):
            id_score += 30
        if final_data.get('study_identification', {}).get('authors'):
            id_score += 20
        if final_data.get('study_identification', {}).get('year'):
            id_score += 20
        if final_data.get('study_identification', {}).get('journal'):
            id_score += 15
        if final_data.get('study_identification', {}).get('trial_registration'):
            id_score += 15
        scores['by_section']['identification'] = id_score / 100.0
        
        # Study Design (0-100)
        design_score = 0
        sd = final_data.get('study_design', {})
        if sd.get('type') and sd.get('type') != 'NOT_REPORTED':
            design_score += 20
        if sd.get('blinding') and sd.get('blinding') != 'NOT_REPORTED':
            design_score += 20
        if sd.get('randomization_method') and sd.get('randomization_method') != 'NOT_REPORTED':
            design_score += 15
        if sd.get('duration_treatment') and sd.get('duration_treatment') != 'NOT_REPORTED':
            design_score += 15
        if sd.get('sites') and len(sd.get('sites', [])) > 0:
            design_score += 15
        if sd.get('country') and sd.get('country') != 'NOT_REPORTED':
            design_score += 15
        scores['by_section']['design'] = design_score / 100.0
        
        # Interventions (0-100)
        interventions = final_data.get('interventions', [])
        if interventions and len(interventions) >= 2:
            intervention_score = 40  # Has multiple arms
            complete_arms = sum(1 for i in interventions if i.get('arm_name') and i.get('dose'))
            intervention_score += (complete_arms / len(interventions)) * 60
            scores['by_section']['interventions'] = intervention_score / 100.0
        else:
            scores['by_section']['interventions'] = 0.0
        
        # Outcomes (0-100) - MOST CRITICAL
        primary_outcomes = final_data.get('outcomes', {}).get('primary', [])
        if primary_outcomes:
            outcome_score = 30  # Has outcomes
            # Check completeness
            complete_outcomes = 0
            for outcome in primary_outcomes:
                has_results = outcome.get('results_by_arm') and len(outcome.get('results_by_arm', [])) > 0
                has_effect = outcome.get('between_group_comparison') or outcome.get('effect_estimate')
                if has_results:
                    complete_outcomes += 0.5
                if has_effect:
                    complete_outcomes += 0.5
            outcome_score += (complete_outcomes / len(primary_outcomes)) * 70
            scores['by_section']['outcomes'] = outcome_score / 100.0
        else:
            scores['by_section']['outcomes'] = 0.0
        
        # Table extraction bonus
        if pdf_data.get('tables') and len(pdf_data['tables']) > 0:
            scores['tables_extracted'] = len(pdf_data['tables']) / 10.0  # 0-1 scale
        else:
            scores['tables_extracted'] = 0.0
        
        # Overall score - weighted average
        weights = {
            'identification': 0.1,
            'design': 0.2,
            'interventions': 0.2,
            'outcomes': 0.5  # Outcomes are MOST important
        }
        
        overall = 0
        for section, weight in weights.items():
            overall += scores['by_section'].get(section, 0) * weight
        
        scores['overall'] = overall
        
        return scores

# ==================== API ROUTES ====================
@app.route('/')
def index():
    """Serve the original frontend for testing"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.1.0',
        'features': [
            'pdf_extraction',
            'pdfplumber_tables',
            'ocr',
            'llm_extraction',
            'database_storage',
            'pdf_blob_storage',
            're_extraction',
            'confidence_scoring'
        ]
    })

@app.route('/api/extract', methods=['POST'])
def extract_trial_data():
    """Main extraction endpoint"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files accepted'}), 400
    
    try:
        file_content = file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check if already processed
        existing_study = Study.query.filter_by(pdf_hash=file_hash).first()
        if existing_study:
            return jsonify({
                'message': 'Study already extracted',
                'study_id': existing_study.id,
                'data': _serialize_study(existing_study)
            })
        
        # Save file temporarily for extraction
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            filepath = tmp_file.name
        
        # Extract data
        extractor = EnsembleExtractor()
        extracted_data, metadata = extractor.extract_comprehensive(filepath)
        
        # Save to database WITH PDF blob
        study = _save_to_database(extracted_data, metadata, file_hash, file_content, file.filename)
        
        # Store multi-source data for the saved study
        try:
            # Check if attributes exist before accessing them
            llm_data = getattr(extractor, '_last_llm_data', {})
            heuristic_data = getattr(extractor, '_last_heuristic_data', {})
            pdf_data = getattr(extractor, '_last_pdf_data', {})
            
            if llm_data or heuristic_data or pdf_data:
                extractor.store_multi_source_data(
                    study.id, 
                    llm_data, 
                    heuristic_data, 
                    pdf_data,
                    pdf_data.get('ocr_text', '')
                )
                metadata['multi_source_storage'] = 'success'
            else:
                metadata['multi_source_storage'] = 'skipped_no_data'
        except Exception as e:
            print(f"Warning: Could not store multi-source data: {e}")
            import traceback
            traceback.print_exc()
            metadata['multi_source_storage_error'] = str(e)
        
        # Clean up temp file
        os.unlink(filepath)
        
        # Get source data for the frontend
        sources_data = {}
        try:
            llm_data = getattr(extractor, '_last_llm_data', {})
            heuristic_data = getattr(extractor, '_last_heuristic_data', {})
            pdf_data = getattr(extractor, '_last_pdf_data', {})
            
            sources_data = {
                'llm': {
                    'data': llm_data,
                    'confidence': 0.8,
                    'icon': '🤖',
                    'available': bool(llm_data)
                },
                'heuristic': {
                    'data': heuristic_data,
                    'confidence': 0.7,
                    'icon': '🔍',
                    'available': bool(heuristic_data)
                },
                'pdfplumber': {
                    'data': {
                        'tables': pdf_data.get('tables', []),
                        'text_length': len(pdf_data.get('text', ''))
                    },
                    'confidence': 0.9,
                    'icon': '📊',
                    'available': bool(pdf_data)
                },
                'ocr': {
                    'data': {
                        'text_length': len(pdf_data.get('ocr_text', ''))
                    },
                    'confidence': 0.6,
                    'icon': '👁️',
                    'available': bool(pdf_data.get('ocr_text', ''))
                }
            }
        except Exception as e:
            print(f"Warning: Could not prepare source data: {e}")
            sources_data = {'error': 'Could not prepare source data'}
        
        return jsonify({
            'success': True,
            'study_id': study.id,
            'data': extracted_data,
            'metadata': metadata,
            'sources': sources_data  # NEW: Include source data for frontend
        })
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/studies/<int:study_id>/re-extract', methods=['POST'])
def re_extract_study(study_id):
    """Re-extract a study from stored PDF blob"""
    
    try:
        study = Study.query.get_or_404(study_id)
        
        if not study.pdf_blob:
            return jsonify({'error': 'No PDF stored for this study'}), 400
        
        # Write blob to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(study.pdf_blob)
            filepath = tmp_file.name
        
        # Delete existing related records
        Intervention.query.filter_by(study_id=study_id).delete()
        Outcome.query.filter_by(study_id=study_id).delete()
        SubgroupAnalysis.query.filter_by(study_id=study_id).delete()
        AdverseEvent.query.filter_by(study_id=study_id).delete()
        
        # Re-run extraction
        extractor = EnsembleExtractor()
        extracted_data, metadata = extractor.extract_comprehensive(filepath)
        
        # Update study record
        study.title = extracted_data.get('study_identification', {}).get('title')
        study.authors = extracted_data.get('study_identification', {}).get('authors', [])
        study.journal = extracted_data.get('study_identification', {}).get('journal')
        study.year = extracted_data.get('study_identification', {}).get('year')
        study.doi = extracted_data.get('study_identification', {}).get('doi')
        study.trial_registration = extracted_data.get('study_identification', {}).get('trial_registration')
        study.study_type = extracted_data.get('study_design', {}).get('type')
        study.blinding = extracted_data.get('study_design', {}).get('blinding')
        study.randomization = extracted_data.get('study_design', {}).get('randomization_method')
        study.duration = extracted_data.get('study_design', {}).get('duration_total')
        study.population_data = extracted_data.get('population')
        study.baseline_characteristics = extracted_data.get('population', {}).get('baseline_disease_severity')
        study.extraction_metadata = metadata
        study.confidence_scores = metadata.get('confidence_scores')
        study.extraction_date = datetime.utcnow()
        
        # Add interventions
        for intervention_data in extracted_data.get('interventions', []):
            intervention = Intervention(
                study_id=study.id,
                arm_name=intervention_data.get('arm_name'),
                n_randomized=_clean_int_value(intervention_data.get('n_randomized')),
                n_analyzed=_clean_int_value(intervention_data.get('n_analyzed_primary')),
                dose=intervention_data.get('dose'),
                frequency=intervention_data.get('frequency'),
                duration=intervention_data.get('duration')
            )
            db.session.add(intervention)
        
        # Add outcomes with multiple timepoints support
        for outcome_type in ['primary', 'secondary']:
            for outcome_data in extracted_data.get('outcomes', {}).get(outcome_type, []):
                # Create main outcome record
                outcome = Outcome(
                    study_id=study.id,
                    outcome_name=outcome_data.get('outcome_name'),
                    outcome_type=outcome_type,
                    outcome_category=outcome_data.get('outcome_type'),  # continuous/dichotomous/time_to_event
                    planned_timepoints=outcome_data.get('planned_timepoints'),
                    data_sources=outcome_data.get('data_source'),
                    additional_data=outcome_data  # Store full outcome data
                )
                db.session.add(outcome)
                db.session.flush()  # Get outcome ID
                
                # Handle multiple timepoints or single timepoint (backward compatibility)
                timepoints_data = outcome_data.get('timepoints', [])
                if not timepoints_data and outcome_data.get('timepoint'):
                    # Convert old single timepoint format to new format
                    timepoints_data = [{
                        'timepoint_name': outcome_data.get('timepoint'),
                        'timepoint_type': 'primary' if outcome_type == 'primary' else 'secondary',
                        'results_by_arm': outcome_data.get('results_by_arm'),
                        'between_group_comparison': outcome_data.get('between_group_comparison'),
                        'data_source': outcome_data.get('data_source')
                    }]
                
                # Add each timepoint
                for tp_idx, tp_data in enumerate(timepoints_data):
                    # Extract timepoint information
                    timepoint_name = tp_data.get('timepoint_name', '')
                    timepoint_value, timepoint_unit = _parse_timepoint(timepoint_name)
                    
                    # Extract statistical data from first arm (for overall values)
                    first_arm = tp_data.get('results_by_arm', [{}])[0] if tp_data.get('results_by_arm') else {}
                    
                    # Extract between-group comparison data
                    comparison = tp_data.get('between_group_comparison', {})
                    
                    timepoint = OutcomeTimepoint(
                        outcome_id=outcome.id,
                        study_id=study.id,
                        timepoint_name=timepoint_name,
                        timepoint_value=timepoint_value,
                        timepoint_unit=_safe_truncate(timepoint_unit),
                        timepoint_type=_safe_truncate(tp_data.get('timepoint_type', outcome_type)),
                        
                        # Sample size
                        n_analyzed=_extract_numeric_value(first_arm.get('n')),
                        
                        # Continuous outcome statistics
                        mean_value=_extract_numeric_value(first_arm.get('mean')),
                        sd_value=_extract_numeric_value(first_arm.get('sd')),
                        median_value=_extract_numeric_value(first_arm.get('median')),
                        iqr_lower=_extract_numeric_value(first_arm.get('iqr_lower')),
                        iqr_upper=_extract_numeric_value(first_arm.get('iqr_upper')),
                        ci_95_lower=_extract_numeric_value(first_arm.get('ci_95_lower')),
                        ci_95_upper=_extract_numeric_value(first_arm.get('ci_95_upper')),
                        
                        # Dichotomous outcome statistics
                        events=_extract_numeric_value(first_arm.get('events')),
                        total_participants=_extract_numeric_value(first_arm.get('total')),
                        
                        # Between-group comparison
                        effect_measure=comparison.get('effect_measure', '').split(' - Source:')[0].strip(),
                        effect_estimate=_extract_numeric_value(comparison.get('effect_estimate')),
                        effect_ci_lower=_extract_numeric_value(comparison.get('ci_95_lower')),
                        effect_ci_upper=_extract_numeric_value(comparison.get('ci_95_upper')),
                        p_value=_extract_p_value(comparison.get('p_value')),
                        p_value_text=_safe_truncate(comparison.get('p_value', '')),
                        
                        # Source tracking
                        data_source=tp_data.get('data_source', ''),
                        source_confidence=tp_data.get('source_confidence', 'medium'),
                        
                        # Store complete arm-by-arm results
                        results_by_arm=tp_data.get('results_by_arm'),
                        additional_statistics=comparison
                    )
                    db.session.add(timepoint)
                    
                    # Set primary timepoint reference
                    if tp_data.get('timepoint_type') == 'primary' or (outcome_type == 'primary' and tp_idx == 0):
                        outcome.primary_timepoint_id = timepoint.id
        
        # Add subgroups
        for subgroup_data in extracted_data.get('subgroup_analyses', []):
            subgroup = SubgroupAnalysis(
                study_id=study.id,
                subgroup_variable=subgroup_data.get('subgroup_variable'),
                subgroups=subgroup_data.get('subgroups'),
                p_interaction=subgroup_data.get('p_interaction')
            )
            db.session.add(subgroup)
        
        # Add adverse events
        for ae_data in extracted_data.get('adverse_events', []):
            adverse_event = AdverseEvent(
                study_id=study.id,
                event_name=ae_data.get('event_name'),
                severity=ae_data.get('severity_grade'),
                results_by_arm=ae_data.get('results_by_arm')
            )
            db.session.add(adverse_event)
        
        db.session.commit()
        
        # Clean up temp file
        os.unlink(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Study re-extracted successfully',
            'study_id': study.id,
            'data': _serialize_study(study),
            'metadata': metadata
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Re-extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/studies', methods=['GET'])
def get_studies():
    """Get all studies"""
    studies = Study.query.all()
    return jsonify({
        'studies': [_serialize_study(s) for s in studies]
    })

@app.route('/api/studies/<int:study_id>', methods=['GET'])
def get_study(study_id):
    """Get specific study"""
    study = Study.query.get_or_404(study_id)
    return jsonify(_serialize_study(study))

@app.route('/api/studies/<int:study_id>/sources', methods=['GET'])
def get_study_sources(study_id):
    """Get detailed source information for extracted data"""
    study = Study.query.get_or_404(study_id)
    
    # Extract source information from the extracted data
    sources_analysis = {
        'study_id': study_id,
        'title': study.title,
        'extraction_metadata': study.extraction_metadata or {},
        'source_breakdown': {},
        'missing_sources': [],
        'confidence_by_section': study.confidence_scores or {}
    }
    
    # Analyze sources from outcomes (which should have data_source fields)
    for outcome in study.outcomes:
        if outcome.results_by_arm:
            for arm_data in outcome.results_by_arm or []:
                if isinstance(arm_data, dict) and 'data_source' in arm_data:
                    sources_analysis['source_breakdown'][f'outcome_{outcome.id}'] = arm_data.get('data_source')
    
    return jsonify(sources_analysis)

# ==================== NEW MULTI-SOURCE DATA ENDPOINTS ====================

@app.route('/api/studies/<int:study_id>/sources/detailed', methods=['GET'])
def get_detailed_sources(study_id):
    """Get detailed source-specific extraction data"""
    study = Study.query.get_or_404(study_id)
    
    # Get all extraction sources for this study
    sources = ExtractionSource.query.filter_by(study_id=study_id).all()
    
    sources_data = {}
    for source in sources:
        sources_data[source.extraction_method] = {
            'raw_data': source.raw_data,
            'confidence_score': source.confidence_score,
            'quality_metrics': source.quality_metrics,
            'processing_time': source.processing_time,
            'extraction_timestamp': source.extraction_timestamp.isoformat() if source.extraction_timestamp else None,
            'success_status': source.success_status,
            'error_messages': source.error_messages
        }
    
    return jsonify({
        'study_id': study_id,
        'sources': sources_data,
        'total_sources': len(sources)
    })

@app.route('/api/studies/<int:study_id>/data-elements', methods=['GET'])
def get_data_elements(study_id):
    """Get individual data elements with source attribution"""
    study = Study.query.get_or_404(study_id)
    
    data_elements = DataElement.query.filter_by(study_id=study_id).all()
    
    elements_by_type = {}
    conflicts = []
    
    for element in data_elements:
        if element.element_type not in elements_by_type:
            elements_by_type[element.element_type] = []
        
        element_data = {
            'id': element.id,
            'element_name': element.element_name,
            'element_path': element.element_path,
            'sources': {
                'pdfplumber': {
                    'value': element.pdfplumber_value,
                    'confidence': element.pdfplumber_confidence,
                    'location': element.pdfplumber_source_location
                },
                'ocr': {
                    'value': element.ocr_value,
                    'confidence': element.ocr_confidence,
                    'location': element.ocr_source_location
                },
                'llm': {
                    'value': element.llm_value,
                    'confidence': element.llm_confidence,
                    'location': element.llm_source_location
                },
                'heuristic': {
                    'value': element.heuristic_value,
                    'confidence': element.heuristic_confidence,
                    'location': element.heuristic_source_location
                }
            },
            'selected_source': element.selected_source,
            'selected_value': element.selected_value,
            'user_notes': element.user_notes,
            'is_validated': element.is_validated,
            'needs_review': element.needs_review,
            'conflicting_sources': element.conflicting_sources
        }
        
        elements_by_type[element.element_type].append(element_data)
        
        if element.conflicting_sources:
            conflicts.append(element_data)
    
    return jsonify({
        'study_id': study_id,
        'elements_by_type': elements_by_type,
        'total_elements': len(data_elements),
        'conflicts': conflicts,
        'conflict_count': len(conflicts)
    })

@app.route('/api/studies/<int:study_id>/data-elements/<int:element_id>/select', methods=['POST'])
def select_data_source(study_id, element_id):
    """Set user selection for a data element"""
    data = request.get_json()
    
    if not data or 'source' not in data or 'value' not in data:
        return jsonify({'error': 'Missing source or value'}), 400
    
    element = DataElement.query.filter_by(id=element_id, study_id=study_id).first_or_404()
    
    element.selected_source = data['source']
    element.selected_value = data['value']
    element.user_notes = data.get('notes', '')
    element.selection_timestamp = datetime.utcnow()
    element.is_validated = True
    element.needs_review = False
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'element_id': element_id,
        'selected_source': element.selected_source,
        'selected_value': element.selected_value
    })

@app.route('/api/studies/<int:study_id>/auto-select', methods=['POST'])
def auto_select_recommended(study_id):
    """Auto-select recommended sources based on confidence and priority"""
    data = request.get_json()
    confidence_threshold = data.get('confidence_threshold', 0.7) if data else 0.7
    
    elements = DataElement.query.filter_by(study_id=study_id).all()
    selections_made = 0
    
    for element in elements:
        if element.is_validated:
            continue  # Skip already validated elements
        
        # Get all source values with confidence
        sources = [
            ('pdfplumber', element.pdfplumber_value, element.pdfplumber_confidence or 0),
            ('heuristic', element.heuristic_value, element.heuristic_confidence or 0),
            ('ocr', element.ocr_value, element.ocr_confidence or 0),
            ('llm', element.llm_value, element.llm_confidence or 0)
        ]
        
        # Filter out None/empty values and sort by priority and confidence
        valid_sources = [(s, v, c) for s, v, c in sources if v and c >= confidence_threshold]
        
        if not valid_sources:
            continue
        
        # Sort by priority (based on SOURCE_PRIORITY_DEFAULT) and confidence
        priority_map = {source: i for i, source in enumerate(SOURCE_PRIORITY_DEFAULT)}
        valid_sources.sort(key=lambda x: (priority_map.get(x[0], 99), -x[2]))
        
        # Select the best source
        best_source, best_value, best_confidence = valid_sources[0]
        
        element.selected_source = best_source
        element.selected_value = best_value
        element.selection_timestamp = datetime.utcnow()
        element.is_validated = True
        element.needs_review = False
        
        selections_made += 1
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'selections_made': selections_made,
        'confidence_threshold': confidence_threshold
    })

@app.route('/api/studies/<int:study_id>/conflicts', methods=['GET'])
def get_extraction_conflicts(study_id):
    """Get all extraction conflicts for a study"""
    conflicts = ExtractionConflict.query.filter_by(study_id=study_id).all()
    
    conflicts_data = []
    for conflict in conflicts:
        conflicts_data.append({
            'id': conflict.id,
            'data_element_id': conflict.data_element_id,
            'conflict_type': conflict.conflict_type,
            'severity': conflict.severity,
            'source_a': conflict.source_a,
            'value_a': conflict.value_a,
            'confidence_a': conflict.confidence_a,
            'source_b': conflict.source_b,
            'value_b': conflict.value_b,
            'confidence_b': conflict.confidence_b,
            'resolution_status': conflict.resolution_status,
            'resolution_method': conflict.resolution_method,
            'resolution_notes': conflict.resolution_notes,
            'created_date': conflict.created_date.isoformat() if conflict.created_date else None
        })
    
    return jsonify({
        'study_id': study_id,
        'conflicts': conflicts_data,
        'total_conflicts': len(conflicts_data),
        'pending_conflicts': len([c for c in conflicts_data if c['resolution_status'] == 'pending'])
    })

@app.route('/api/studies/<int:study_id>/finalize', methods=['POST'])
def finalize_study_selections(study_id):
    """Finalize all user selections and generate final extraction data"""
    study = Study.query.get_or_404(study_id)
    
    # Get all validated data elements
    elements = DataElement.query.filter_by(study_id=study_id, is_validated=True).all()
    
    # Rebuild the study data using selected sources
    final_data = _rebuild_study_from_selections(study, elements)
    
    # Update the study record
    study.extraction_metadata = study.extraction_metadata or {}
    study.extraction_metadata['finalized'] = True
    study.extraction_metadata['finalization_timestamp'] = datetime.utcnow().isoformat()
    study.extraction_metadata['validated_elements'] = len(elements)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'study_id': study_id,
        'finalized_data': final_data,
        'validated_elements': len(elements)
    })

def _rebuild_study_from_selections(study, elements):
    """Rebuild study data structure from user-selected sources"""
    final_data = {
        'study_identification': {},
        'study_design': {},
        'population': {},
        'interventions': [],
        'outcomes': {'primary': [], 'secondary': []}
    }
    
    # Group elements by type and rebuild data structure
    for element in elements:
        if element.selected_value:
            # Parse element path and set value in final_data structure
            path_parts = element.element_path.split('.') if element.element_path else [element.element_name]
            _set_nested_value(final_data, path_parts, element.selected_value)
    
    return final_data

def _set_nested_value(data_dict, path_parts, value):
    """Set a value in a nested dictionary using a path"""
    current = data_dict
    for part in path_parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[path_parts[-1]] = value

@app.route('/api/studies/<int:study_id>', methods=['DELETE'])
def delete_study(study_id):
    """Delete a study and all related records"""
    print(f"\n{'='*60}")
    print(f"🗑️  DELETE REQUEST RECEIVED for study_id={study_id}")
    print(f"{'='*60}\n")
    
    try:
        study = Study.query.get_or_404(study_id)
        print(f"✓ Found study: {study.title[:50] if study.title else 'No title'}...")
        
        # Delete related records first to avoid foreign key constraint violations
        print("\n🧹 Cleaning up related records...")
        
        # Delete multi-source extraction data
        count = ExtractionSource.query.filter_by(study_id=study_id).delete()
        print(f"  - Deleted {count} ExtractionSource records")
        
        count = DataElement.query.filter_by(study_id=study_id).delete()
        print(f"  - Deleted {count} DataElement records")
        
        count = ExtractionConflict.query.filter_by(study_id=study_id).delete()
        print(f"  - Deleted {count} ExtractionConflict records")
        
        # Delete outcome timepoints (not automatically cascaded from Study)
        count = OutcomeTimepoint.query.filter_by(study_id=study_id).delete()
        print(f"  - Deleted {count} OutcomeTimepoint records")
        
        # Now delete the study (this will cascade to interventions, outcomes, subgroups, adverse_events)
        print(f"\n🗑️  Deleting study record...")
        db.session.delete(study)
        
        db.session.commit()
        print(f"\n✅ Study {study_id} deleted successfully!\n")
        
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        print(f"\n❌ DELETE FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export/<int:study_id>', methods=['GET'])
def export_study(study_id):
    """Export study data as JSON"""
    study = Study.query.get_or_404(study_id)
    data = _serialize_study(study)
    
    return jsonify(data)

# ==================== HELPER FUNCTIONS ====================

def _clean_int_value(value):
    """Convert 'Not Reported' strings to None for integer fields and extract number from source citations"""
    if value in [None, '', 'Not Reported', 'NOT_REPORTED', 'N/A', 'Not reported', 'Not applicable']:
        return None
    if isinstance(value, str):
        # Handle source citations like "2025 - Source: page 1"
        if " - Source:" in value:
            actual_value = value.split(" - Source:")[0].strip()
        else:
            actual_value = value
        try:
            return int(actual_value)
        except (ValueError, TypeError):
            return None
    return value

def _extract_value_and_source(value_with_source):
    """Extract the actual value and source citation separately"""
    if isinstance(value_with_source, str) and " - Source:" in value_with_source:
        parts = value_with_source.split(" - Source:", 1)
        return parts[0].strip(), f"Source: {parts[1].strip()}"
    return value_with_source, None

def _extract_numeric_value(value_with_source):
    """Extract numeric value from string that may contain source citations"""
    if value_with_source is None:
        return None
    
    if isinstance(value_with_source, (int, float)):
        return float(value_with_source)
    
    if isinstance(value_with_source, str):
        # Remove source citations
        clean_value = value_with_source.split(" - Source:")[0].strip()
        
        # Handle "NOT_SPECIFIED", "NOT_FOUND", etc.
        if clean_value.upper() in ['NOT_SPECIFIED', 'NOT_FOUND', 'NOT_REPORTED', 'N/A', '']:
            return None
            
        # Try to extract number
        try:
            return float(clean_value)
        except (ValueError, TypeError):
            # Try to extract first number from string
            import re
            numbers = re.findall(r'-?\d+\.?\d*', clean_value)
            if numbers:
                return float(numbers[0])
            return None
    
    return None

def _extract_p_value(p_value_string):
    """Extract numeric p-value from string, handling <0.001, etc."""
    if p_value_string is None:
        return None
    
    clean_p = str(p_value_string).split(" - Source:")[0].strip()
    
    # Handle special cases
    if clean_p.upper() in ['NS', 'NOT SIGNIFICANT', 'NOT_SPECIFIED', 'NOT_FOUND']:
        return None
    
    # Extract numeric value
    import re
    # Handle cases like "<0.001", "p=0.05", "0.001"
    p_match = re.search(r'([<>=]?\s*)?(\d+\.?\d*(?:[eE]-?\d+)?)', clean_p)
    if p_match:
        try:
            return float(p_match.group(2))
        except (ValueError, TypeError):
            return None
    
    return None

def _safe_truncate(value, max_length=50):
    """Safely truncate string values for database fields with length limits"""
    if not value:
        return value
    if not isinstance(value, str):
        value = str(value)
    
    # Remove source information if it makes the string too long
    if " - Source:" in value and len(value) > max_length:
        value = value.split(" - Source:")[0].strip()
    
    # Truncate if still too long
    if len(value) > max_length:
        value = value[:max_length-3] + "..."
    
    return value

def _parse_timepoint(timepoint_name):
    """Parse timepoint string to extract numeric value and unit"""
    if not timepoint_name or not isinstance(timepoint_name, str):
        return None, None
    
    clean_name = timepoint_name.split(" - Source:")[0].strip().lower()
    
    import re
    # Look for patterns like "12 weeks", "6 months", "24 hours", "1 year"
    patterns = [
        r'(\d+\.?\d*)\s*(week|weeks|wk|wks)',
        r'(\d+\.?\d*)\s*(month|months|mo|mos)',
        r'(\d+\.?\d*)\s*(day|days|d)',
        r'(\d+\.?\d*)\s*(year|years|yr|yrs)',
        r'(\d+\.?\d*)\s*(hour|hours|hr|hrs|h)',
        r'(\d+\.?\d*)\s*(minute|minutes|min|mins)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean_name)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            # Normalize units
            if unit in ['week', 'weeks', 'wk', 'wks']:
                return value, 'weeks'
            elif unit in ['month', 'months', 'mo', 'mos']:
                return value, 'months'
            elif unit in ['day', 'days', 'd']:
                return value, 'days'
            elif unit in ['year', 'years', 'yr', 'yrs']:
                return value, 'years'
            elif unit in ['hour', 'hours', 'hr', 'hrs', 'h']:
                return value, 'hours'
            elif unit in ['minute', 'minutes', 'min', 'mins']:
                return value, 'minutes'
    
    # Handle special cases
    if 'baseline' in clean_name:
        return 0, 'baseline'
    elif 'end of treatment' in clean_name or 'eot' in clean_name:
        return None, 'end_of_treatment'
    elif 'follow' in clean_name:
        return None, 'follow_up'
    
    return None, None

def _save_to_database(data: Dict, metadata: Dict, pdf_hash: str, pdf_blob: bytes, pdf_filename: str) -> Study:
    """Save extracted data to database"""
    
    # Extract clean values from source citations
    title_clean, title_source = _extract_value_and_source(data.get('study_identification', {}).get('title'))
    year_clean = _clean_int_value(data.get('study_identification', {}).get('year'))
    journal_clean, journal_source = _extract_value_and_source(data.get('study_identification', {}).get('journal'))
    doi_clean, doi_source = _extract_value_and_source(data.get('study_identification', {}).get('doi'))
    trial_reg_clean, trial_reg_source = _extract_value_and_source(data.get('study_identification', {}).get('trial_registration'))
    
    # Clean other fields that might contain source citations
    study_type_clean, study_type_source = _extract_value_and_source(data.get('study_design', {}).get('type'))
    blinding_clean, blinding_source = _extract_value_and_source(data.get('study_design', {}).get('blinding'))
    randomization_clean, randomization_source = _extract_value_and_source(data.get('study_design', {}).get('randomization_method'))
    duration_clean, duration_source = _extract_value_and_source(data.get('study_design', {}).get('duration_total'))
    
    # Create study with clean values
    study = Study(
        pdf_hash=pdf_hash,
        pdf_blob=pdf_blob,
        pdf_filename=pdf_filename,
        title=title_clean,
        authors=data.get('study_identification', {}).get('authors', []),
        journal=journal_clean,
        year=year_clean,
        doi=doi_clean,
        trial_registration=trial_reg_clean,
        study_type=study_type_clean,
        blinding=blinding_clean,
        randomization=randomization_clean,
        duration=duration_clean,
        population_data=data.get('population'),
        baseline_characteristics=data.get('population', {}).get('baseline_disease_severity'),
        extraction_metadata=metadata,
        confidence_scores=metadata.get('confidence_scores'),
        source_tracking={
            'title_source': title_source,
            'journal_source': journal_source, 
            'doi_source': doi_source,
            'trial_registration_source': trial_reg_source,
            'study_type_source': study_type_source,
            'blinding_source': blinding_source,
            'randomization_source': randomization_source,
            'duration_source': duration_source,
            'raw_extraction': data  # Store the original data with sources
        }
    )
    
    db.session.add(study)
    db.session.flush()
    
    # Add interventions
    for intervention_data in data.get('interventions', []):
        intervention = Intervention(
            study_id=study.id,
            arm_name=intervention_data.get('arm_name'),
            n_randomized=_clean_int_value(intervention_data.get('n_randomized')),
            n_analyzed=_clean_int_value(intervention_data.get('n_analyzed_primary')),
            dose=intervention_data.get('dose'),
            frequency=intervention_data.get('frequency'),
            duration=intervention_data.get('duration')
        )
        db.session.add(intervention)
    
        # Add outcomes with multiple timepoints support
        for outcome_type in ['primary', 'secondary']:
            for outcome_data in data.get('outcomes', {}).get(outcome_type, []):
                # Create main outcome record
                outcome = Outcome(
                    study_id=study.id,
                    outcome_name=outcome_data.get('outcome_name'),
                    outcome_type=outcome_type,
                    outcome_category=outcome_data.get('outcome_type'),  # continuous/dichotomous/time_to_event
                    planned_timepoints=outcome_data.get('planned_timepoints'),
                    data_sources=outcome_data.get('data_source'),
                    additional_data=outcome_data  # Store full outcome data
                )
                db.session.add(outcome)
                db.session.flush()  # Get outcome ID
                
                # Handle multiple timepoints or single timepoint (backward compatibility)
                timepoints_data = outcome_data.get('timepoints', [])
                if not timepoints_data and outcome_data.get('timepoint'):
                    # Convert old single timepoint format to new format
                    timepoints_data = [{
                        'timepoint_name': outcome_data.get('timepoint'),
                        'timepoint_type': 'primary' if outcome_type == 'primary' else 'secondary',
                        'results_by_arm': outcome_data.get('results_by_arm'),
                        'between_group_comparison': outcome_data.get('between_group_comparison'),
                        'data_source': outcome_data.get('data_source')
                    }]
                
                # Add each timepoint
                for tp_idx, tp_data in enumerate(timepoints_data):
                    # Extract timepoint information
                    timepoint_name = tp_data.get('timepoint_name', '')
                    timepoint_value, timepoint_unit = _parse_timepoint(timepoint_name)
                    
                    # Extract statistical data from first arm (for overall values)
                    first_arm = tp_data.get('results_by_arm', [{}])[0] if tp_data.get('results_by_arm') else {}
                    
                    # Extract between-group comparison data
                    comparison = tp_data.get('between_group_comparison', {})
                    
                    timepoint = OutcomeTimepoint(
                        outcome_id=outcome.id,
                        study_id=study.id,
                        timepoint_name=timepoint_name,
                        timepoint_value=timepoint_value,
                        timepoint_unit=_safe_truncate(timepoint_unit),
                        timepoint_type=_safe_truncate(tp_data.get('timepoint_type', outcome_type)),
                        
                        # Sample size
                        n_analyzed=_extract_numeric_value(first_arm.get('n')),
                        
                        # Continuous outcome statistics
                        mean_value=_extract_numeric_value(first_arm.get('mean')),
                        sd_value=_extract_numeric_value(first_arm.get('sd')),
                        median_value=_extract_numeric_value(first_arm.get('median')),
                        iqr_lower=_extract_numeric_value(first_arm.get('iqr_lower')),
                        iqr_upper=_extract_numeric_value(first_arm.get('iqr_upper')),
                        ci_95_lower=_extract_numeric_value(first_arm.get('ci_95_lower')),
                        ci_95_upper=_extract_numeric_value(first_arm.get('ci_95_upper')),
                        
                        # Dichotomous outcome statistics
                        events=_extract_numeric_value(first_arm.get('events')),
                        total_participants=_extract_numeric_value(first_arm.get('total')),
                        
                        # Between-group comparison
                        effect_measure=comparison.get('effect_measure', '').split(' - Source:')[0].strip(),
                        effect_estimate=_extract_numeric_value(comparison.get('effect_estimate')),
                        effect_ci_lower=_extract_numeric_value(comparison.get('ci_95_lower')),
                        effect_ci_upper=_extract_numeric_value(comparison.get('ci_95_upper')),
                        p_value=_extract_p_value(comparison.get('p_value')),
                        p_value_text=_safe_truncate(comparison.get('p_value', '')),
                        
                        # Source tracking
                        data_source=tp_data.get('data_source', ''),
                        source_confidence=tp_data.get('source_confidence', 'medium'),
                        
                        # Store complete arm-by-arm results
                        results_by_arm=tp_data.get('results_by_arm'),
                        additional_statistics=comparison
                    )
                    db.session.add(timepoint)
                    
                    # Set primary timepoint reference
                    if tp_data.get('timepoint_type') == 'primary' or (outcome_type == 'primary' and tp_idx == 0):
                        outcome.primary_timepoint_id = timepoint.id    # Add subgroups
    for subgroup_data in data.get('subgroup_analyses', []):
        subgroup = SubgroupAnalysis(
            study_id=study.id,
            subgroup_variable=subgroup_data.get('subgroup_variable'),
            subgroups=subgroup_data.get('subgroups'),
            p_interaction=subgroup_data.get('p_interaction')
        )
        db.session.add(subgroup)
    
    # Add adverse events
    for ae_data in data.get('adverse_events', []):
        adverse_event = AdverseEvent(
            study_id=study.id,
            event_name=ae_data.get('event_name'),
            severity=ae_data.get('severity_grade'),
            results_by_arm=ae_data.get('results_by_arm')
        )
        db.session.add(adverse_event)
    
    db.session.commit()
    
    return study

def _serialize_study(study: Study) -> Dict:
    """Convert study object to dictionary"""
    
    # Get metadata from database - handle both during extraction and from DB
    db_metadata = study.extraction_metadata if hasattr(study, 'extraction_metadata') and study.extraction_metadata else {}
    db_confidence = study.confidence_scores if hasattr(study, 'confidence_scores') and study.confidence_scores else {}
    
    serialized = {
        'id': study.id,
        'study_identification': {
            'title': study.title,
            'authors': study.authors,
            'journal': study.journal,
            'year': study.year,
            'doi': study.doi,
            'trial_registration': study.trial_registration
        },
        'study_design': {
            'type': study.study_type,
            'blinding': study.blinding,
            'randomization': study.randomization,
            'duration': study.duration
        },
        'population': study.population_data,
        'interventions': [
            {
                'arm_name': i.arm_name,
                'n_randomized': i.n_randomized,
                'n_analyzed': i.n_analyzed,
                'dose': i.dose,
                'frequency': i.frequency,
                'duration': i.duration
            }
            for i in study.interventions
        ],
        'outcomes': {
            'primary': [
                {
                    'outcome_name': o.outcome_name,
                    'outcome_category': o.outcome_category,
                    'planned_timepoints': o.planned_timepoints,
                    'timepoints': [
                        {
                            'timepoint_name': tp.timepoint_name,
                            'timepoint_value': tp.timepoint_value,
                            'timepoint_unit': tp.timepoint_unit,
                            'timepoint_type': tp.timepoint_type,
                            'n_analyzed': tp.n_analyzed,
                            'mean_value': tp.mean_value,
                            'sd_value': tp.sd_value,
                            'median_value': tp.median_value,
                            'ci_95_lower': tp.ci_95_lower,
                            'ci_95_upper': tp.ci_95_upper,
                            'events': tp.events,
                            'total_participants': tp.total_participants,
                            'effect_measure': tp.effect_measure,
                            'effect_estimate': tp.effect_estimate,
                            'effect_ci_lower': tp.effect_ci_lower,
                            'effect_ci_upper': tp.effect_ci_upper,
                            'p_value': tp.p_value,
                            'data_source': tp.data_source,
                            'source_confidence': tp.source_confidence
                        }
                        for tp in o.timepoints
                    ]
                }
                for o in study.outcomes if o.outcome_type == 'primary'
            ],
            'secondary': [
                {
                    'outcome_name': o.outcome_name,
                    'outcome_category': o.outcome_category,
                    'planned_timepoints': o.planned_timepoints,
                    'timepoints': [
                        {
                            'timepoint_name': tp.timepoint_name,
                            'timepoint_value': tp.timepoint_value,
                            'timepoint_unit': tp.timepoint_unit,
                            'timepoint_type': tp.timepoint_type,
                            'n_analyzed': tp.n_analyzed,
                            'mean_value': tp.mean_value,
                            'sd_value': tp.sd_value,
                            'median_value': tp.median_value,
                            'ci_95_lower': tp.ci_95_lower,
                            'ci_95_upper': tp.ci_95_upper,
                            'events': tp.events,
                            'total_participants': tp.total_participants,
                            'effect_measure': tp.effect_measure,
                            'effect_estimate': tp.effect_estimate,
                            'effect_ci_lower': tp.effect_ci_lower,
                            'effect_ci_upper': tp.effect_ci_upper,
                            'p_value': tp.p_value,
                            'data_source': tp.data_source,
                            'source_confidence': tp.source_confidence
                        }
                        for tp in o.timepoints
                    ]
                }
                for o in study.outcomes if o.outcome_type == 'secondary'
            ]
        },
        'subgroup_analyses': [
            {
                'subgroup_variable': s.subgroup_variable,
                'subgroups': s.subgroups,
                'p_interaction': s.p_interaction
            }
            for s in study.subgroups
        ],
        'adverse_events': [
            {
                'event_name': ae.event_name,
                'severity': ae.severity,
                'results_by_arm': ae.results_by_arm
            }
            for ae in study.adverse_events
        ],
        'extraction_metadata': db_metadata,
        'confidence_scores': db_confidence,
        'extraction_date': study.extraction_date.isoformat() if study.extraction_date else None,
        'has_pdf': study.pdf_blob is not None if hasattr(study, 'pdf_blob') else False,
        'pdf_filename': study.pdf_filename if hasattr(study, 'pdf_filename') else None
    }
    
    return serialized

# Helper function for parsing arm-specific effect estimates
def _parse_arm_specific_effect(combined_effect_str: str, arm_name: str, all_arms: list) -> dict:
    """
    Parse combined effect estimate strings and extract arm-specific values.
    
    Example input: "-5.5 percentage points (Orforglipron 6 mg), -6.3 percentage points (Orforglipron 12 mg), -9.1 percentage points (Orforglipron 36 mg)"
    Returns: For "Orforglipron 6 mg" arm, returns "-5.5 percentage points"
    
    Also handles CI format: "-6.5 (Orforglipron 6 mg), -7.3 (Orforglipron 12 mg), -10.1 (Orforglipron 36 mg)"
    """
    import re
    
    if not combined_effect_str or not arm_name:
        return {'estimate': combined_effect_str, 'ci_lower': None, 'ci_upper': None}
    
    combined_str = str(combined_effect_str)
    
    # Try to find arm-specific value in the combined string
    # Pattern 1: "value (arm_name)" or "value arm_name"
    # Pattern 2: Handle partial arm name matches (e.g., "6 mg", "12 mg", "36 mg")
    
    # Extract key part of arm name (dose)
    arm_key = arm_name.lower().strip()
    
    # Try different patterns to find the value
    patterns = [
        rf'([-+]?\d+\.?\d*)\s*(?:percentage points?|mg|cm|%|mmHg)?\s*\([^)]*{re.escape(arm_key)}[^)]*\)',
        rf'([-+]?\d+\.?\d*)\s*\([^)]*{re.escape(arm_key)}[^)]*\)',
        rf'([-+]?\d+\.?\d*)\s+{re.escape(arm_key)}'
    ]
    
    # Also try matching just the dose part (e.g., "6 mg", "12 mg")
    if 'mg' in arm_key or 'placebo' in arm_key.lower():
        dose_match = re.search(r'(\d+)\s*mg|placebo', arm_key.lower())
        if dose_match:
            dose_key = dose_match.group(0)
            patterns.extend([
                rf'([-+]?\d+\.?\d*)\s*(?:percentage points?|mg|cm|%|mmHg)?\s*\([^)]*{re.escape(dose_key)}[^)]*\)',
                rf'([-+]?\d+\.?\d*)\s*\([^)]*{re.escape(dose_key)}[^)]*\)'
            ])
    
    for pattern in patterns:
        match = re.search(pattern, combined_str, re.IGNORECASE)
        if match:
            return {
                'estimate': match.group(1),
                'ci_lower': None,
                'ci_upper': None
            }
    
    # If no match found, return the original combined string
    return {
        'estimate': combined_effect_str,
        'ci_lower': None,
        'ci_upper': None
    }

# Export endpoints - FIXED COMPREHENSIVE EXPORT
@app.route('/api/export/csv/<int:study_id>', methods=['GET'])
def export_csv(study_id):
    """Export study data as comprehensive Excel file with ALL extracted data"""
    study = Study.query.get_or_404(study_id)
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Study Overview (Complete identification)
        study_info = pd.DataFrame([{
            'Study_ID': study.id,
            'Title': study.title,
            'Authors': ', '.join(study.authors) if study.authors else 'N/A',
            'Journal': study.journal,
            'Year': study.year,
            'DOI': study.doi,
            'Trial_Registration': study.trial_registration,
            'Study_Type': study.study_type,
            'Blinding': study.blinding,
            'Randomization_Method': study.randomization,
            'Study_Duration': study.duration,
            'PDF_Filename': study.pdf_filename if hasattr(study, 'pdf_filename') else 'N/A',
            'Extraction_Date': study.extraction_date.strftime('%Y-%m-%d %H:%M') if study.extraction_date else 'N/A',
            'Overall_Confidence_%': round(study.confidence_scores.get('overall', 0) * 100, 1) if study.confidence_scores else 0
        }])
        study_info.to_excel(writer, sheet_name='Study Overview', index=False)
        
        # Sheet 2: Population & Eligibility
        if study.population_data:
            pop_df = pd.DataFrame([{
                'Total_Screened': study.population_data.get('total_screened'),
                'Total_Randomized': study.population_data.get('total_randomized'),
                'Total_Analyzed_ITT': study.population_data.get('total_analyzed_itt'),
                'Total_Analyzed_PP': study.population_data.get('total_analyzed_pp'),
                'Age_Mean': study.population_data.get('age_mean'),
                'Age_SD': study.population_data.get('age_sd'),
                'Age_Median': study.population_data.get('age_median'),
                'Age_Range': study.population_data.get('age_range'),
                'Male_N': study.population_data.get('sex_male_n'),
                'Male_Percent': study.population_data.get('sex_male_percent'),
                'Inclusion_Criteria': '; '.join(study.population_data.get('inclusion_criteria', [])) if isinstance(study.population_data.get('inclusion_criteria'), list) else study.population_data.get('inclusion_criteria'),
                'Exclusion_Criteria': '; '.join(study.population_data.get('exclusion_criteria', [])) if isinstance(study.population_data.get('exclusion_criteria'), list) else study.population_data.get('exclusion_criteria')
            }])
            pop_df.to_excel(writer, sheet_name='Population', index=False)
        
        # Sheet 3: Interventions/Arms (CONSORT Flow)
        if study.interventions:
            int_data = []
            for i, intervention in enumerate(study.interventions, 1):
                # Calculate dropouts safely
                dropout_n = None
                dropout_percent = None
                if intervention.n_randomized and intervention.n_analyzed:
                    try:
                        n_rand = int(intervention.n_randomized) if intervention.n_randomized else None
                        n_anal = int(intervention.n_analyzed) if intervention.n_analyzed else None
                        if n_rand and n_anal and n_rand > 0:
                            dropout_n = n_rand - n_anal
                            dropout_percent = round((dropout_n / n_rand * 100), 1)
                    except (ValueError, TypeError):
                        pass
                
                int_data.append({
                    'Arm_Number': i,
                    'Arm_Name': intervention.arm_name,
                    'N_Randomized': intervention.n_randomized,
                    'N_Analyzed': intervention.n_analyzed,
                    'Drug/Intervention': intervention.arm_name,  # Parsing from arm name
                    'Dose': intervention.dose,
                    'Frequency': intervention.frequency,
                    'Duration': intervention.duration,
                    'Dropout_N': dropout_n,
                    'Dropout_%': dropout_percent
                })
            pd.DataFrame(int_data).to_excel(writer, sheet_name='Interventions', index=False)
        
        # Sheet 4: Primary Outcomes (Full Statistical Detail) - ONE ROW PER ARM
        primary_outcomes = [o for o in study.outcomes if o.outcome_type == 'primary']
        if primary_outcomes:
            outcome_rows = []
            seen_outcomes = set()  # Track processed outcomes to avoid duplicates
            
            for outcome in primary_outcomes:
                # Skip if outcome name is empty, placeholder, or already processed
                if not outcome.outcome_name:
                    continue
                if outcome.outcome_name.startswith('Potential'):
                    continue
                if outcome.outcome_name in seen_outcomes:
                    continue
                seen_outcomes.add(outcome.outcome_name)
                
                # Get timepoint info
                timepoint_str = outcome.planned_timepoints or "Not specified"
                if outcome.timepoints:
                    primary_tp = next((tp for tp in outcome.timepoints if tp.timepoint_type == 'primary'), outcome.timepoints[0])
                    timepoint_str = f"{primary_tp.timepoint_value} {primary_tp.timepoint_unit}" if primary_tp else timepoint_str
                
                # Get arm-by-arm results from additional_data JSON field
                results_by_arm = []
                if outcome.additional_data:
                    # Check timepoints array first
                    timepoints_data = outcome.additional_data.get('timepoints', [])
                    if timepoints_data:
                        for tp in timepoints_data:
                            if tp.get('results_by_arm'):
                                results_by_arm = tp.get('results_by_arm', [])
                                break
                    # Fallback to direct results_by_arm
                    if not results_by_arm:
                        results_by_arm = outcome.additional_data.get('results_by_arm', [])
                
                # Get between-group comparison if available
                comparison_data = {}
                if outcome.additional_data:
                    timepoints_data = outcome.additional_data.get('timepoints', [])
                    if timepoints_data:
                        comparison_data = timepoints_data[0].get('between_group_comparison', {})
                    if not comparison_data:
                        comparison_data = outcome.additional_data.get('between_group_comparison', {})
                
                # Create row for each arm
                if results_by_arm:
                    for arm_data in results_by_arm:
                        arm_name = arm_data.get('arm', '')
                        
                        # Parse arm-specific effect estimate from combined string
                        effect_estimate_str = comparison_data.get('effect_estimate', '')
                        ci_lower_str = comparison_data.get('ci_95_lower', '')
                        ci_upper_str = comparison_data.get('ci_95_upper', '')
                        
                        # Extract arm-specific values
                        arm_effect = _parse_arm_specific_effect(effect_estimate_str, arm_name, results_by_arm)
                        arm_ci_lower = _parse_arm_specific_effect(ci_lower_str, arm_name, results_by_arm)
                        arm_ci_upper = _parse_arm_specific_effect(ci_upper_str, arm_name, results_by_arm)
                        
                        row = {
                            'Outcome_Name': outcome.outcome_name,
                            'Timepoint': timepoint_str,
                            'Arm': arm_name,
                            'N_Analyzed': arm_data.get('n'),
                            'Mean': arm_data.get('mean'),
                            'SD': arm_data.get('sd'),
                            'SE': arm_data.get('se'),
                            'Median': arm_data.get('median'),
                            'IQR': arm_data.get('iqr'),
                            'Events': arm_data.get('events'),
                            'Total': arm_data.get('total'),
                            'Percent': arm_data.get('percent'),
                            # Add arm-specific between-group comparison
                            'Effect_Type': comparison_data.get('effect_measure'),
                            'Effect_Estimate': arm_effect.get('estimate'),
                            'CI_95_Lower': arm_ci_lower.get('estimate'),
                            'CI_95_Upper': arm_ci_upper.get('estimate'),
                            'P_Value': comparison_data.get('p_value')  # P-value typically same for all
                        }
                        outcome_rows.append(row)
                else:
                    # No arm-specific data, create single row with aggregate data
                    row = {
                        'Outcome_Name': outcome.outcome_name,
                        'Timepoint': timepoint_str,
                        'Effect_Type': comparison_data.get('effect_measure'),
                        'Effect_Estimate': comparison_data.get('effect_estimate'),
                        'CI_95_Lower': comparison_data.get('ci_95_lower'),
                        'CI_95_Upper': comparison_data.get('ci_95_upper'),
                        'P_Value': comparison_data.get('p_value')
                    }
                    outcome_rows.append(row)
            
            if outcome_rows:
                pd.DataFrame(outcome_rows).to_excel(writer, sheet_name='Primary Outcomes', index=False)
        
        # Sheet 5: Secondary Outcomes - ONE ROW PER ARM
        secondary_outcomes = [o for o in study.outcomes if o.outcome_type == 'secondary']
        if secondary_outcomes:
            sec_rows = []
            seen_outcomes = set()  # Track processed outcomes to avoid duplicates
            
            for outcome in secondary_outcomes:
                # Skip if outcome name is empty, placeholder, or already processed
                if not outcome.outcome_name:
                    continue
                if outcome.outcome_name.startswith('Potential'):
                    continue
                if outcome.outcome_name in seen_outcomes:
                    continue
                seen_outcomes.add(outcome.outcome_name)
                
                # Get timepoint info
                timepoint_str = outcome.planned_timepoints or "Not specified"
                if outcome.timepoints:
                    primary_tp = next((tp for tp in outcome.timepoints if tp.timepoint_type == 'primary'), outcome.timepoints[0])
                    timepoint_str = f"{primary_tp.timepoint_value} {primary_tp.timepoint_unit}" if primary_tp else timepoint_str
                
                # Get arm-by-arm results from additional_data
                results_by_arm = []
                if outcome.additional_data:
                    timepoints_data = outcome.additional_data.get('timepoints', [])
                    if timepoints_data:
                        for tp in timepoints_data:
                            if tp.get('results_by_arm'):
                                results_by_arm = tp.get('results_by_arm', [])
                                break
                    if not results_by_arm:
                        results_by_arm = outcome.additional_data.get('results_by_arm', [])
                
                # Get between-group comparison
                comparison_data = {}
                if outcome.additional_data:
                    timepoints_data = outcome.additional_data.get('timepoints', [])
                    if timepoints_data:
                        comparison_data = timepoints_data[0].get('between_group_comparison', {})
                    if not comparison_data:
                        comparison_data = outcome.additional_data.get('between_group_comparison', {})
                
                # Create row for each arm
                if results_by_arm:
                    for arm_data in results_by_arm:
                        arm_name = arm_data.get('arm', '')
                        
                        # Parse arm-specific effect estimate from combined string
                        effect_estimate_str = comparison_data.get('effect_estimate', '')
                        ci_lower_str = comparison_data.get('ci_95_lower', '')
                        ci_upper_str = comparison_data.get('ci_95_upper', '')
                        
                        # Extract arm-specific values
                        arm_effect = _parse_arm_specific_effect(effect_estimate_str, arm_name, results_by_arm)
                        arm_ci_lower = _parse_arm_specific_effect(ci_lower_str, arm_name, results_by_arm)
                        arm_ci_upper = _parse_arm_specific_effect(ci_upper_str, arm_name, results_by_arm)
                        
                        row = {
                            'Outcome_Name': outcome.outcome_name,
                            'Timepoint': timepoint_str,
                            'Arm': arm_name,
                            'N_Analyzed': arm_data.get('n'),
                            'Mean': arm_data.get('mean'),
                            'SD': arm_data.get('sd'),
                            'Events': arm_data.get('events'),
                            'Total': arm_data.get('total'),
                            # Add arm-specific between-group comparison
                            'Effect_Type': comparison_data.get('effect_measure'),
                            'Effect_Estimate': arm_effect.get('estimate'),
                            'CI_95_Lower': arm_ci_lower.get('estimate'),
                            'CI_95_Upper': arm_ci_upper.get('estimate'),
                            'P_Value': comparison_data.get('p_value')  # P-value typically same for all
                        }
                        sec_rows.append(row)
                else:
                    # No arm-specific data
                    row = {
                        'Outcome_Name': outcome.outcome_name,
                        'Timepoint': timepoint_str,
                        'Effect_Type': comparison_data.get('effect_measure'),
                        'Effect_Estimate': comparison_data.get('effect_estimate'),
                        'CI_95_Lower': comparison_data.get('ci_95_lower'),
                        'CI_95_Upper': comparison_data.get('ci_95_upper'),
                        'P_Value': comparison_data.get('p_value')
                    }
                    sec_rows.append(row)
            
            if sec_rows:
                pd.DataFrame(sec_rows).to_excel(writer, sheet_name='Secondary Outcomes', index=False)
        
        # Sheet 6: Adverse Events
        if study.adverse_events:
            ae_rows = []
            for ae in study.adverse_events:
                base_ae = {
                    'Event_Name': ae.event_name,
                    'Severity': ae.severity
                }
                
                if ae.results_by_arm:
                    for arm_result in ae.results_by_arm:
                        row = base_ae.copy()
                        
                        # Convert to numbers safely
                        participants = arm_result.get('participants_with_event')
                        total = arm_result.get('total_exposed')
                        
                        # Calculate percentage safely
                        percent = None
                        if participants and total:
                            try:
                                participants_num = float(participants) if isinstance(participants, (int, float, str)) else None
                                total_num = float(total) if isinstance(total, (int, float, str)) else None
                                if participants_num and total_num and total_num > 0:
                                    percent = round((participants_num / total_num * 100), 1)
                            except (ValueError, TypeError):
                                percent = None
                        
                        row.update({
                            'Arm': arm_result.get('arm'),
                            'Events': arm_result.get('events'),
                            'Participants_With_Event': arm_result.get('participants_with_event'),
                            'Total_Exposed': arm_result.get('total_exposed'),
                            'Percent': percent
                        })
                        ae_rows.append(row)
                else:
                    ae_rows.append(base_ae)
            
            if ae_rows:
                pd.DataFrame(ae_rows).to_excel(writer, sheet_name='Adverse Events', index=False)
        
        # Sheet 7: Subgroup Analyses
        if study.subgroups:
            sg_rows = []
            for sg in study.subgroups:
                base_sg = {
                    'Subgroup_Variable': sg.subgroup_variable,
                    'P_Interaction': sg.p_interaction
                }
                
                if sg.subgroups:
                    # Flatten nested subgroup data
                    for subgroup in sg.subgroups:
                        row = base_sg.copy()
                        row['Subgroup_Name'] = subgroup.get('subgroup_name')
                        row['Subgroup_Definition'] = subgroup.get('subgroup_definition')
                        
                        # Add effect estimates
                        if subgroup.get('effect_estimate'):
                            effect = subgroup['effect_estimate']
                            row.update({
                                'Effect_Type': effect.get('type'),
                                'Effect_Value': effect.get('value'),
                                'CI_Lower': effect.get('ci_lower'),
                                'CI_Upper': effect.get('ci_upper'),
                                'P_Value': effect.get('p_value')
                            })
                        
                        sg_rows.append(row)
                else:
                    sg_rows.append(base_sg)
            
            if sg_rows:
                pd.DataFrame(sg_rows).to_excel(writer, sheet_name='Subgroup Analyses', index=False)
        
        # Sheet 8: Data Quality & Confidence
        if study.confidence_scores:
            conf_data = pd.DataFrame([{
                'Overall_Confidence_%': round(study.confidence_scores.get('overall', 0) * 100, 1),
                'Identification_Score_%': round(study.confidence_scores.get('by_section', {}).get('identification', 0) * 100, 1),
                'Design_Score_%': round(study.confidence_scores.get('by_section', {}).get('design', 0) * 100, 1),
                'Interventions_Score_%': round(study.confidence_scores.get('by_section', {}).get('interventions', 0) * 100, 1),
                'Outcomes_Score_%': round(study.confidence_scores.get('by_section', {}).get('outcomes', 0) * 100, 1),
                'Tables_Extracted': study.confidence_scores.get('tables_extracted', 0)
            }])
            conf_data.to_excel(writer, sheet_name='Data Quality', index=False)
        
        # Sheet 9: Extraction Details
        if study.extraction_metadata:
            meta = study.extraction_metadata
            meta_data = pd.DataFrame([{
                'Extraction_Methods': ', '.join(meta.get('extraction_methods', [])) if isinstance(meta.get('extraction_methods'), list) else meta.get('extraction_methods'),
                'PDF_Pages': meta.get('pdf_pages'),
                'Tables_Found': meta.get('tables_found'),
                'LLM_Model': meta.get('llm_model'),
                'LLM_Tokens_Used': meta.get('llm_tokens'),
                'LLM_Finish_Reason': meta.get('llm_finish_reason'),
                'Extraction_Date': study.extraction_date.strftime('%Y-%m-%d %H:%M:%S') if study.extraction_date else 'N/A'
            }])
            meta_data.to_excel(writer, sheet_name='Extraction Metadata', index=False)
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'study_{study_id}_{study.trial_registration or "data"}.xlsx'
    )

@app.route('/api/studies/search', methods=['GET'])
def search_studies():
    """Search and filter studies"""
    query = request.args.get('q', '')
    year = request.args.get('year', '')
    intervention = request.args.get('intervention', '')
    
    studies_query = Study.query
    
    if query:
        studies_query = studies_query.filter(Study.title.ilike(f'%{query}%'))
    
    if year:
        studies_query = studies_query.filter(Study.year == int(year))
    
    if intervention:
        studies_query = studies_query.join(Intervention).filter(
            Intervention.arm_name.ilike(f'%{intervention}%')
        )
    
    studies = studies_query.all()
    
    return jsonify({
        'studies': [_serialize_study(s) for s in studies],
        'count': len(studies)
    })

@app.route('/api/studies/compare', methods=['POST'])
def compare_studies():
    """Compare multiple studies side by side"""
    study_ids = request.json.get('study_ids', [])
    
    if not study_ids or len(study_ids) < 2:
        return jsonify({'error': 'Need at least 2 study IDs'}), 400
    
    studies = Study.query.filter(Study.id.in_(study_ids)).all()
    
    comparison = {
        'studies': [_serialize_study(s) for s in studies],
        'comparison_matrix': {
            'interventions': {},
            'outcomes': {},
            'populations': {}
        }
    }
    
    for study in studies:
        study_key = f"{study.trial_registration or study.id}"
        
        comparison['comparison_matrix']['interventions'][study_key] = [
            {'name': i.arm_name, 'n': i.n_randomized} 
            for i in study.interventions
        ]
        
        comparison['comparison_matrix']['populations'][study_key] = {
            'n': study.population_data.get('total_randomized') if study.population_data else None,
            'age': study.population_data.get('age_mean') if study.population_data else None
        }
    
    return jsonify(comparison)

@app.route('/api/export/all-studies', methods=['GET'])
def export_all_studies_combined():
    """Export all studies in NMA-ready format"""
    studies = Study.query.all()
    
    if not studies:
        return jsonify({'error': 'No studies found'}), 404
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_data = []
        for study in studies:
            summary_data.append({
                'Study_ID': study.id,
                'Trial_Registration': study.trial_registration,
                'Title': study.title,
                'Year': study.year,
                'N_Randomized': study.population_data.get('total_randomized') if study.population_data else None,
                'N_Arms': len(study.interventions),
                'Extraction_Date': study.extraction_date.strftime('%Y-%m-%d')
            })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='All Studies', index=False)
    
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='all_studies_NMA_ready.xlsx'
    )

# ==================== DATABASE MIGRATION ====================

@app.cli.command()
def init_db():
    """Initialize the database"""
    db.create_all()
    print("Database initialized!")

@app.cli.command()
def migrate_add_pdf_columns():
    """Add pdf_blob and pdf_filename columns to existing database"""
    from sqlalchemy import text
    
    try:
        with db.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='studies' AND column_name IN ('pdf_blob', 'pdf_filename')"
            ))
            existing_columns = [row[0] for row in result]
        
        if 'pdf_blob' not in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE studies ADD COLUMN pdf_blob BYTEA"))
                conn.commit()
            print("✓ Added pdf_blob column")
        else:
            print("✓ pdf_blob column already exists")
        
        if 'pdf_filename' not in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE studies ADD COLUMN pdf_filename VARCHAR(500)"))
                conn.commit()
            print("✓ Added pdf_filename column")
        else:
            print("✓ pdf_filename column already exists")
        
        print("\n✓ Migration complete!")
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)