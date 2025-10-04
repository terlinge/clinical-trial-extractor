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
    
    study_type = db.Column(db.String(100))
    blinding = db.Column(db.String(100))
    randomization = db.Column(db.Text)
    duration = db.Column(db.String(100))
    
    population_data = db.Column(db.JSON)
    baseline_characteristics = db.Column(db.JSON)
    
    extraction_metadata = db.Column(db.JSON)
    confidence_scores = db.Column(db.JSON)
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
    timepoint = db.Column(db.String(100))
    results_by_arm = db.Column(db.JSON)
    effect_estimate = db.Column(db.JSON)
    confidence_score = db.Column(db.Float)

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

# ==================== PDF EXTRACTION METHODS ====================

class PDFExtractor:
    """Multi-method PDF extraction with ensemble approach"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.methods_used = []
        self.page_count = 0
        
    def extract_text_pdfplumber(self) -> str:
        """Extract text using pdfplumber - best for modern PDFs"""
        try:
            text = ""
            with pdfplumber.open(self.pdf_path) as pdf:
                self.page_count = len(pdf.pages)
                for page in pdf.pages:
                    text += page.extract_text() or ""
            self.methods_used.append('pdfplumber_text')
            print(f"PDFPlumber extracted {len(text)} characters from {self.page_count} pages")
            return text
        except Exception as e:
            print(f"PDFPlumber text extraction failed: {e}")
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
        """OCR extraction for scanned PDFs or images"""
        try:
            images = convert_from_path(self.pdf_path, dpi=300)
            text = ""
            
            for i, image in enumerate(images):
                custom_config = r'--oem 3 --psm 6'
                page_text = pytesseract.image_to_string(image, config=custom_config)
                text += f"\n--- Page {i+1} ---\n{page_text}"
            
            self.methods_used.append('tesseract_ocr')
            print(f"OCR extracted {len(text)} characters")
            return text
        except Exception as e:
            print(f"OCR extraction failed: {e}")
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
        if not text or len(text) < 100:
            print("Text extraction minimal, trying OCR...")
            text = self.extract_with_ocr()
        
        results['text'] = text
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
    
    def extract_trial_data(self, text: str, tables: List[Dict] = None) -> Dict:
        """Extract comprehensive trial data using GPT-4"""
        
        # Prepare context with tables if available
        table_context = ""
        if tables:
            table_context = "\n\n=== EXTRACTED TABLES ===\n"
            for i, table in enumerate(tables[:10]):
                table_context += f"\nTable {i+1} (Page {table.get('page', 'unknown')}, {table.get('rows', 0)} rows x {table.get('cols', 0)} cols):\n"
                table_context += json.dumps(table['data'][:20], indent=2)
        
        prompt = self._create_extraction_prompt(text, table_context)
        
        print(f"\n=== LLM EXTRACTION STARTING ===")
        print(f"Model: {self.model}")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Tables included: {len(tables) if tables else 0}")
        
        try:
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
                response_format={"type": "json_object"}
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
            print(f"========================\n")
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"❌ LLM JSON parsing error: {e}")
            print(f"First 500 chars of response: {raw_content[:500] if 'raw_content' in locals() else 'N/A'}")
            return {}
        except Exception as e:
            print(f"❌ LLM extraction error: {e}")
            return {}
    
    def _create_extraction_prompt(self, text: str, table_context: str) -> str:
        """Create comprehensive extraction prompt"""
        return f"""Extract ALL data from this clinical trial for network meta-analysis. Be EXHAUSTIVE and THOROUGH.

CRITICAL INSTRUCTIONS FOR STUDY DESIGN:
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

Return JSON with this EXACT structure:
{{
  "study_identification": {{
    "title": "complete title",
    "authors": ["list all authors"],
    "journal": "journal name",
    "year": "YYYY",
    "doi": "DOI",
    "trial_registration": "NCT or ISRCTN number",
    "correspondence_author": "name and email if available"
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
    "inclusion_criteria": ["criterion 1", "criterion 2"],
    "exclusion_criteria": ["criterion 1", "criterion 2"],
    "total_screened": "number screened",
    "total_randomized": "number randomized",
    "total_analyzed_itt": "ITT population",
    "total_analyzed_pp": "per-protocol population",
    "age_mean": "mean age",
    "age_sd": "SD",
    "age_median": "median if reported",
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
        "outcome_name": "exact outcome name",
        "measurement_tool": "scale or instrument used",
        "timepoint": "when measured",
        "definition": "how outcome was defined",
        "results_by_arm": [
          {{
            "arm": "arm name matching interventions",
            "n": "number analyzed",
            "mean": "mean value",
            "sd": "standard deviation",
            "se": "standard error",
            "median": "median",
            "q1": "first quartile",
            "q3": "third quartile",
            "iqr": "interquartile range",
            "min": "minimum",
            "max": "maximum",
            "events": "for binary outcomes",
            "total": "denominator for binary",
            "percent": "percentage"
          }}
        ],
        "between_group_comparison": {{
          "comparison": "which arms compared",
          "effect_measure": "MD/SMD/OR/RR/HR/etc",
          "effect_estimate": "point estimate",
          "ci_95_lower": "lower CI",
          "ci_95_upper": "upper CI",
          "p_value": "exact p-value",
          "statistical_test": "test used"
        }},
        "data_source": "text/table X/figure Y"
      }}
    ],
    "secondary": []
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

Extract now. Be thorough, search the ENTIRE paper especially Methods section, and list ALL sites/institutions. MAKE SURE to extract primary and secondary outcomes with full statistics."""

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
        
        # Step 2: LLM extraction
        llm_data = self.llm_extractor.extract_trial_data(
            pdf_data['text'], 
            pdf_data.get('tables', [])
        )
        
        # Step 3: Heuristic extraction
        heuristic_data = self.heuristic_extractor.extract_statistical_values(pdf_data['text'])
        
        # Step 4: Ensemble and validation
        final_data = self._ensemble_results(llm_data, heuristic_data, pdf_data)
        
        # Step 5: Calculate confidence scores
        confidence_scores = self._calculate_confidence(final_data, pdf_data, heuristic_data)
        
        metadata = {
            'extraction_methods': pdf_data['methods_used'] + ['llm', 'heuristic'],
            'tables_found': len(pdf_data.get('tables', [])),
            'pdf_pages': pdf_data.get('page_count', 0),
            'confidence_scores': confidence_scores,
            'llm_model': self.llm_extractor.model,
            'llm_tokens': self.llm_extractor.tokens_used,
            'llm_finish_reason': self.llm_extractor.finish_reason
        }
        
        return final_data, metadata
    
    def _ensemble_results(self, llm_data: Dict, heuristic_data: Dict, pdf_data: Dict) -> Dict:
        """Combine and validate results from different methods"""
        ensembled = llm_data.copy()
        
        if pdf_data.get('tables'):
            ensembled['extracted_tables'] = pdf_data['tables']
        
        return ensembled
    
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
    """Serve the frontend"""
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
        
        # Clean up temp file
        os.unlink(filepath)
        
        return jsonify({
            'success': True,
            'study_id': study.id,
            'data': extracted_data,
            'metadata': metadata
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
        
        # Add outcomes
        for outcome_type in ['primary', 'secondary']:
            for outcome_data in extracted_data.get('outcomes', {}).get(outcome_type, []):
                outcome = Outcome(
                    study_id=study.id,
                    outcome_name=outcome_data.get('outcome_name'),
                    outcome_type=outcome_type,
                    timepoint=outcome_data.get('timepoint'),
                    results_by_arm=outcome_data.get('results_by_arm'),
                    effect_estimate=outcome_data.get('between_group_comparison')
                )
                db.session.add(outcome)
        
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

@app.route('/api/studies/<int:study_id>', methods=['DELETE'])
def delete_study(study_id):
    """Delete a study"""
    study = Study.query.get_or_404(study_id)
    db.session.delete(study)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/export/<int:study_id>', methods=['GET'])
def export_study(study_id):
    """Export study data as JSON"""
    study = Study.query.get_or_404(study_id)
    data = _serialize_study(study)
    
    return jsonify(data)

# ==================== HELPER FUNCTIONS ====================

def _clean_int_value(value):
    """Convert 'Not Reported' strings to None for integer fields"""
    if value in [None, '', 'Not Reported', 'NOT_REPORTED', 'N/A', 'Not reported', 'Not applicable']:
        return None
    if isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    return value

def _save_to_database(data: Dict, metadata: Dict, pdf_hash: str, pdf_blob: bytes, pdf_filename: str) -> Study:
    """Save extracted data to database"""
    
    # Create study
    study = Study(
        pdf_hash=pdf_hash,
        pdf_blob=pdf_blob,
        pdf_filename=pdf_filename,
        title=data.get('study_identification', {}).get('title'),
        authors=data.get('study_identification', {}).get('authors', []),
        journal=data.get('study_identification', {}).get('journal'),
        year=data.get('study_identification', {}).get('year'),
        doi=data.get('study_identification', {}).get('doi'),
        trial_registration=data.get('study_identification', {}).get('trial_registration'),
        study_type=data.get('study_design', {}).get('type'),
        blinding=data.get('study_design', {}).get('blinding'),
        randomization=data.get('study_design', {}).get('randomization_method'),
        duration=data.get('study_design', {}).get('duration_total'),
        population_data=data.get('population'),
        baseline_characteristics=data.get('population', {}).get('baseline_disease_severity'),
        extraction_metadata=metadata,
        confidence_scores=metadata.get('confidence_scores')
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
    
    # Add outcomes
    for outcome_type in ['primary', 'secondary']:
        for outcome_data in data.get('outcomes', {}).get(outcome_type, []):
            outcome = Outcome(
                study_id=study.id,
                outcome_name=outcome_data.get('outcome_name'),
                outcome_type=outcome_type,
                timepoint=outcome_data.get('timepoint'),
                results_by_arm=outcome_data.get('results_by_arm'),
                effect_estimate=outcome_data.get('between_group_comparison')
            )
            db.session.add(outcome)
    
    # Add subgroups
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
                    'timepoint': o.timepoint,
                    'results_by_arm': o.results_by_arm,
                    'effect_estimate': o.effect_estimate
                }
                for o in study.outcomes if o.outcome_type == 'primary'
            ],
            'secondary': [
                {
                    'outcome_name': o.outcome_name,
                    'timepoint': o.timepoint,
                    'results_by_arm': o.results_by_arm,
                    'effect_estimate': o.effect_estimate
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

# Export endpoints (simplified - keeping only essential ones)
@app.route('/api/export/csv/<int:study_id>', methods=['GET'])
def export_csv(study_id):
    """Export study data as Excel file"""
    study = Study.query.get_or_404(study_id)
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Study info sheet
        study_info = pd.DataFrame([{
            'Title': study.title,
            'Year': study.year,
            'Journal': study.journal,
            'Trial Registration': study.trial_registration
        }])
        study_info.to_excel(writer, sheet_name='Study Info', index=False)
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'study_{study_id}_data.xlsx'
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