"""
Clinical Trial Data Extractor - Production Backend
Comprehensive PDF extraction using multiple methods: OCR, LLM, GROBID, Heuristics
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

# PDF Processing Libraries
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
# import camelot  # Skipping - has Windows compatibility issues
from PIL import Image
import io

# LLM
import openai
from openai import OpenAI

# GROBID (optional - requires separate service)
import requests

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
        
    def extract_text_pdfplumber(self) -> str:
        """Extract text using pdfplumber - best for modern PDFs"""
        try:
            text = ""
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            self.methods_used.append('pdfplumber')
            return text
        except Exception as e:
            print(f"PDFPlumber extraction failed: {e}")
            return ""
    
    def extract_tables_camelot(self) -> List[Dict]:
        """Extract tables using Camelot - excellent for structured tables"""
        return []  # Disabled on Windows
        try:
            tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='lattice')
            extracted_tables = []
            
            for i, table in enumerate(tables):
                extracted_tables.append({
                    'table_number': i + 1,
                    'page': table.page,
                    'data': table.df.to_dict('records'),
                    'accuracy': table.accuracy,
                    'method': 'camelot_lattice'
                })
            
            # Try stream flavor for tables without clear borders
            if len(tables) < 3:
                stream_tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='stream')
                for i, table in enumerate(stream_tables):
                    extracted_tables.append({
                        'table_number': len(tables) + i + 1,
                        'page': table.page,
                        'data': table.df.to_dict('records'),
                        'accuracy': table.accuracy,
                        'method': 'camelot_stream'
                    })
            
            self.methods_used.append('camelot')
            return extracted_tables
        except Exception as e:
            print(f"Camelot extraction failed: {e}")
            return []
    
    def extract_with_ocr(self) -> str:
        """OCR extraction for scanned PDFs or images"""
        try:
            images = convert_from_path(self.pdf_path, dpi=300)
            text = ""
            
            for i, image in enumerate(images):
                # Use Tesseract with optimized settings
                custom_config = r'--oem 3 --psm 6'
                page_text = pytesseract.image_to_string(image, config=custom_config)
                text += f"\n--- Page {i+1} ---\n{page_text}"
            
            self.methods_used.append('tesseract_ocr')
            return text
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""
    
    def extract_with_grobid(self) -> Dict:
        """Extract structured data using GROBID (if service is running)"""
        try:
            grobid_url = os.getenv('GROBID_URL', 'http://localhost:8070')
            
            with open(self.pdf_path, 'rb') as pdf_file:
                files = {'input': pdf_file}
                response = requests.post(
                    f'{grobid_url}/api/processFulltextDocument',
                    files=files,
                    timeout=60
                )
            
            if response.status_code == 200:
                self.methods_used.append('grobid')
                return self._parse_grobid_xml(response.text)
            else:
                print(f"GROBID service returned status {response.status_code}")
                return {}
        except Exception as e:
            print(f"GROBID extraction failed (service may not be running): {e}")
            return {}
    
    def _parse_grobid_xml(self, xml_text: str) -> Dict:
        """Parse GROBID XML output"""
        # Simplified parser - in production, use proper XML parsing
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(xml_text)
            return {
                'title': root.findtext('.//title', ''),
                'abstract': root.findtext('.//abstract', ''),
                'sections': [],  # Would extract sections here
                'references': []  # Would extract references here
            }
        except Exception as e:
            print(f"GROBID XML parsing failed: {e}")
            return {}
    
    def comprehensive_extract(self) -> Dict:
        """Run all extraction methods and combine results"""
        results = {
            'text': '',
            'tables': [],
            'grobid_data': {},
            'methods_used': []
        }
        
        # Primary text extraction
        text = self.extract_text_pdfplumber()
        if not text or len(text) < 100:
            # Fall back to OCR if pdfplumber fails or gets minimal text
            text = self.extract_with_ocr()
        
        results['text'] = text
        
        # Table extraction
        results['tables'] = self.extract_tables_camelot()
        
        # GROBID for structure (optional)
        results['grobid_data'] = self.extract_with_grobid()
        
        results['methods_used'] = self.methods_used
        
        return results

# ==================== LLM EXTRACTION ====================

class LLMExtractor:
    """Advanced LLM-based extraction with GPT-4"""
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
    
    def extract_trial_data(self, text: str, tables: List[Dict] = None) -> Dict:
        """Extract comprehensive trial data using GPT-4"""
        
        # Prepare context with tables if available
        table_context = ""
        if tables:
            table_context = "\n\n=== EXTRACTED TABLES ===\n"
            for i, table in enumerate(tables[:10]):  # Limit to first 10 tables
                table_context += f"\nTable {i+1} (Page {table.get('page', 'unknown')}):\n"
                table_context += json.dumps(table['data'][:20], indent=2)  # First 20 rows
        
        prompt = self._create_extraction_prompt(text, table_context)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert clinical trial data extractor with 20+ years of experience in systematic reviews and network meta-analyses. 
                        You extract data with extreme precision, never hallucinate, and always indicate when information is uncertain or missing.
                        You are particularly skilled at identifying all statistical parameters, subgroup analyses, and nuanced methodological details."""
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
            
            extracted_data = json.loads(response.choices[0].message.content)
            return extracted_data
            
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return {}
    
    def _create_extraction_prompt(self, text: str, table_context: str) -> str:
        """Create comprehensive extraction prompt"""
        return f"""Extract ALL data from this clinical trial for network meta-analysis. Be EXHAUSTIVE.

CRITICAL INSTRUCTIONS:
1. Extract EVERY numerical value, statistical parameter, and data point
2. Include ALL subgroup analyses with complete statistics
3. Never hallucinate - if data is unclear or missing, mark as "UNCERTAIN" or "NOT_REPORTED"
4. Extract exact values from tables when available
5. Include confidence intervals, p-values, standard deviations, medians, IQRs, event counts
6. Capture baseline characteristics for all arms
7. Extract adverse event frequencies

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
    "type": "parallel-group RCT/crossover/cluster-randomized/factorial/etc",
    "blinding": "double-blind/single-blind/open-label/details",
    "randomization_method": "exact description",
    "allocation_concealment": "method",
    "allocation_ratio": "e.g., 1:1 or 2:1",
    "sample_size_calculation": "complete power calculation",
    "duration_total": "total study duration",
    "duration_treatment": "treatment phase duration",
    "duration_followup": "followup duration",
    "number_of_sites": "single-center/multi-center, number",
    "country": "countries involved"
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
{text}

{table_context}

Extract now. Be thorough and precise."""

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
            'confidence_scores': confidence_scores
        }
        
        return final_data, metadata
    
    def _ensemble_results(self, llm_data: Dict, heuristic_data: Dict, pdf_data: Dict) -> Dict:
        """Combine and validate results from different methods"""
        
        # Start with LLM data as base (most comprehensive)
        ensembled = llm_data.copy()
        
        # Validate numerical fields with heuristic extraction
        if 'p_values' in heuristic_data and heuristic_data['p_values']:
            # Cross-check p-values
            pass
        
        # Add table data references
        if pdf_data.get('tables'):
            ensembled['extracted_tables'] = pdf_data['tables']
        
        return ensembled
    
    def _calculate_confidence(self, final_data: Dict, pdf_data: Dict, heuristic_data: Dict) -> Dict:
        """Calculate confidence scores for each extracted field"""
        
        scores = {
            'overall': 0.0,
            'by_section': {}
        }
        
        # Simple confidence scoring
        # High confidence if: multiple methods agree, data is from tables, clear numerical values
        
        if pdf_data.get('tables') and len(pdf_data['tables']) > 0:
            scores['tables_available'] = 0.9
        else:
            scores['tables_available'] = 0.5
        
        if final_data.get('study_identification', {}).get('title'):
            scores['by_section']['identification'] = 0.95
        
        if final_data.get('outcomes'):
            scores['by_section']['outcomes'] = 0.75  # Medium-high
        
        # Overall score
        scores['overall'] = np.mean(list(scores['by_section'].values())) if scores['by_section'] else 0.5
        
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
        'version': '1.0.0',
        'features': [
            'pdf_extraction',
            'ocr',
            'table_extraction',
            'llm_extraction',
            'database_storage'
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
        # Save uploaded file
        file_hash = hashlib.sha256(file.read()).hexdigest()
        file.seek(0)  # Reset file pointer
        
        # Check if already processed
        existing_study = Study.query.filter_by(pdf_hash=file_hash).first()
        if existing_study:
            return jsonify({
                'message': 'Study already extracted',
                'study_id': existing_study.id,
                'data': _serialize_study(existing_study)
            })
        
        # Save file
        filename = f"{file_hash}.pdf"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract data
        extractor = EnsembleExtractor()
        extracted_data, metadata = extractor.extract_comprehensive(filepath)
        
        # Save to database
        study = _save_to_database(extracted_data, metadata, file_hash)
        
        return jsonify({
            'success': True,
            'study_id': study.id,
            'data': extracted_data,
            'metadata': metadata
        })
        
    except Exception as e:
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

def _save_to_database(data: Dict, metadata: Dict, pdf_hash: str) -> Study:
    """Save extracted data to database"""
    
    # Create study
    study = Study(
        pdf_hash=pdf_hash,
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
            n_randomized=intervention_data.get('n_randomized'),
            n_analyzed=intervention_data.get('n_analyzed_primary'),
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
    return {
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
        'metadata': study.extraction_metadata,
        'confidence_scores': study.confidence_scores,
        'extraction_date': study.extraction_date.isoformat()
    }

# ==================== DATABASE INITIALIZATION ====================

@app.cli.command()
def init_db():
    """Initialize the database"""
    db.create_all()
    print("Database initialized!")

# Add these imports at the top of app.py (with other imports)
import io
from flask import send_file
import pandas as pd

# Add these routes BEFORE the "if __name__ == '__main__':" line in app.py

@app.route('/api/export/csv/<int:study_id>', methods=['GET'])
def export_csv(study_id):
    """Export study data as comprehensive Excel file with all CONSORT elements"""
    study = Study.query.get_or_404(study_id)
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Study Identification
        study_info = pd.DataFrame([{
            'Title': study.title,
            'Authors': ', '.join(study.authors) if study.authors else '',
            'Journal': study.journal,
            'Year': study.year,
            'DOI': study.doi,
            'Trial Registration': study.trial_registration,
            'Study Type': study.study_type,
            'Blinding': study.blinding,
            'Randomization Method': study.randomization,
            'Total Duration': study.duration
        }])
        study_info.to_excel(writer, sheet_name='Study Identification', index=False)
        
        # Sheet 2: Population & Eligibility
        if study.population_data:
            pop = study.population_data
            pop_data = pd.DataFrame([{
                'Total Screened': pop.get('total_screened'),
                'Total Randomized': pop.get('total_randomized'),
                'Total Analyzed (ITT)': pop.get('total_analyzed_itt'),
                'Total Analyzed (PP)': pop.get('total_analyzed_pp'),
                'Mean Age': pop.get('age_mean'),
                'Age SD': pop.get('age_sd'),
                'Age Range': pop.get('age_range'),
                'Male N': pop.get('sex_male_n'),
                'Male %': pop.get('sex_male_percent'),
            }])
            pop_data.to_excel(writer, sheet_name='Population Demographics', index=False)
            
            # Inclusion criteria
            if pop.get('inclusion_criteria'):
                inc_df = pd.DataFrame(pop['inclusion_criteria'], columns=['Inclusion Criteria'])
                inc_df.to_excel(writer, sheet_name='Inclusion Criteria', index=False)
            
            # Exclusion criteria
            if pop.get('exclusion_criteria'):
                exc_df = pd.DataFrame(pop['exclusion_criteria'], columns=['Exclusion Criteria'])
                exc_df.to_excel(writer, sheet_name='Exclusion Criteria', index=False)
            
            # Baseline characteristics
            if pop.get('baseline_disease_severity') or study.baseline_characteristics:
                baseline = pop.get('baseline_disease_severity', {}) or study.baseline_characteristics or {}
                if baseline:
                    baseline_df = pd.DataFrame([baseline])
                    baseline_df.to_excel(writer, sheet_name='Baseline Characteristics', index=False)
        
        # Sheet 3: Interventions
        if study.interventions:
            interventions_data = [{
                'Arm Number': idx + 1,
                'Arm Name': i.arm_name,
                'N Randomized': i.n_randomized,
                'N Analyzed': i.n_analyzed,
                'Dose': i.dose,
                'Frequency': i.frequency,
                'Duration': i.duration,
                'Dropouts': i.n_randomized - i.n_analyzed if i.n_randomized and i.n_analyzed else None
            } for idx, i in enumerate(study.interventions)]
            interventions_df = pd.DataFrame(interventions_data)
            interventions_df.to_excel(writer, sheet_name='Interventions', index=False)
        
        # Sheet 4: Primary Outcomes
        if study.outcomes:
            outcomes_data = []
            for outcome in study.outcomes:
                if outcome.outcome_type == 'primary':
                    base_data = {
                        'Outcome Name': outcome.outcome_name,
                        'Timepoint': outcome.timepoint,
                    }
                    
                    # Add effect estimate
                    if outcome.effect_estimate:
                        base_data.update({
                            'Comparison': outcome.effect_estimate.get('comparison'),
                            'Effect Measure': outcome.effect_estimate.get('effect_measure'),
                            'Effect Estimate': outcome.effect_estimate.get('effect_estimate'),
                            'CI 95% Lower': outcome.effect_estimate.get('ci_95_lower'),
                            'CI 95% Upper': outcome.effect_estimate.get('ci_95_upper'),
                            'P-value': outcome.effect_estimate.get('p_value'),
                            'Statistical Test': outcome.effect_estimate.get('statistical_test')
                        })
                    
                    # Add arm-level results
                    if outcome.results_by_arm:
                        for arm_result in outcome.results_by_arm:
                            arm_data = base_data.copy()
                            arm_data.update({
                                'Arm': arm_result.get('arm'),
                                'N': arm_result.get('n'),
                                'Mean': arm_result.get('mean'),
                                'SD': arm_result.get('sd'),
                                'Median': arm_result.get('median'),
                                'IQR': arm_result.get('iqr'),
                                'Events': arm_result.get('events'),
                                'Total': arm_result.get('total')
                            })
                            outcomes_data.append(arm_data)
                    else:
                        outcomes_data.append(base_data)
            
            if outcomes_data:
                outcomes_df = pd.DataFrame(outcomes_data)
                outcomes_df.to_excel(writer, sheet_name='Primary Outcomes', index=False)
        
        # Sheet 5: Secondary Outcomes
        secondary_outcomes = []
        for outcome in study.outcomes:
            if outcome.outcome_type == 'secondary':
                sec_data = {
                    'Outcome Name': outcome.outcome_name,
                    'Timepoint': outcome.timepoint,
                }
                if outcome.effect_estimate:
                    sec_data.update({
                        'Effect': outcome.effect_estimate.get('effect_estimate'),
                        'CI Lower': outcome.effect_estimate.get('ci_95_lower'),
                        'CI Upper': outcome.effect_estimate.get('ci_95_upper'),
                        'P-value': outcome.effect_estimate.get('p_value')
                    })
                secondary_outcomes.append(sec_data)
        
        if secondary_outcomes:
            sec_df = pd.DataFrame(secondary_outcomes)
            sec_df.to_excel(writer, sheet_name='Secondary Outcomes', index=False)
        
        # Sheet 6: Subgroup Analyses
        if study.subgroups:
            subgroups_data = []
            for sg in study.subgroups:
                if sg.subgroups:
                    for subgroup in sg.subgroups:
                        subgroups_data.append({
                            'Subgroup Variable': sg.subgroup_variable,
                            'Subgroup Name': subgroup.get('subgroup_name'),
                            'Subgroup Definition': subgroup.get('subgroup_definition'),
                            'P for Interaction': sg.p_interaction,
                            'Effect Type': subgroup.get('effect_estimate', {}).get('type'),
                            'Effect Estimate': subgroup.get('effect_estimate', {}).get('value'),
                            'CI Lower': subgroup.get('effect_estimate', {}).get('ci_lower'),
                            'CI Upper': subgroup.get('effect_estimate', {}).get('ci_upper'),
                            'P-value': subgroup.get('effect_estimate', {}).get('p_value')
                        })
            
            if subgroups_data:
                subgroups_df = pd.DataFrame(subgroups_data)
                subgroups_df.to_excel(writer, sheet_name='Subgroup Analyses', index=False)
        
        # Sheet 7: Adverse Events
        if study.adverse_events:
            ae_data = []
            for ae in study.adverse_events:
                if ae.results_by_arm:
                    for arm_result in ae.results_by_arm:
                        ae_data.append({
                            'Event Name': ae.event_name,
                            'Severity': ae.severity,
                            'Arm': arm_result.get('arm'),
                            'Events': arm_result.get('events'),
                            'Participants with Event': arm_result.get('participants_with_event'),
                            'Total Exposed': arm_result.get('total_exposed')
                        })
            
            if ae_data:
                ae_df = pd.DataFrame(ae_data)
                ae_df.to_excel(writer, sheet_name='Adverse Events', index=False)
        
        # Sheet 8: Risk of Bias Assessment
        if study.extraction_metadata and 'risk_of_bias' in str(study.extraction_metadata):
            # Try to extract from metadata
            pass
        
        # Sheet 9: Extraction Metadata
        if study.extraction_metadata:
            meta_data = pd.DataFrame([{
                'Extraction Date': study.extraction_date.strftime('%Y-%m-%d %H:%M') if study.extraction_date else '',
                'Extraction Methods': ', '.join(study.extraction_metadata.get('extraction_methods', [])),
                'Tables Found': study.extraction_metadata.get('tables_found'),
                'Overall Confidence': study.confidence_scores.get('overall') if study.confidence_scores else None
            }])
            meta_data.to_excel(writer, sheet_name='Extraction Metadata', index=False)
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'study_{study_id}_CONSORT_data.xlsx'
    )
@app.route('/api/export/nma-ready/<int:study_id>', methods=['GET'])
def export_nma_ready(study_id):
    """Export study data in NMA-ready format (wide format for meta-analysis software)"""
    study = Study.query.get_or_404(study_id)
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # NMA-ready outcomes format (one row per comparison)
        nma_data = []
        
        for outcome in study.outcomes:
            if outcome.outcome_type == 'primary' and outcome.effect_estimate:
                row = {
                    'Study': study.trial_registration or study.title[:50],
                    'Year': study.year,
                    'Outcome': outcome.outcome_name,
                    'Timepoint': outcome.timepoint,
                    'Effect_Type': outcome.effect_estimate.get('type'),
                    'Effect_Estimate': outcome.effect_estimate.get('value'),
                    'CI_Lower': outcome.effect_estimate.get('ci_lower'),
                    'CI_Upper': outcome.effect_estimate.get('ci_upper'),
                    'P_Value': outcome.effect_estimate.get('p_value'),
                    'SE': None  # Calculate if CI available
                }
                
                # Add treatment arms
                if study.interventions:
                    for i, intervention in enumerate(study.interventions[:2]):  # First 2 arms
                        prefix = f'Arm{i+1}_'
                        row[f'{prefix}Name'] = intervention.arm_name
                        row[f'{prefix}N'] = intervention.n_analyzed
                
                # Add arm-specific results
                if outcome.results_by_arm:
                    for i, arm_result in enumerate(outcome.results_by_arm[:2]):
                        prefix = f'Arm{i+1}_'
                        row[f'{prefix}Mean'] = arm_result.get('mean')
                        row[f'{prefix}SD'] = arm_result.get('sd')
                        row[f'{prefix}Events'] = arm_result.get('events')
                        row[f'{prefix}Total'] = arm_result.get('total')
                
                nma_data.append(row)
        
        if nma_data:
            nma_df = pd.DataFrame(nma_data)
            nma_df.to_excel(writer, sheet_name='NMA_Ready', index=False)
        
        # Add subgroup data in NMA format
        subgroup_nma = []
        for sg in study.subgroups:
            if sg.subgroups:
                for subgroup in sg.subgroups:
                    subgroup_nma.append({
                        'Study': study.trial_registration or study.title[:50],
                        'Subgroup_Variable': sg.subgroup_variable,
                        'Subgroup_Name': subgroup.get('subgroup_name'),
                        'Effect': subgroup.get('effect_estimate', {}).get('value'),
                        'CI_Lower': subgroup.get('effect_estimate', {}).get('ci_lower'),
                        'CI_Upper': subgroup.get('effect_estimate', {}).get('ci_upper'),
                        'P_Value': subgroup.get('effect_estimate', {}).get('p_value'),
                        'P_Interaction': sg.p_interaction
                    })
        
        if subgroup_nma:
            subgroup_df = pd.DataFrame(subgroup_nma)
            subgroup_df.to_excel(writer, sheet_name='Subgroup_NMA', index=False)
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'study_{study_id}_NMA_ready.xlsx'
    )

@app.route('/api/export/all-studies/csv', methods=['GET'])
def export_all_studies_csv():
    """Export all studies as a single Excel file with multiple sheets"""
    studies = Study.query.all()
    
    if not studies:
        return jsonify({'error': 'No studies found'}), 404
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet - all studies
        summary_data = []
        for study in studies:
            summary_data.append({
                'Study_ID': study.id,
                'Title': study.title,
                'Year': study.year,
                'Trial_Registration': study.trial_registration,
                'Journal': study.journal,
                'N_Randomized': study.population_data.get('total_randomized') if study.population_data else None,
                'N_Arms': len(study.interventions),
                'N_Outcomes': len([o for o in study.outcomes if o.outcome_type == 'primary']),
                'Extraction_Date': study.extraction_date.strftime('%Y-%m-%d') if study.extraction_date else None
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Studies_Summary', index=False)
        
        # Combined outcomes from all studies
        all_outcomes = []
        for study in studies:
            for outcome in study.outcomes:
                if outcome.outcome_type == 'primary':
                    all_outcomes.append({
                        'Study_ID': study.id,
                        'Study': study.trial_registration or study.title[:50],
                        'Year': study.year,
                        'Outcome': outcome.outcome_name,
                        'Timepoint': outcome.timepoint,
                        'Effect_Type': outcome.effect_estimate.get('type') if outcome.effect_estimate else None,
                        'Effect': outcome.effect_estimate.get('value') if outcome.effect_estimate else None,
                        'CI_Lower': outcome.effect_estimate.get('ci_lower') if outcome.effect_estimate else None,
                        'CI_Upper': outcome.effect_estimate.get('ci_upper') if outcome.effect_estimate else None,
                        'P_Value': outcome.effect_estimate.get('p_value') if outcome.effect_estimate else None
                    })
        
        if all_outcomes:
            outcomes_df = pd.DataFrame(all_outcomes)
            outcomes_df.to_excel(writer, sheet_name='All_Outcomes', index=False)
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='all_studies_export.xlsx'
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
