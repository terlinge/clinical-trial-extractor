# 🏥 Clinical Trial Data Extractor v2.2

**AI-Powered Multi-Source Extraction System for Systematic Reviews & Network Meta-Analyses**

A comprehensive web application that extracts clinical trial data from PDF papers using **multiple extraction methods** (AI, OCR, pattern matching, and direct table parsing), with **user control over data source selection** and **complete transparency**, designed to meet Cochrane systematic review standards.

---

## 🎯 What This App Does

The Clinical Trial Data Extractor automates the tedious process of extracting quantitative data from clinical trial PDF publications. It:

1. **Reads PDF clinical trial papers** and extracts structured data (study design, participants, interventions, outcomes, statistics)
2. **Uses 4 different extraction methods** (PDFPlumber tables, Heuristic patterns, OpenAI LLM, Tesseract OCR) to maximize data capture
3. **Stores data in a PostgreSQL database** with comprehensive source attribution for every data point
4. **Provides a web interface** for uploading PDFs, reviewing extractions, and exporting results
5. **Generates Excel exports** with arm-by-arm outcome results, ready for meta-analysis
6. **Supports multiple timepoints** per outcome (e.g., results at 4 weeks, 12 weeks, 6 months)
7. **Enables user control** - review extraction results from all sources and choose which to trust

## 🔧 How It Works

### **Multi-Source Extraction Pipeline**

```
PDF Upload → Four Parallel Extractions → Ensemble Combination → Database Storage → User Review → Export
```

**Step 1: PDF Upload**
- User uploads clinical trial PDF via web interface
- System calculates file hash to prevent duplicate processing
- PDF stored as binary blob in database for future re-extraction

**Step 2: Multi-Method Extraction (Parallel Processing)**

1. **📊 PDFPlumber (Table Extraction)**
   - Extracts structured tables directly from PDF
   - Captures baseline characteristics, demographics, outcome tables
   - **Highest priority** for numerical data
   - Example: "Table 2: Mean change in systolic BP at 12 weeks"

2. **🔍 Heuristic (Pattern Matching)**
   - Uses regex patterns to find statistical values in text
   - Extracts: p-values (p<0.001, p=0.045), confidence intervals [95% CI: 2.3-5.7], effect sizes
   - Captures: sample sizes (n=264), percentages (45.2% male), means ± SDs (72.3±8.5)
   - **Second priority** for numerical data

3. **🤖 OpenAI LLM (AI Interpretation)**
   - GPT-4-turbo analyzes full PDF text with specialized prompt
   - Understands context, study design, arm names, outcome definitions
   - **Two prompt modes**: Full detailed (22K tokens) vs Ultra-compact (15K tokens)
   - Returns structured JSON with source citations
   - **Lower priority** for conflicts (AI can hallucinate)

4. **👁️ Tesseract OCR (Image Text Extraction)**
   - Converts PDF pages to images, then extracts text
   - Fallback for scanned PDFs or data in figures
   - **Lowest priority** but captures data missed by other methods

**Step 3: Ensemble Combination**
- Combines results from all 4 methods intelligently
- Prioritizes: Tables > Patterns > AI > OCR
- Fills in missing data (e.g., calculates total sample size from arm sizes)
- Validates data consistency across sources

**Step 4: Database Storage**
- Saves to PostgreSQL with comprehensive schema:
  - **Studies**: metadata, design, population
  - **Interventions**: arms, doses, sample sizes
  - **Outcomes**: outcome definitions
  - **OutcomeTimepoints**: 27 fields per timepoint (means, SDs, CIs, p-values, effect estimates)
  - **ExtractionSource**: raw data from each method
  - **DataElement**: individual fields with source attribution
  - **ExtractionConflict**: tracks disagreements between methods

**Step 5: User Review & Selection**
- Web interface shows extraction results from all sources
- User can see conflicting values and choose which source to trust
- Auto-select recommendations based on confidence scores
- Manual override capability for systematic review requirements

**Step 6: Export**
- Excel/CSV export with 4 rows per outcome (one per intervention arm)
- Includes: N, Mean, SD, CI, p-values, effect estimates
- Source citations for every data point
- Ready for import into RevMan, R, or Stata for meta-analysis

---

## ✨ Key Features (v2.2)

## ✨ Key Features (v2.2)

### 🆕 **User-Controlled Multi-Source Data Selection**
- **Review Interface**: See extraction results from all 4 methods side-by-side
- **Source Selection**: Choose which source to trust for each data point
- **Conflict Detection**: Automatic identification when sources disagree
- **Confidence Scores**: Each source has a confidence rating (0.0-1.0)
- **Auto-Select**: Recommend highest-confidence sources automatically
- **Manual Override**: Full user control for systematic review standards

### 📊 **Bug Fixes & Data Quality Improvements**
- **Fixed Delete Functionality**: Properly handles foreign key cascades (OutcomeTimepoint cleanup)
- **Fixed CSV Export**: 
  - Deduplication (no more duplicate outcome rows)
  - Arm-by-arm results (4 rows per outcome, one per arm)
  - Removed "Potential..." placeholder rows
  - Extracts data from JSON fields correctly
- **Fixed Database Constraints**: Increased VARCHAR sizes (50→200) for long text fields
- **Fixed AttributeErrors**: Proper access to outcome.timepoints relationship

### 🤖 **LLM Prompt Optimization**
- **Two-Tier System**: Full detailed prompt (22K tokens) + Ultra-compact prompt (15K tokens)
- **Automatic Switching**: Uses compact version when text is large
- **Smart Text Extraction**: Keeps only essential sections (methods, baseline, results, tables)
- **Inline JSON Template**: Reduces instruction bloat
- **30% Token Reduction**: From ~22K to ~15K tokens per extraction

### 🔍 **Enhanced Multi-Method PDF Extraction**
- **PDFPlumber Text**: High-quality text extraction (100k+ characters)
- **Tesseract OCR**: Image and figure text recognition  
- **Table Detection**: Direct extraction of structured data from tables
- **Pattern Matching**: Regex-based statistical value extraction (p-values, CIs, effect sizes)
- **Demographics Extraction**: Direct parsing of baseline characteristics tables
- **Heuristic Analysis**: Rule-based extraction for common clinical trial patterns

### 📊 **Cochrane-Compliant Data Structure**
- **Multiple Timepoints**: Same outcome measured at different times (4 weeks, 12 weeks, 6 months)
- **Enhanced Timepoint Support**: 27 database fields per timepoint for comprehensive data
- **Universal Outcome Types**: Continuous, dichotomous, time-to-event, count data
- **Complete Effect Estimates**: Between-group comparisons with confidence intervals and p-values
- **Source Attribution**: Every data point traceable to specific pages/tables
- **Missing Data Tracking**: Clear identification of unavailable data points

### 💾 **Advanced Database Architecture**
- **PostgreSQL Backend**: Robust relational database with advanced indexing
- **OutcomeTimepoints Table**: 27 fields capturing all statistical parameters
- **Multi-Source Tables**: ExtractionSource, DataElement, ExtractionConflict, UserPreferences
- **Source Citation Storage**: Complete provenance tracking for systematic reviews
- **Meta-Analysis Ready**: Direct export capabilities for RevMan, R, and Stata
- **Complex Queries**: Filter by timepoint, effect size, significance, confidence levels
- **Smart Data Handling**: Increased VARCHAR sizes, proper foreign key cascades

### 🌐 **Enhanced Web Interface**
- **Real-Time Progress**: Live extraction with method-by-method feedback
- **Data Review Tab**: Side-by-side comparison of all extraction sources
- **Conflict Highlighting**: Visual indicators when sources disagree
- **Source Icons**: Clean visual attribution (📊 Table, 🔍 Pattern, 🤖 AI, 👁️ OCR)
- **Tabular Display**: Structured tables instead of verbose text
- **Token Usage Monitoring**: OpenAI API cost tracking
- **Export Options**: Multiple formats with source attribution

---

## 🆕 What's New in v2.2

### **🔧 Critical Bug Fixes**
- **Delete Functionality**: Fixed foreign key constraint errors when deleting studies
  - Now properly deletes OutcomeTimepoint records before study deletion
  - Added comprehensive logging showing what gets deleted
- **CSV Export Fixed**:
  - **Deduplication**: Tracks processed outcomes with `seen_outcomes` set
  - **Arm-by-arm results**: Extracts from `additional_data` JSON field
  - **4 rows per outcome**: One row for each intervention arm
  - **Removed placeholders**: Filters out "Potential secondary outcome" empty rows
- **Database Constraints**: Increased `effect_measure` and `p_value_text` from VARCHAR(50) to VARCHAR(200)
- **AttributeErrors**: Fixed `outcome.timepoint` → `outcome.timepoints` relationship access

### **🚀 Performance Optimizations**
- **LLM Prompt Optimization**: Reduced token usage by 30% (22K → 15K tokens)
  - Ultra-compact prompt with inline JSON template
  - Smart text extraction keeping only essential sections
  - Preserved maximum table data (35KB) while minimizing instructions
- **Automatic Prompt Switching**: Uses compact version for large documents

### **📦 Multi-Source Architecture**
- **New Database Tables**:
  - `extraction_sources`: Raw data from each method with quality metrics
  - `data_elements`: Individual fields with source attribution
  - `user_preferences`: Customizable selection rules
  - `extraction_conflicts`: Tracks disagreements between sources
- **Source Prioritization**: Tables > Patterns > AI > OCR (configurable)
- **User Control**: Review and select preferred sources for each field

### **🎨 Enhanced User Interface**
- **Review & Select Data Tab**: View all extraction sources side-by-side
- **Auto-Select Recommended**: Apply confidence-based recommendations
- **Show Conflicts Only**: Filter to disputed values
- **Finalize Selections**: Apply user choices to final data
- **Enhanced Export**: Include source attribution in exports

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **PostgreSQL 18+**
- **Tesseract OCR**
- **Poppler PDF utilities**
- **OpenAI API key**

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/terlinge/clinical-trial-extractor.git
cd clinical-trial-extractor
```

2. **Set up Python environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

3. **Configure environment variables**
Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://username:password@localhost/clinical_trials
```

4. **Start the application**
```powershell
# Windows (recommended)
.\START_APP.ps1

# Or manually
python app.py
```

5. **Access the web interface**
Open http://localhost:5000 in your browser

---

## 🆕 What's New in v2.2 (Current Release)

### **🔧 Critical Bug Fixes**
✅ **Delete Functionality**: Fixed foreign key constraint errors when deleting studies
   - Now properly deletes OutcomeTimepoint records before study deletion
   - Added comprehensive logging showing what gets deleted
   
✅ **CSV Export Fixed**:
   - **Deduplication**: Tracks processed outcomes with `seen_outcomes` set
   - **Arm-by-arm results**: Extracts from `additional_data` JSON field
   - **4 rows per outcome**: One row for each intervention arm (e.g., Drug 6mg, Drug 12mg, Drug 36mg, Placebo)
   - **Removed placeholders**: Filters out "Potential secondary outcome" empty rows
   
✅ **Database Constraints**: Increased `effect_measure` and `p_value_text` from VARCHAR(50) to VARCHAR(200)

✅ **AttributeErrors**: Fixed `outcome.timepoint` → `outcome.timepoints` relationship access

### **� Performance Optimizations**
⚡ **LLM Prompt Optimization**: Reduced token usage by 30% (22K → 15K tokens)
   - Ultra-compact prompt with inline JSON template
   - Smart text extraction keeping only essential sections
   - Preserved maximum table data (35KB) while minimizing instructions
   
⚡ **Automatic Prompt Switching**: Uses compact version for large documents

### **📦 Multi-Source Architecture**
🆕 **New Database Tables**:
   - `extraction_sources`: Raw data from each method with quality metrics
   - `data_elements`: Individual fields with source attribution
   - `user_preferences`: Customizable selection rules
   - `extraction_conflicts`: Tracks disagreements between sources
   
🆕 **Source Prioritization**: Tables > Patterns > AI > OCR (configurable)

🆕 **User Control**: Review and select preferred sources for each field

### **🎨 Enhanced User Interface**
🆕 **Review & Select Data Tab**: View all extraction sources side-by-side

🆕 **Auto-Select Recommended**: Apply confidence-based recommendations

🆕 **Show Conflicts Only**: Filter to disputed values

🆕 **Finalize Selections**: Apply user choices to final data

🆕 **Enhanced Export**: Include source attribution in exports

---

## 📋 Version History

### **v2.2 - Multi-Source Control & Bug Fixes** (November 2025)

**Critical Fixes:**
- Fixed delete functionality with proper foreign key cascade handling
- Fixed CSV export: deduplication, arm-by-arm results, removed placeholders  
- Fixed database column size constraints (VARCHAR 50→200)
- Fixed AttributeError bugs in outcome extraction

**New Features:**
- Multi-source data architecture with user selection capability
- Enhanced frontend with data review and source comparison UI
- Ultra-compact LLM prompt optimization (30% token reduction)
- Auto-select recommendations based on confidence scores

**New Files:**
- `MULTI_SOURCE_ARCHITECTURE.md` - Architecture documentation
- `database_schema_enhancement.py` - Multi-source table schemas
- `fix_column_sizes.py` - Database migration script
- `migrate_database.py` - Full migration utility
- `templates/enhanced_index.html` - New UI with source selection

### **v2.0 - Enhanced Multi-Timepoint Support** (October 2025)

## 📁 Project Structure

```
clinical-trial-extractor/
├── app.py                          # Main Flask application (3700+ lines, v2.2)
├── templates/
│   ├── index.html                  # Original web interface
│   └── enhanced_index.html         # NEW: Multi-source review interface
├── requirements.txt                # Python dependencies
├── START_APP.ps1                   # Windows startup script
├── START_APP.bat                   # Alternative batch startup
├── .env                            # Environment variables (create this)
├── uploads/                        # Temporary PDF storage
│
├── CHANGELOG.md                    # Detailed version history
├── DEVELOPER_REFERENCE.md          # Technical implementation details
├── TECHNICAL_ARCHITECTURE.md       # System architecture documentation
├── MULTI_SOURCE_ARCHITECTURE.md    # NEW: Multi-source data architecture
├── INSTALLATION.md                 # Step-by-step installation guide
├── COMMIT_CHECKLIST.md            # Pre-commit validation checklist
│
├── fix_column_sizes.py            # NEW: Database migration for VARCHAR increases
├── migrate_database.py            # NEW: Multi-source table creation
├── database_schema_enhancement.py  # NEW: Multi-source schema definitions
├── fix_truncation.py              # Database optimization utility
├── backup_database.py             # Database backup utility
├── database_query_examples.py     # Advanced querying examples
├── check_system.py                # System requirements validator
├── add_columns.py                 # Database schema updater
│
└── services/                       # Future microservices (planned)
    ├── celery_tasks.py
    ├── docker-compose.yml
    └── improved_llm_extractor.py
```

---

## 🧠 How the v2.2 Multi-Source Extraction Works

Our **Enhanced Multi-Source Architecture** combines 4 extraction methods with user control:

### **🔄 Extraction Pipeline Flow**

```
PDF Upload → Parallel Extraction (4 Methods) → Ensemble Merge → Database Storage → User Review → Export
```

**Phase 1: Parallel Extraction (All methods run simultaneously)**

1. **📊 PDFPlumber (Table Extraction)** - Confidence: 0.9
   - Extracts structured tables directly from PDF
   - Preserves numerical data and table structure
   - Captures: baseline tables, outcome tables, CONSORT diagrams
   - **Best for**: Numerical data, sample sizes, statistical results
   - Example output: "Table 2, page 5: Mean 72.3, SD 8.5, N=264"

2. **🔍 Heuristic (Pattern Matching)** - Confidence: 0.7
   - Uses 50+ regex patterns to find statistical values
   - Extracts: p-values, confidence intervals, effect sizes, sample sizes
   - Captures: means±SDs, percentages, odds ratios, hazard ratios
   - **Best for**: Statistical parameters in text
   - Example pattern: `r'p\s*[=<]\s*0\.\d+'` → finds "p=0.001", "p<0.05"

3. **🤖 OpenAI LLM (AI Interpretation)** - Confidence: 0.8
   - GPT-4-turbo with specialized clinical trial prompt
   - Understands context, study design, outcome definitions
   - Two modes: Full (22K tokens) / Ultra-compact (15K tokens)
   - **Best for**: Complex interpretations, identifying primary outcomes
   - Example: Understands "change from baseline" vs "absolute value"

4. **👁️ Tesseract OCR (Image Extraction)** - Confidence: 0.6
   - Converts PDF pages to images, extracts text
   - Fallback for scanned PDFs or data in figures
   - **Best for**: Legacy PDFs, figure text
   - Example: Extracts text from Kaplan-Meier curves

**Phase 2: Intelligent Ensemble Combination**
```python
# Prioritization Logic
if pdfplumber_value and confidence > 0.85:
    use pdfplumber_value  # Trust table data first
elif heuristic_value and pattern_matched:
    use heuristic_value   # Trust direct text patterns second
elif llm_value and confidence > 0.75:
    use llm_value         # Trust AI interpretation third
elif ocr_value:
    use ocr_value         # Use OCR as last resort
```

- Fills missing data (e.g., calculates total_randomized from arm sizes)
- Validates consistency (e.g., does N_analyzed ≤ N_randomized?)
- Detects conflicts and flags for user review

**Phase 3: Multi-Source Database Storage**
- **Primary Storage**: Final ensemble results in Studies, Outcomes, OutcomeTimepoints
- **Source Attribution**: Raw data stored in ExtractionSource table
- **Field-Level Tracking**: Individual values stored in DataElement table
- **Conflict Recording**: Disagreements stored in ExtractionConflict table

**Phase 4: User Review & Selection**
- Web interface displays all 4 source results side-by-side
- Color-coded confidence levels: 🟢 High (>80%), 🟡 Medium (60-80%), 🔴 Low (<60%)
- User can override ensemble selection for any field
- Auto-select applies confidence-based recommendations

**Phase 5: Export Generation**
- Excel/CSV with arm-by-arm results (4 rows per outcome)
- Source citations included for each data point
- Deduplication ensures clean output
- Ready for RevMan, R, or Stata meta-analysis

### **📊 Example Extraction Comparison**

For "Systolic BP change from baseline at 12 weeks":

| Source | Value | Confidence | Location |
|--------|-------|------------|----------|
| 📊 PDFPlumber | -15.2±8.5 mmHg | 0.92 | Table 2, page 5 |
| 🔍 Heuristic | -15.2 (SD 8.5) | 0.75 | Results, page 5 |
| 🤖 LLM | -15.2 mmHg | 0.85 | AI interpretation |
| 👁️ OCR | -15 mmHg | 0.50 | Figure 1 |

**Ensemble Decision**: Use PDFPlumber (highest confidence, structured source)  
**User Override**: Available if reviewer prefers different source

---

## 🎯 Enhanced Extraction Capabilities

### **Multi-Timepoint Support**
The system now captures outcomes measured at different timepoints:
```
Example: Pain Score Outcome
├── Baseline (Week 0)
├── Short-term (Week 4) - Secondary timepoint
├── Primary endpoint (Week 12) - Primary timepoint  
└── Follow-up (Week 26) - Follow-up timepoint
```

### **Source Citation Tracking**
Every extracted data point includes precise source information:
```json
{
  "mean_value": 7.2,
  "data_source": "Table 2 page 5",
  "source_confidence": "high"
}
```

### **Smart Prompt Management**
- **Full Extraction**: For documents under 100k characters
- **Focused Extraction**: Prioritizes statistical tables and outcome data for large documents
- **Universal Keywords**: Works across all clinical trial types (not limited to specific diseases)

---

## 🗄️ Database Schema (v2.2)

### **Core Data Tables**

**Studies Table**
```sql
CREATE TABLE studies (
    id SERIAL PRIMARY KEY,
    pdf_hash VARCHAR(64) UNIQUE NOT NULL,  -- Prevent duplicate processing
    pdf_blob BYTEA,                        -- Store PDF for re-extraction
    pdf_filename VARCHAR(500),
    title TEXT,
    authors JSON,                          -- Array of author names
    journal VARCHAR(500),
    year INTEGER,
    doi VARCHAR(200),
    trial_registration VARCHAR(100),       -- NCT number, etc.
    study_type TEXT,                       -- With source citations
    blinding TEXT,
    randomization TEXT,
    duration TEXT,
    population_data JSON,                  -- Demographics
    baseline_characteristics JSON,
    extraction_metadata JSON,
    confidence_scores JSON,
    source_tracking JSON,                  -- Source attribution
    extraction_date TIMESTAMP
);
```

**Interventions Table**
```sql
CREATE TABLE interventions (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES studies(id) ON DELETE CASCADE,
    arm_name VARCHAR(200),                 -- "Orforglipron 36mg", "Placebo"
    n_randomized INTEGER,                  -- Number randomized to arm
    n_analyzed INTEGER,                    -- Number analyzed in ITT
    dose VARCHAR(200),
    frequency VARCHAR(200),
    duration VARCHAR(200)
);
```

**Outcomes Table**
```sql
CREATE TABLE outcomes (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES studies(id) ON DELETE CASCADE,
    outcome_name TEXT,                     -- "Change in systolic BP"
    outcome_type VARCHAR(50),              -- 'primary', 'secondary'
    outcome_category VARCHAR(100),         -- 'continuous', 'dichotomous', etc.
    planned_timepoints TEXT,               -- "12, 24, 52 weeks"
    primary_timepoint_id INTEGER,          -- Links to main timepoint
    additional_data JSON,                  -- Complex nested data
    data_sources JSON                      -- Source tracking
);
```

**OutcomeTimepoints Table** ⭐ **27 Fields per Timepoint**
```sql
CREATE TABLE outcome_timepoints (
    id SERIAL PRIMARY KEY,
    outcome_id INTEGER REFERENCES outcomes(id) ON DELETE CASCADE,
    study_id INTEGER REFERENCES studies(id),
    
    -- Timepoint identification
    timepoint_name TEXT,                   -- "12 weeks", "end of treatment"
    timepoint_value FLOAT,                 -- 12
    timepoint_unit VARCHAR(50),            -- "weeks"
    timepoint_type VARCHAR(50),            -- "primary", "secondary", "follow_up"
    
    -- Sample size
    n_analyzed INTEGER,
    
    -- Continuous outcome statistics
    mean_value FLOAT,
    sd_value FLOAT,
    median_value FLOAT,
    iqr_lower FLOAT,
    iqr_upper FLOAT,
    ci_95_lower FLOAT,
    ci_95_upper FLOAT,
    
    -- Dichotomous outcome statistics
    events INTEGER,
    total_participants INTEGER,
    
    -- Between-group comparison
    effect_measure VARCHAR(200),           -- "MD", "SMD", "OR", "RR", "HR"
    effect_estimate FLOAT,
    effect_ci_lower FLOAT,
    effect_ci_upper FLOAT,
    p_value FLOAT,
    p_value_text VARCHAR(200),             -- "<0.001", "NS"
    
    -- Source attribution
    data_source TEXT,                      -- "Table 2 page 5"
    source_confidence VARCHAR(20),         -- "high", "medium", "low"
    
    -- Complex data storage
    results_by_arm JSON,                   -- Arm-specific results
    additional_statistics JSON
);
CREATE INDEX idx_timepoint_value_unit ON outcome_timepoints(timepoint_value, timepoint_unit);
CREATE INDEX idx_study_outcome ON outcome_timepoints(study_id, outcome_id);
```

### **Multi-Source Tables** 🆕 **v2.2**

**ExtractionSource Table**
```sql
CREATE TABLE extraction_sources (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES studies(id) ON DELETE CASCADE,
    extraction_method VARCHAR(50),         -- 'pdfplumber', 'ocr', 'llm', 'heuristic'
    source_version VARCHAR(20),            -- 'GPT-4-turbo-preview'
    extraction_timestamp TIMESTAMP,
    raw_data JSON,                         -- Complete extraction result
    confidence_score FLOAT,                -- 0.0 - 1.0
    quality_metrics JSON,                  -- Method-specific indicators
    processing_time FLOAT,
    error_messages JSON,
    success_status BOOLEAN
);
CREATE INDEX idx_study_method ON extraction_sources(study_id, extraction_method);
```

**DataElement Table**
```sql
CREATE TABLE data_elements (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES studies(id) ON DELETE CASCADE,
    element_type VARCHAR(50),              -- 'outcome', 'intervention', 'demographic'
    element_name VARCHAR(200),             -- 'mean_value', 'n_randomized'
    element_path VARCHAR(500),             -- 'outcomes.primary[0].timepoints[0].mean_value'
    
    -- Values from each source
    pdfplumber_value TEXT,
    pdfplumber_confidence FLOAT,
    pdfplumber_source_location VARCHAR(200),
    
    ocr_value TEXT,
    ocr_confidence FLOAT,
    ocr_source_location VARCHAR(200),
    
    llm_value TEXT,
    llm_confidence FLOAT,
    llm_source_location VARCHAR(200),
    
    heuristic_value TEXT,
    heuristic_confidence FLOAT,
    heuristic_source_location VARCHAR(200),
    
    -- User selection
    selected_source VARCHAR(50),           -- User's chosen source
    selected_value TEXT,                   -- Final value
    user_notes TEXT,
    selection_timestamp TIMESTAMP,
    
    -- Flags
    is_validated BOOLEAN DEFAULT FALSE,
    needs_review BOOLEAN DEFAULT FALSE,
    conflicting_sources BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_study_element_type ON data_elements(study_id, element_type);
CREATE INDEX idx_needs_review ON data_elements(needs_review);
```

**ExtractionConflict Table**
```sql
CREATE TABLE extraction_conflicts (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES studies(id) ON DELETE CASCADE,
    data_element_id INTEGER REFERENCES data_elements(id),
    conflict_type VARCHAR(50),             -- 'value_mismatch', 'missing_data'
    severity VARCHAR(20),                  -- 'low', 'medium', 'high'
    
    -- Conflicting values
    source_a VARCHAR(50),
    value_a TEXT,
    confidence_a FLOAT,
    source_b VARCHAR(50),
    value_b TEXT,
    confidence_b FLOAT,
    
    -- Resolution
    resolution_status VARCHAR(50),         -- 'pending', 'resolved'
    resolution_method VARCHAR(100),
    resolution_notes TEXT,
    resolved_by VARCHAR(100),
    resolved_date TIMESTAMP,
    created_date TIMESTAMP
);
CREATE INDEX idx_study_conflicts ON extraction_conflicts(study_id, resolution_status);
```

### **Powerful Querying Examples**

```python
# Find all 12-week primary outcomes across studies
from app import OutcomeTimepoint
outcomes_12w = OutcomeTimepoint.query.filter(
    OutcomeTimepoint.timepoint_value == 12,
    OutcomeTimepoint.timepoint_unit == 'weeks',
    OutcomeTimepoint.timepoint_type == 'primary'
).all()

# Get significant outcomes with large effect sizes
significant = OutcomeTimepoint.query.filter(
    OutcomeTimepoint.p_value < 0.05,
    OutcomeTimepoint.effect_estimate.isnot(None)
).order_by(OutcomeTimepoint.effect_estimate.desc()).all()

# Compare extraction methods for a study
from app import ExtractionSource
sources = ExtractionSource.query.filter_by(study_id=1).all()
for source in sources:
    print(f"{source.extraction_method}: confidence={source.confidence_score}")

# Find conflicting extractions needing review
from app import DataElement
conflicts = DataElement.query.filter(
    DataElement.conflicting_sources == True,
    DataElement.is_validated == False
).all()

# Meta-analysis ready export with arm-specific data
outcomes = OutcomeTimepoint.query.join(Outcome).filter(
    Outcome.outcome_type == 'primary',
    OutcomeTimepoint.timepoint_type == 'primary'
).all()

for outcome in outcomes:
    arms = outcome.results_by_arm  # JSON with per-arm statistics
    print(f"{outcome.outcome.outcome_name}: {len(arms)} arms")
```

---

## 🔧 System Requirements

### **Software Dependencies**
- **Python 3.11+** with virtual environment
- **PostgreSQL 18+** database server
- **Tesseract OCR v5.5+** for image text extraction
- **Poppler 25.07+** for PDF-to-image conversion
- **OpenAI API access** for GPT-4 analysis

### **Hardware Recommendations**
- **RAM**: 8GB+ (16GB recommended for large PDFs)
- **Storage**: 10GB+ for database and temporary files
- **CPU**: Multi-core processor for OCR processing

---

## 📊 Data Extraction Standards

### **Cochrane Systematic Review Compliance**
✅ Multiple timepoints per outcome  
✅ Primary vs secondary endpoint designation  
✅ Interim analysis detection  
✅ Source citation for every data point  
✅ Effect size extraction with confidence intervals  
✅ Meta-analysis ready data export  
✅ CONSORT flow diagram data capture  

### **Supported Clinical Trial Types**
- Randomized controlled trials (parallel, crossover, cluster)
- Dose-response studies
- Safety and efficacy trials
- Biomarker studies
- Patient-reported outcome studies

### **Supported Outcome Types**
- **Continuous**: Means, SDs, medians, IQRs, confidence intervals
- **Dichotomous**: Events, totals, proportions, odds ratios, relative risks
- **Time-to-event**: Survival analysis, hazard ratios, Kaplan-Meier data

---

## 📋 Version History

### **v2.0 - Major Architecture Overhaul** (Current)

#### 🔧 **Core Fixes**
- **CRITICAL**: Fixed ensemble logic that was discarding 80% of extracted data
- **Database**: Resolved varchar(50) field truncation causing data loss
- **Population Data**: Demographics now properly extracted from tables and patterns

#### ✨ **New Features**
- **Process Transparency**: Real-time diagnostics showing what each method extracts
- **Source Attribution**: Every data point traceable to specific pages/tables
- **Enhanced Demographics**: Age, gender, sample sizes from multiple extraction sources
- **Smart Token Management**: Automatic OpenAI prompt optimization within rate limits

#### 📊 **Quality Improvements**
- **Extraction Confidence**: Now achieving 90%+ confidence levels consistently
- **Multi-Method Validation**: LLM + direct parsing + pattern matching + OCR
- **Statistical Pattern Matching**: Direct regex extraction of p-values, CIs, effect sizes
- **Table-Based Demographics**: Baseline characteristics parsed directly from tables

#### 🔍 **Technical Enhancements**
- **Enhanced Error Handling**: Graceful database constraint and API limit handling
- **Improved OCR**: Better text extraction from images and complex formatting
- **Robust Table Detection**: Multiple algorithms for comprehensive table parsing
- **Performance Monitoring**: Token usage tracking and method success indicators

### **v1.x - Initial Implementation**
- Basic PDF text extraction with LLM analysis
- Simple web interface for file upload and results display
- PostgreSQL database integration
- Basic outcome and timepoint extraction

---

## 🛠️ Troubleshooting & Tips

### **Common Issues & Solutions**

**Database Connection Errors**
```bash
# Check PostgreSQL is running
pg_isready

# Verify connection string in .env
DATABASE_URL=postgresql://username:password@localhost/clinical_trials

# Test connection
psql -U username -d clinical_trials
```

**OpenAI API Issues**
- Verify API key is valid: `echo $OPENAI_API_KEY`
- Check account has credits: https://platform.openai.com/usage
- Monitor token usage in web interface
- Ultra-compact prompt used for large PDFs (automatically)

**PDF Extraction Problems**
- **Low-quality scans**: OCR may struggle → Try higher resolution PDF
- **Complex tables**: May require manual review → Use source selection interface
- **Large files**: Processing may take 2-5 minutes → Monitor progress bar
- **Scanned PDFs**: OCR automatically runs → Allow extra time

**CSV Export Issues**
- **Missing data**: Check if outcome has `additional_data` JSON field
- **Duplicate rows**: Fixed in v2.2 with deduplication logic
- **Empty values**: Check source attribution to see if data was found

**Database Constraint Errors**
- **VARCHAR too small**: Fixed in v2.2 (increased to VARCHAR(200))
- **Foreign key violation**: Fixed in v2.2 (proper cascade deletes)
- Run migration: `python fix_column_sizes.py`

### **Performance Optimization Tips**

**For Large PDFs (>50 pages)**
- System automatically uses ultra-compact LLM prompt (15K tokens vs 22K)
- Extraction takes 3-5 minutes vs 1-2 minutes for small PDFs
- Consider splitting very large PDFs (>100 pages)

**For Batch Processing**
- Process during off-peak hours to reduce API costs
- Monitor token usage: ~15K-22K tokens per PDF = $0.30-0.45/PDF
- Use database export instead of re-extracting same studies

**For Best Extraction Quality**
1. **Upload high-quality PDFs** - Native digital PDFs work best
2. **Use complete papers** - Full text with tables and figures
3. **Review source selections** - Check conflicting extractions
4. **Validate critical data** - Manually verify primary outcomes
5. **Export to Excel** - Review arm-by-arm results before meta-analysis

### **Understanding Extraction Quality**

**Confidence Levels**
- **High (0.8-1.0)**: 🟢 Direct table extraction, clear patterns
- **Medium (0.6-0.8)**: 🟡 AI interpretation, pattern matching
- **Low (0.4-0.6)**: 🔴 OCR, ambiguous text, requires review
- **Very Low (<0.4)**: ⚫ Not found, missing data

**When to Trust Each Method**
- **📊 PDFPlumber**: Best for structured tables with clear formatting
- **🔍 Heuristic**: Best for statistical values in text (p-values, CIs)
- **🤖 LLM**: Best for complex interpretations, identifying primary outcomes
- **👁️ OCR**: Use for scanned PDFs or when other methods fail

**Conflict Resolution Strategy**
1. View all 4 source extractions in review tab
2. Check confidence scores (prefer high confidence)
3. Verify source locations (prefer table data)
4. Compare values for consistency
5. Select most reliable source or manually enter correct value

---
- **Count data**: Rate ratios, incidence rates

---

## 🔄 Version History

### **v2.0 - Enhanced Multi-Timepoint Support** (October 2025)
- Fixed ensemble logic: all extraction methods now contribute to results
- Enhanced multi-timepoint support with OutcomeTimepoints table
- Comprehensive demographics extraction from multiple sources
- Smart token management and prompt optimization
- Real-time process diagnostics in web interface

### **v1.0 - Core Functionality** (Initial Release)
- Basic PDF extraction with OCR and AI
- PostgreSQL database integration
- Web interface with studies library
- Excel export capabilities

---

## 🤝 Contributing

This system is designed for systematic reviewers and meta-analysis researchers. Contributions welcome for:
- Additional extraction methods
- Database query optimizations  
- UI/UX improvements
- Documentation enhancements

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🆘 Support

For issues or questions:
1. Check the database_query_examples.py for advanced usage
2. Review extraction logs in the terminal output
3. Verify all dependencies are correctly installed
4. Ensure PostgreSQL service is running

**GitHub Repository**: https://github.com/terlinge/clinical-trial-extractor