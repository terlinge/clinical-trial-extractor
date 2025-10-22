# 🏗️ Clinical Trial Extractor: Technical Architecture & Workflow

## 📋 **Table of Contents**
- [System Overview](#system-overview)
- [Extraction Pipeline](#extraction-pipeline)
- [Decision Logic](#decision-logic)
- [Data Prioritization](#data-prioritization)
- [Confidence Scoring](#confidence-scoring)
- [Source Attribution](#source-attribution)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting Guide](#troubleshooting-guide)

---

## 🎯 **System Overview**

The Clinical Trial Extractor v2.1 uses a **5-stage ensemble architecture** that combines multiple extraction methods to achieve 90%+ data extraction confidence. Each method contributes unique strengths:

- **📝 PDFPlumber**: Clean text extraction (50k-150k chars)
- **🔍 Tesseract OCR**: Image/scanned content extraction  
- **📊 Table Detection**: Structured data parsing
- **🎯 Heuristic Patterns**: Direct regex pattern matching
- **🤖 GPT-4 LLM**: Intelligent contextual analysis

---

## 🔄 **Extraction Pipeline**

### **Stage 1: PDF Data Extraction**
```python
PDFExtractor.comprehensive_extract()
├── PDFPlumber text extraction
│   ├── Clean text content (50k-150k characters)
│   ├── Page-by-page mapping
│   └── Document structure preservation
├── Tesseract OCR processing
│   ├── Image text extraction
│   ├── Scanned document handling
│   └── Complex formatting fallback
└── Table detection & parsing
    ├── Multiple detection algorithms
    ├── Row/column structure parsing
    └── Baseline characteristics tables
```

**Key Files**: `app.py` lines 300-396 (PDFExtractor class)

### **Stage 2: LLM Analysis**
```python
LLMExtractor.extract_trial_data()
├── Smart prompt management
│   ├── Token limit monitoring (30k max)
│   ├── Dynamic prompt optimization
│   └── Context prioritization
├── Context preparation
│   ├── Full document text
│   ├── Extracted tables (up to 10)
│   └── Page-by-page content (3k chars each)
├── GPT-4 Turbo API call
│   ├── Structured JSON extraction
│   ├── Temperature: 0.1 (low randomness)
│   └── Max tokens: 4000
└── Source attribution tracking
    ├── Page number mapping
    ├── Table reference tracking
    └── Confidence level assignment
```

**Decision Logic**:
- **>25k tokens**: Create focused prompt, prioritize statistical tables
- **>20k tokens**: Selective optimization, truncate redundant content
- **<20k tokens**: Full extraction with complete context

**Key Files**: `app.py` lines 397-550 (LLMExtractor class)

### **Stage 3: Heuristic Pattern Matching**
```python
HeuristicExtractor.extract_statistical_values()
├── Statistical patterns
│   ├── P-values: p = 0.001, p < 0.05
│   ├── Confidence intervals: 95% CI [1.2-3.4]
│   ├── Effect sizes: OR 2.1, RR 1.8, HR 0.7
│   └── Mean±SD: 65.3 (12.1), 45±8
├── Demographic patterns
│   ├── Age extraction: mean age 65.5 years
│   ├── Gender distribution: 62% male
│   └── Sample sizes: n=156, 200 patients
└── Participant flow patterns
    ├── Screened: "156 patients screened"
    ├── Randomized: "120 were randomized"
    ├── Analyzed: "115 completed analysis"
    └── CONSORT detection: flow diagram identification
```

**Pattern Examples**:
```regex
'screened': r'(\d+)\s*(?:patients?|participants?|subjects?)?\s*(?:were\s*)?screened'
'randomized_total': r'(\d+)\s*(?:patients?|participants?|subjects?)?\s*(?:were\s*)?randomized'
'p_values': r'p\s*[=<>]\s*([0-9.]+|0\.0*1)'
'confidence_intervals': r'95%?\s*CI[:\s]*\[?([0-9.]+)\s*[-–to]\s*([0-9.]+)\]?'
```

**Key Files**: `app.py` lines 1101-1215 (HeuristicExtractor class)

### **Stage 4: Ensemble Combination**
```python
_ensemble_results(llm_data, heuristic_data, pdf_data)
├── Start with LLM structure (baseline)
├── Enhance with direct extractions
│   ├── Add PDF tables
│   ├── Merge statistical values from patterns
│   ├── Combine demographics from all sources
│   └── Integrate participant flow data
├── Source attribution
│   ├── Track data origin for every field
│   ├── Assign reliability scores
│   └── Maintain extraction method metadata
└── Conflict resolution
    ├── Prioritize by source reliability
    ├── Take maximum values for counts
    └── Flag contradictions with low confidence
```

**Key Files**: `app.py` lines 1274-1350 (Ensemble method)

### **Stage 5: Confidence Scoring**
```python
_calculate_confidence(final_data, pdf_data, heuristic_data)
├── Section-by-section scoring (0-100%)
├── Weighted overall confidence
└── Data completeness validation
```

---

## 🧠 **Decision Logic**

### **Token Management (LLM Stage)**
```python
estimated_tokens = len(prompt) // 4

if estimated_tokens > 25000:     # Red zone
    # Create focused prompt, prioritize tables
    prompt = _create_focused_extraction_prompt()
elif estimated_tokens > 20000:  # Yellow zone  
    # Selective optimization, reduce redundancy
    # Truncate page context, preserve tables
else:                           # Green zone
    # Use full prompt with complete context
```

### **Data Fusion Priority**
```python
# Participant Flow Numbers
Priority 1: Heuristic regex patterns (highest reliability)
Priority 2: Table extraction (structured data)
Priority 3: LLM interpretation (contextual validation)

# Demographics (Age/Gender)  
Priority 1: Table baseline characteristics
Priority 2: Heuristic pattern matching
Priority 3: LLM extraction from text

# Statistical Values
Priority 1: Direct regex patterns (p-values, CIs)
Priority 2: Table numerical data
Priority 3: LLM statistical interpretation
```

---

## 📊 **Data Prioritization**

### **Source Reliability Hierarchy**

| Data Type | Priority 1 | Priority 2 | Priority 3 |
|-----------|------------|------------|------------|
| **Participant Flow** | Heuristic Patterns | Table Extraction | LLM Analysis |
| **Demographics** | Baseline Tables | Pattern Matching | LLM Extraction |
| **Statistical Values** | Direct Regex | Table Numbers | LLM Interpretation |
| **Study Design** | LLM Context | Keyword Matching | Table Metadata |

### **Conflict Resolution Rules**

1. **Numerical Conflicts**: Take highest/most complete value
2. **Missing Data**: Combine all available sources
3. **Contradictions**: Flag with low confidence, prioritize by source reliability
4. **Multiple Matches**: Use maximum values for counts, most detailed for descriptions

---

## 🎯 **Confidence Scoring Algorithm**

### **Section Scoring (0-100%)**
```python
# Study Identification (100%)
Title:           30 points
Authors:         20 points  
Year:            20 points
Journal:         15 points
Registration:    15 points

# Study Design (100%)
Type:            20 points
Blinding:        20 points
Randomization:   15 points
Duration:        15 points
Sites:           15 points
Country:         15 points

# Interventions (100%)
Multiple arms:   40 points
Completeness:    60 points (dose, schedule, duration)

# Outcomes (100%) - MOST CRITICAL
Primary outcomes: 30 points
Statistical data: 50 points
Source attribution: 20 points

# Population (100%)
Demographics:     40 points (age, gender)
Flow data:        35 points (screened, randomized, analyzed)
Baseline chars:   25 points
```

### **Overall Confidence**
```python
overall_score = (
    identification * 0.15 +
    design * 0.20 +
    interventions * 0.20 +
    outcomes * 0.30 +      # Highest weight
    population * 0.15
)
```

---

## 🔍 **Source Attribution System**

Every extracted data point includes source metadata:

### **Attribution Fields**
```python
{
    "total_screened": 156,
    "screened_source": "heuristic_pattern",
    "age_mean": 65.3,
    "age_source": "table_extraction", 
    "p_value": 0.023,
    "p_value_source": "regex_pattern",
    "primary_outcome": "Pain reduction",
    "outcome_source": "llm_analysis_page_12"
}
```

### **Source Types**
- `heuristic_pattern`: Direct regex match
- `table_extraction`: Parsed from tables
- `llm_analysis`: GPT-4 interpretation
- `llm_analysis_page_N`: LLM with page reference
- `assessed_eligibility_pattern`: Screening data variant
- `calculation`: Derived from other values

---

## ⚡ **Performance Optimization**

### **Memory Management**
```python
# Text chunking for large documents
if len(pdf_text) > 100000:
    # Process in 70k character chunks
    # Preserve table context
    # Maintain page boundaries

# Table processing limits
max_tables = 10          # Prevent memory overflow
max_table_rows = 20      # Limit table size
max_page_chars = 3000    # Per-page context limit
```

### **API Efficiency**
```python
# OpenAI token optimization
token_limit = 30000      # Hard limit
warning_threshold = 25000  # Start optimization
yellow_zone = 20000      # Selective reduction

# Response optimization
temperature = 0.1        # Low randomness
max_tokens = 4000       # Sufficient for JSON
timeout = 120           # 2-minute limit
```

---

## 🔧 **Troubleshooting Guide**

### **Common Issues**

#### **1. Missing Participant Flow Data**
```bash
# Check heuristic patterns
grep -n "screened\|randomized\|analyzed" extracted_text.txt

# Verify pattern matching
python -c "
import re
text = open('extracted_text.txt').read()
screened = re.findall(r'(\d+)\s*(?:patients?)?\s*screened', text, re.I)
print('Screened numbers found:', screened)
"
```

#### **2. Low Confidence Scores**
- **Check table extraction**: Verify tables are being detected
- **Review LLM prompts**: Ensure sufficient context
- **Validate patterns**: Test regex patterns against text

#### **3. Token Limit Exceeded**
```python
# Monitor token usage
print(f"Estimated tokens: {len(prompt) // 4}")
print(f"LLM tokens used: {llm_extractor.tokens_used}")

# Check optimization triggers
if len(prompt) > 100000:  # ~25k tokens
    # Implement focused extraction
```

#### **4. Database Field Truncation**
```python
# Use safe truncation
def _safe_truncate(value, max_length=50):
    if value and len(str(value)) > max_length:
        return str(value)[:max_length-3] + "..."
    return value
```

### **Debug Commands**
```bash
# Check extraction process
python -c "from app import EnsembleExtractor; e = EnsembleExtractor(); data, meta = e.extract_comprehensive('test.pdf'); print(meta)"

# Verify patterns
python -c "from app import HeuristicExtractor; h = HeuristicExtractor(); print(h.extract_statistical_values(open('text.txt').read()))"

# Test LLM connectivity
python -c "import openai; print(openai.api_key[:10] if openai.api_key else 'No API key')"
```

---

## 📝 **Configuration Files**

### **Key Configuration**
- **Environment**: `.env` (API keys, database URL)
- **Dependencies**: `requirements.txt`
- **Database**: PostgreSQL schema in `app.py`
- **Startup**: `START_APP.ps1` (Windows) or `START_APP.bat`

### **Important Constants**
```python
# app.py configuration
MAX_TOKENS = 30000        # OpenAI limit
MAX_TABLES = 10           # Table processing limit
CONFIDENCE_THRESHOLD = 0.7  # Minimum acceptable confidence
TRUNCATION_LENGTH = 50    # Database varchar limit
```

---

## 🚀 **Recent Improvements (v2.1)**

### **Architecture Fixes**
- ✅ Fixed ensemble logic discarding 80% of extracted data
- ✅ Added participant flow pattern matching
- ✅ Enhanced source attribution system
- ✅ Implemented smart database truncation

### **Quality Improvements**
- ✅ Confidence scores improved to 90%+ (from 60-70%)
- ✅ Comprehensive participant flow extraction
- ✅ Multi-source demographic validation
- ✅ Real-time process diagnostics

---

## 📚 **Additional Resources**

- **CHANGELOG.md**: Version history and improvements
- **README.md**: Installation and usage instructions
- **COMMIT_CHECKLIST.md**: Development workflow
- **database_query_examples.py**: Advanced querying examples

---

*Last Updated: January 26, 2025 - v2.1 Architecture Documentation*
*For technical support, refer to the troubleshooting section above.*