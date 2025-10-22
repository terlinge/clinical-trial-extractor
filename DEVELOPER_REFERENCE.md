# 🔧 Developer Quick Reference

## 🚀 **Quick Start Commands**

```bash
# Start application
.\START_APP.ps1

# Activate environment manually
.\.venv\Scripts\Activate.ps1

# Test extraction
python -c "from app import EnsembleExtractor; e = EnsembleExtractor(); print('Ready')"

# Check database
python -c "from app import db, app; app.app_context().push(); db.create_all(); print('DB ready')"
```

## 🎯 **Key Methods & Classes**

### **Main Extraction Flow**
```python
# app.py lines 1224-1270
EnsembleExtractor.extract_comprehensive(pdf_path)
├── PDFExtractor.comprehensive_extract()      # Lines 300-396
├── LLMExtractor.extract_trial_data()         # Lines 406-550  
├── HeuristicExtractor.extract_statistical_values()  # Lines 1104-1130
└── _ensemble_results()                       # Lines 1274-1330
```

### **Critical Code Locations**

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Ensemble Logic** | `app.py` | 1274-1330 | Combines all extraction methods |
| **Participant Flow** | `app.py` | 1395-1450 | Extracts screened/randomized/analyzed |
| **Pattern Matching** | `app.py` | 1104-1130 | Regex patterns for statistical data |
| **LLM Prompts** | `app.py` | 550-700 | GPT-4 extraction prompts |
| **Confidence Scoring** | `app.py` | 1485-1600 | Quality assessment algorithm |

## 🔍 **Debug & Testing**

### **Test Individual Components**
```python
# Test PDF extraction
from app import PDFExtractor
pdf = PDFExtractor('test.pdf')
data = pdf.comprehensive_extract()
print(f"Text: {len(data['text'])} chars, Tables: {len(data['tables'])}")

# Test heuristic patterns  
from app import HeuristicExtractor
h = HeuristicExtractor()
stats = h.extract_statistical_values(text)
print(f"P-values: {stats.get('p_values', [])}")
print(f"Screened: {stats.get('screened', [])}")

# Test LLM extraction
from app import LLMExtractor
llm = LLMExtractor()
result = llm.extract_trial_data(text)
print(f"Tokens used: {llm.tokens_used}")
```

### **Common Debug Patterns**
```python
# Check extraction metadata
data, metadata = extractor.extract_comprehensive('test.pdf')
print(json.dumps(metadata, indent=2))

# Verify participant flow extraction
if 'participants' in data:
    flow = data['participants']
    print(f"Screened: {flow.get('total_screened')} ({flow.get('screened_source')})")
    print(f"Randomized: {flow.get('total_randomized')} ({flow.get('randomized_source')})")

# Monitor token usage
print(f"LLM tokens: {metadata.get('llm_tokens', 0)}")
print(f"Estimated cost: ${metadata.get('llm_tokens', 0) * 0.00003:.4f}")
```

## 📊 **Database Schema Quick Reference**

### **Key Tables**
```sql
-- Main studies
SELECT title, confidence_scores, extraction_date FROM studies;

-- Population data (JSON field)
SELECT population_data->'total_screened' as screened,
       population_data->'total_randomized' as randomized
FROM studies;

-- Extraction metadata
SELECT extraction_metadata->'llm_tokens' as tokens,
       extraction_metadata->'confidence_scores' as confidence  
FROM studies;
```

### **Useful Queries**
```python
# Get latest study with full metadata
from app import Study, db
latest = Study.query.order_by(Study.extraction_date.desc()).first()
print(f"Title: {latest.title}")
print(f"Confidence: {latest.confidence_scores}")
print(f"Flow data: {latest.population_data}")
```

## ⚙️ **Configuration Quick Reference**

### **Environment Variables**
```bash
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost/clinical_trials
FLASK_ENV=development
```

### **Key Constants**
```python
# Token limits
MAX_TOKENS = 30000        # OpenAI hard limit
TOKEN_WARNING = 25000     # Start optimization
TOKEN_YELLOW = 20000      # Selective reduction

# Processing limits  
MAX_TABLES = 10           # Table extraction limit
MAX_PAGE_CHARS = 3000     # Per-page context
TRUNCATION_LENGTH = 50    # Database field limit
```

## 🚨 **Common Issues & Fixes**

### **"Missing participant flow data"**
```python
# Check patterns are working
text = "156 patients were screened, 120 randomized"
import re
screened = re.findall(r'(\d+)\s*(?:patients?)?\s*screened', text, re.I)
randomized = re.findall(r'(\d+)\s*(?:patients?)?\s*randomized', text, re.I)
print(f"Found: screened={screened}, randomized={randomized}")
```

### **"Database field too long"**
```python
# Use safe truncation
def _safe_truncate(value, max_length=50):
    if value and len(str(value)) > max_length:
        return str(value)[:max_length-3] + "..."
    return value
```

### **"Token limit exceeded"**
```python
# Check prompt size
prompt_tokens = len(prompt) // 4
if prompt_tokens > 25000:
    print("⚠️ Need to optimize prompt")
    # Use focused extraction
```

### **"Low confidence scores"**
```python
# Check data completeness
confidence = metadata['confidence_scores']['by_section']
for section, score in confidence.items():
    if score < 0.7:
        print(f"⚠️ {section}: {score:.2f} - needs improvement")
```

## 🎯 **Performance Monitoring**

### **Key Metrics**
```python
# Extraction time
start = time.time()
data, metadata = extractor.extract_comprehensive(pdf_path)
duration = time.time() - start
print(f"Extraction time: {duration:.1f}s")

# API costs
tokens = metadata.get('llm_tokens', 0)
cost = tokens * 0.00003  # GPT-4 pricing
print(f"API cost: ${cost:.4f}")

# Data quality
overall_confidence = metadata['confidence_scores']['overall']
print(f"Confidence: {overall_confidence:.1%}")
```

## 📝 **Adding New Features**

### **New Heuristic Pattern**
```python
# Add to HeuristicExtractor.extract_statistical_values()
patterns = {
    # ... existing patterns ...
    'your_new_pattern': r'your_regex_here',
}
```

### **New Confidence Metric**
```python
# Add to _calculate_confidence()
new_score = 0
if final_data.get('your_field'):
    new_score += 50
scores['by_section']['your_section'] = new_score / 100.0
```

### **New Database Field**
```python
# Add to Study model
your_field = db.Column(db.String(255))

# Add to _save_to_database()
study.your_field = extracted_data.get('your_field')
```

---

*Quick Reference v2.1 - Updated January 26, 2025*