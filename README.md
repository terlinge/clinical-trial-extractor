# 🏥 Clinical Trial Data Extractor

**AI-Powered Extraction System for Systematic Reviews & Network Meta-Analyses**

A comprehensive web application that extracts clinical trial data from PDF papers using multiple AI and OCR techniques, designed to meet Cochrane systematic review standards.

---

## ✨ Key Features

### 🔍 **Multi-Method PDF Extraction**
- **OCR Integration**: Tesseract OCR for extracting text from figures and images
- **AI-Powered Analysis**: OpenAI GPT-4 with specialized clinical trial prompts
- **Table Extraction**: PDFPlumber + Camelot for comprehensive table detection
- **Smart Content Prioritization**: Focused extraction for large documents
- **Source Citation Tracking**: Every extracted value includes source page/table reference

### 📊 **Cochrane-Compliant Data Structure**
- **Multiple Timepoints**: Same outcome measured at different times (4 weeks, 12 weeks, 6 months)
- **Timepoint Classification**: Primary, secondary, interim, follow-up, post-hoc analyses
- **Universal Outcome Types**: Continuous, dichotomous, time-to-event, count data
- **Effect Estimates**: Between-group comparisons with confidence intervals and p-values
- **Source Tracking**: Detailed citation tracking for systematic review requirements

### 💾 **Advanced Database Architecture**
- **PostgreSQL Backend**: Robust relational database with indexed querying
- **Dual-Table Structure**: Outcomes + TimePoints for efficient multi-timepoint storage
- **Meta-Analysis Ready**: Direct export capabilities for systematic review software
- **Complex Queries**: Filter by timepoint, effect size, significance, confidence levels

### 🌐 **User-Friendly Web Interface**
- **Real-Time Progress**: Live extraction progress with detailed status updates
- **Extraction Summary**: Comprehensive feedback on what was found and extracted
- **Studies Library**: Browse, search, and manage extracted studies
- **Study Comparison**: Side-by-side comparison of multiple studies
- **Export Options**: JSON, Excel with CONSORT-compliant worksheets

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

## 📁 Project Structure

```
clinical-trial-extractor/
├── app.py                      # Main Flask application
├── templates/
│   └── index.html             # Web interface
├── requirements.txt           # Python dependencies
├── START_APP.ps1             # Windows startup script
├── backup_database.py        # Database backup utility
├── database_query_examples.py # Advanced querying examples
├── .env                      # Environment variables (create this)
└── uploads/                  # Temporary PDF storage
```

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

## 🗄️ Database Schema

### **Enhanced Multi-Timepoint Architecture**

**Studies Table**
- Study identification (title, authors, journal, year, DOI)
- Study design (type, blinding, randomization, duration)
- Source tracking and metadata

**Outcomes Table**
- Outcome metadata (name, type, planned timepoints)
- Links to multiple timepoints

**OutcomeTimepoints Table** ⭐ *New Enhanced Table*
- Individual timepoint data (value, unit, type)
- Statistical results (means, SDs, CIs, p-values)
- Effect estimates and between-group comparisons
- Source citations and confidence levels

### **Powerful Querying Examples**
```python
# Find all 12-week primary outcomes
outcomes_12w = OutcomeTimepoint.query.filter(
    timepoint_value == 12, 
    timepoint_unit == 'weeks',
    timepoint_type == 'primary'
).all()

# Compare short-term vs long-term effects
short_term = OutcomeTimepoint.query.filter(timepoint_value <= 4).all()
long_term = OutcomeTimepoint.query.filter(timepoint_value >= 12).all()

# Meta-analysis ready export
significant_outcomes = OutcomeTimepoint.query.filter(
    p_value < 0.05, outcome_type == 'secondary'
).all()
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
- **Count data**: Rate ratios, incidence rates

---

## 🔄 Version History

### **v2.0 - Enhanced Multi-Timepoint Support** (Current)
- Added Cochrane-compliant multiple timepoint extraction
- Enhanced database schema with OutcomeTimepoints table
- Implemented smart prompt management for large documents
- Added comprehensive source citation tracking
- Improved UI with real-time progress and extraction summaries

### **v1.0 - Core Functionality**
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