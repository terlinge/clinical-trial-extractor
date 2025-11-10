# Clinical Trial Extractor v2.2 - Multi-Source Data Architecture

## 🎯 Key Improvements Implemented

### 1. **Multi-Source Data Storage & User Control**
- **Problem Solved**: LLM interpretations were overwriting reliable table data
- **Solution**: Each extraction method (PDFPlumber, OCR, LLM, Heuristic) now stores data separately
- **User Control**: Users can review and select which source to use for each data element

### 2. **Enhanced User Interface** 
- **Clean Source Attribution**: Icons instead of verbose text (📊 Table, 🔍 Pattern, 🤖 AI, 👁️ OCR)
- **Tabular Data Display**: Structured tables for outcomes instead of verbose text
- **Conflict Highlighting**: Visual indicators when sources disagree
- **Selection Interface**: Users can choose preferred sources with confidence scores

### 3. **Improved Database Architecture**
- **ExtractionSource**: Stores raw data from each method with quality metrics
- **DataElement**: Individual fields with source attribution and user selections
- **ExtractionConflict**: Tracks disagreements between sources
- **UserPreferences**: Customizable selection rules and thresholds

## 🚀 New Features

### **Multi-Source Review Tab**
- View extraction results from all sources side-by-side
- Select preferred sources for each data element
- Auto-select recommendations based on confidence scores
- Manual conflict resolution interface

### **Enhanced Export Options**
- Cochrane RevMan format with source attribution
- CSV export with confidence scores
- JSON format with complete source metadata
- Option to include/exclude conflicting values

### **Smart Source Prioritization**
```python
SOURCE_PRIORITY_DEFAULT = [
    'pdfplumber',  # Structured table data (highest priority)
    'heuristic',   # Pattern matching from text
    'ocr',         # Visual extraction
    'llm'          # AI interpretation (lowest priority for conflicts)
]
```

## 🔧 API Endpoints Added

### Multi-Source Data Management
- `GET /api/studies/{id}/sources/detailed` - Get source-specific extraction data
- `GET /api/studies/{id}/data-elements` - Get individual fields with source attribution
- `POST /api/studies/{id}/data-elements/{id}/select` - Set user selection for a field
- `POST /api/studies/{id}/auto-select` - Auto-select based on confidence thresholds
- `GET /api/studies/{id}/conflicts` - Get extraction conflicts
- `POST /api/studies/{id}/finalize` - Apply user selections to generate final data

### Enhanced Extraction
- Modified `/api/extract` to return source data
- Added multi-source storage during extraction
- Improved source location tracking

## 📊 Database Schema

### **ExtractionSource Table**
```sql
CREATE TABLE extraction_sources (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES studies(id),
    extraction_method VARCHAR(50) NOT NULL,  -- 'pdfplumber', 'ocr', 'llm', 'heuristic'
    raw_data JSON,                          -- Complete extraction result
    confidence_score FLOAT,                 -- 0.0 - 1.0 confidence
    quality_metrics JSON,                   -- Method-specific quality indicators
    processing_time FLOAT,                  -- Seconds
    extraction_timestamp TIMESTAMP DEFAULT NOW()
);
```

### **DataElement Table**
```sql
CREATE TABLE data_elements (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES studies(id),
    element_type VARCHAR(50) NOT NULL,      -- 'outcome', 'intervention', 'demographic'
    element_name VARCHAR(200),              -- Specific field name
    element_path VARCHAR(500),              -- JSON path for nested data
    
    -- Values from different sources
    pdfplumber_value TEXT,
    pdfplumber_confidence FLOAT,
    pdfplumber_source_location VARCHAR(200),
    
    llm_value TEXT,
    llm_confidence FLOAT,
    llm_source_location VARCHAR(200),
    
    heuristic_value TEXT,
    heuristic_confidence FLOAT,
    heuristic_source_location VARCHAR(200),
    
    ocr_value TEXT,
    ocr_confidence FLOAT,
    ocr_source_location VARCHAR(200),
    
    -- User selection
    selected_source VARCHAR(50),
    selected_value TEXT,
    user_notes TEXT,
    selection_timestamp TIMESTAMP,
    
    -- Validation flags
    is_validated BOOLEAN DEFAULT FALSE,
    needs_review BOOLEAN DEFAULT FALSE,
    conflicting_sources BOOLEAN DEFAULT FALSE
);
```

## 🎯 Impact on Systematic Reviews

### **Before v2.2**
- Single extraction result with limited transparency
- Good table data could be overwritten by AI interpretation
- Verbose source attribution text
- No user control over data source selection

### **After v2.2**
- Multiple extraction sources with full transparency
- Users control which source to trust for each data element
- Clean icon-based source attribution
- Systematic review workflow with conflict resolution
- Tabular output format for better readability

## 🔄 Migration Process

1. **Run Migration Script**:
   ```bash
   python migrate_database.py
   ```

2. **New Tables Created**:
   - extraction_sources
   - data_elements
   - user_preferences
   - extraction_conflicts

3. **Existing Studies**: Automatically migrated with legacy source markers

## 🎮 User Workflow

1. **Upload PDF** → Multi-source extraction runs automatically
2. **Review Sources** → Switch to "Review & Select Data" tab
3. **Resolve Conflicts** → Choose preferred sources for conflicting values
4. **Auto-Select** → Apply confidence-based recommendations
5. **Finalize** → Generate final extraction with user selections
6. **Export** → Download results with source attribution

## 🏆 Benefits

### **For Systematic Reviewers**
- **Transparency**: See exactly where each data point came from
- **Control**: Choose which extraction method to trust
- **Quality**: Prevent good table data from being overwritten
- **Efficiency**: Auto-selection for high-confidence extractions

### **For Data Quality**
- **Conflict Detection**: Automatic identification of disagreements
- **Source Prioritization**: Prefer structured data over AI interpretation
- **Validation Tracking**: Know which fields have been reviewed
- **Audit Trail**: Complete record of extraction and selection process

## 📈 Future Enhancements

1. **Machine Learning**: Train models on user selections to improve auto-selection
2. **Collaborative Review**: Multi-user validation workflows
3. **Advanced Conflict Resolution**: Statistical agreement measures
4. **Integration**: Direct export to systematic review software
5. **Quality Metrics**: Real-time assessment of extraction reliability

## 🎉 Version 2.2 Achievement

✅ **Solved the fundamental flaw**: LLM no longer overwrites good table data  
✅ **User empowerment**: Full control over data source selection  
✅ **Transparency**: Clear source attribution for every data point  
✅ **Systematic review ready**: Workflow designed for evidence synthesis  
✅ **Quality assurance**: Conflict detection and resolution tools  

This represents a paradigm shift from automated extraction to **human-guided, source-transparent data selection** - exactly what systematic reviewers need for high-quality evidence synthesis.