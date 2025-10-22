# 🚀 Installation Guide - Clinical Trial Data Extractor

Complete step-by-step installation guide for the enhanced multi-timepoint extraction system.

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [ ] Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+
- [ ] Administrator/sudo access for installing system dependencies
- [ ] Internet connection for downloading packages
- [ ] OpenAI API key (with GPT-4 access)
- [ ] 10GB+ free disk space

---

## 🔧 System Dependencies Installation

### **Windows Installation**

1. **Install Python 3.11+**
   ```powershell
   # Download from python.org and install
   # Verify installation
   python --version
   # Should show Python 3.11.x or higher
   ```

2. **Install PostgreSQL 18+**
   ```powershell
   # Download from postgresql.org
   # During installation, note the password for 'postgres' user
   # Verify installation
   pg_config --version
   ```

3. **Install Tesseract OCR**
   ```powershell
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   # Install to default location (C:\Program Files\Tesseract-OCR)
   # Add to PATH: C:\Program Files\Tesseract-OCR
   # Verify installation
   tesseract --version
   ```

4. **Install Poppler**
   ```powershell
   # Download from: https://blog.alivate.com.au/poppler-windows/
   # Extract to C:\Program Files\poppler
   # Add to PATH: C:\Program Files\poppler\Library\bin
   # Verify installation
   pdftoppm -h
   ```

### **macOS Installation**

1. **Install Python 3.11+**
   ```bash
   # Using Homebrew (recommended)
   brew install python@3.11
   python3.11 --version
   ```

2. **Install PostgreSQL**
   ```bash
   brew install postgresql@18
   brew services start postgresql@18
   createdb clinical_trials
   ```

3. **Install Tesseract OCR**
   ```bash
   brew install tesseract
   tesseract --version
   ```

4. **Install Poppler**
   ```bash
   brew install poppler
   pdftoppm -h
   ```

### **Linux (Ubuntu) Installation**

1. **Install Python 3.11+**
   ```bash
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3-pip
   python3.11 --version
   ```

2. **Install PostgreSQL**
   ```bash
   sudo apt install postgresql postgresql-contrib
   sudo systemctl start postgresql
   sudo -u postgres createdb clinical_trials
   ```

3. **Install Tesseract OCR**
   ```bash
   sudo apt install tesseract-ocr
   tesseract --version
   ```

4. **Install Poppler**
   ```bash
   sudo apt install poppler-utils
   pdftoppm -h
   ```

---

## 📦 Application Installation

### **Step 1: Clone Repository**
```bash
git clone https://github.com/terlinge/clinical-trial-extractor.git
cd clinical-trial-extractor
```

### **Step 2: Create Virtual Environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux  
python3.11 -m venv .venv
source .venv/bin/activate
```

### **Step 3: Install Python Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **Step 4: Configure Environment Variables**
Create `.env` file in the project root:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration  
DATABASE_URL=postgresql://postgres:your_password@localhost/clinical_trials

# Optional: Custom Settings
MAX_UPLOAD_SIZE=50MB
DEBUG_MODE=True
```

### **Step 5: Initialize Database**
```bash
python -c "from app import app, db; app.app_context().push(); db.create_all(); print('Database initialized!')"
```

### **Step 6: Test Installation**
```bash
python -c "
import pdfplumber, pytesseract, openai, psycopg2
from pdf2image import convert_from_path
print('✅ All dependencies working!')
"
```

---

## 🏃‍♂️ Running the Application

### **Option 1: Windows PowerShell Script (Recommended)**
```powershell
.\START_APP.ps1
```

### **Option 2: Manual Start**
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Start application
python app.py
```

### **Access the Application**
Open your browser and navigate to: **http://localhost:5000**

---

## 🧪 Testing the Installation

### **Basic Functionality Test**
1. Upload a sample PDF clinical trial paper
2. Click "Extract All Trial Data"
3. Monitor the real-time progress indicators
4. Review the extraction summary
5. Check the Studies Library for saved results

### **Database Verification**
```python
python database_query_examples.py
```

### **Component Testing**
```bash
# Test OCR
python -c "import pytesseract; print('OCR ready:', pytesseract.get_tesseract_version())"

# Test PDF processing  
python -c "import pdfplumber; print('PDF processing ready')"

# Test database connection
python -c "from app import app, db; app.app_context().push(); print('Database connected:', db.engine.url)"

# Test OpenAI API
python -c "from openai import OpenAI; client = OpenAI(); print('OpenAI API ready')"
```

---

## ⚠️ Troubleshooting

### **Common Issues & Solutions**

**1. Tesseract not found**
```bash
# Windows: Add to PATH or set TESSDATA_PREFIX
set TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata

# macOS/Linux: Install language data
sudo apt install tesseract-ocr-eng  # Linux
brew install tesseract-lang         # macOS
```

**2. Poppler not found**
```bash
# Windows: Verify PATH includes poppler/Library/bin
echo $env:PATH | Select-String poppler

# macOS/Linux: Reinstall poppler
brew reinstall poppler     # macOS
sudo apt reinstall poppler-utils  # Linux
```

**3. PostgreSQL connection failed**
```bash
# Check service status
sudo systemctl status postgresql  # Linux
brew services list | grep postgres  # macOS
Get-Service postgresql*           # Windows PowerShell

# Reset database
dropdb clinical_trials && createdb clinical_trials
```

**4. OpenAI API issues**
```bash
# Verify API key
python -c "import os; print('API key set:', bool(os.getenv('OPENAI_API_KEY')))"

# Test API access
python -c "from openai import OpenAI; client = OpenAI(); print(client.models.list().data[0].id)"
```

**5. Memory issues with large PDFs**
- Increase system RAM allocation
- Process PDFs in smaller chunks
- Use focused extraction mode (automatic for large files)

---

## 🔧 Advanced Configuration

### **Custom Database Setup**
```bash
# Create custom database
createdb my_clinical_trials

# Update .env file
DATABASE_URL=postgresql://username:password@localhost/my_clinical_trials
```

### **Performance Optimization**
```env
# .env optimizations
OCR_THREAD_COUNT=4
MAX_PDF_PAGES=50
EXTRACTION_TIMEOUT=300
ENABLE_CACHING=True
```

### **Development Mode**
```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black app.py

# Check code quality
flake8 app.py
```

---

## 📊 Verification Checklist

After installation, verify these components work:

- [ ] **Web Interface**: Loads at http://localhost:5000
- [ ] **PDF Upload**: Can select and upload PDF files
- [ ] **OCR Processing**: Extracts text from images/figures
- [ ] **AI Extraction**: GPT-4 analyzes clinical trial data
- [ ] **Database Storage**: Studies save to PostgreSQL
- [ ] **Multi-timepoint Support**: Captures multiple measurement times
- [ ] **Source Citations**: Tracks data source locations
- [ ] **Export Functions**: JSON download works
- [ ] **Studies Library**: Browse saved extractions
- [ ] **Real-time Progress**: Status updates during extraction

---

## 🆘 Getting Help

If you encounter issues:

1. **Check logs**: Review terminal output for detailed error messages
2. **Verify dependencies**: Ensure all system dependencies are installed
3. **Test components**: Use the testing commands above
4. **Check permissions**: Ensure write access to project directory
5. **Review configuration**: Verify .env file settings

**Need support?** Create an issue on the GitHub repository with:
- Operating system and version
- Python version
- Complete error message
- Steps to reproduce the issue

---

**🎉 Installation Complete!**

You now have a fully functional Cochrane-compliant clinical trial data extraction system with multi-timepoint support, enhanced source tracking, and advanced querying capabilities.