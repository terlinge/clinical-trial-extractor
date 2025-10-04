\# Clinical Trial Data Extractor



\## Current Status

Fully functional web app for extracting clinical trial data using AI.



\## Location

`C:\\Users\\terli\\ClinicalTrialExtractor`



\## Quick Start

```bash

cd C:\\Users\\terli\\ClinicalTrialExtractor

venv\\Scripts\\activate

python app.py





What's Built



Multi-method PDF extraction (PDFPlumber, GPT-4, heuristics)

PostgreSQL database with full schema

3-tab UI: Extract, Library, Compare Studies

Excel export with 10+ CONSORT-compliant sheets

Studies library with search/filter

Study comparison view

Database backup script



Environment Setup



Python virtual environment: venv/

Database: PostgreSQL (database name: postgres)

API Keys in: .env file



GitHub

https://github.com/terlinge/clinical-trial-extractor



Key Files



app.py - Backend server

templates/index.html - Frontend UI

requirements.txt - Python dependencies

.env - API keys (not in git)

backup\_database.py - Database backup script

