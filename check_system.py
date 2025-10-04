#!/usr/bin/env python3
"""
System Checker for Clinical Trial Extractor
Checks if all required software is installed
"""

import sys
import subprocess
import shutil

def check_command(command, name, install_instruction):
    """Check if a command exists"""
    if shutil.which(command):
        try:
            result = subprocess.run([command, '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            version = result.stdout.split('\n')[0] if result.stdout else result.stderr.split('\n')[0]
            print(f"✅ {name}: {version}")
            return True
        except:
            print(f"⚠️  {name}: Installed but couldn't get version")
            return True
    else:
        print(f"❌ {name}: NOT FOUND")
        print(f"   Install: {install_instruction}")
        return False

def check_python_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        print(f"✅ Python package '{package_name}' is installed")
        return True
    except ImportError:
        print(f"❌ Python package '{package_name}' is NOT installed")
        return False

print("=" * 60)
print("CLINICAL TRIAL EXTRACTOR - SYSTEM CHECK")
print("=" * 60)

# Check Python
print("\n📌 Checking Python...")
print(f"✅ Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 9):
    print("⚠️  Warning: Python 3.9+ recommended")

# Check PostgreSQL
print("\n📌 Checking PostgreSQL...")
check_command('psql', 'PostgreSQL', 
              'Windows: https://www.postgresql.org/download/windows/\n   Mac: brew install postgresql')

# Check Tesseract
print("\n📌 Checking Tesseract OCR...")
check_command('tesseract', 'Tesseract', 
              'Windows: https://github.com/UB-Mannheim/tesseract/wiki\n   Mac: brew install tesseract')

# Check Poppler
print("\n📌 Checking Poppler (PDF tools)...")
poppler_found = check_command('pdftoppm', 'Poppler', 
                               'Windows: Download from https://blog.alivate.com.au/poppler-windows/\n   Mac: brew install poppler')

# Check Git (optional but useful)
print("\n📌 Checking Git (optional)...")
check_command('git', 'Git', 
              'https://git-scm.com/downloads')

# Check Python packages
print("\n📌 Checking Python Packages...")
important_packages = [
    'flask',
    'openai',
    'pdfplumber',
    'pytesseract',
    'sqlalchemy',
]

packages_installed = []
for package in important_packages:
    if check_python_package(package):
        packages_installed.append(package)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if poppler_found:
    print("\n✅ All system dependencies look good!")
else:
    print("\n⚠️  Some dependencies are missing. Install them before proceeding.")

if len(packages_installed) < len(important_packages):
    print("⚠️  Some Python packages are missing (this is normal - we'll install them)")
else:
    print("✅ All major Python packages already installed!")

print("\n" + "=" * 60)
print("Next step: Run the installation script!")
print("=" * 60)