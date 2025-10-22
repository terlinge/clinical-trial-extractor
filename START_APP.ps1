#!/usr/bin/env pwsh
# PowerShell startup script for Clinical Trial Extractor

Set-Location "C:\Users\TateErlinger\Git Repository Destination\clinical-trial-extractor"

Write-Host ""
Write-Host "========================================"
Write-Host " Clinical Trial Extractor"
Write-Host "========================================"
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

# Check PostgreSQL
Write-Host "Checking PostgreSQL..."
$service = Get-Service -Name "postgresql-x64-18" -ErrorAction SilentlyContinue
if (-not $service -or $service.Status -ne "Running") {
    Write-Host "Starting PostgreSQL..."
    Start-Service "postgresql-x64-18"
    Start-Sleep 3
}

# Check database
Write-Host "Checking database..."
python -c "from app import db, app; app.app_context().push(); db.create_all(); print('Database ready!')"

Write-Host ""
Write-Host "========================================"
Write-Host " Starting Backend Server..."
Write-Host "========================================"
Write-Host ""
Write-Host "Backend available at: http://localhost:5000"
Write-Host "Press Ctrl+C to stop the server"
Write-Host "========================================"
Write-Host ""

# Start Flask app
python app.py