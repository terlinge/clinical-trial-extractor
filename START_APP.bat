@echo off
REM ============================================
REM Clinical Trial Extractor - One-Click Startup
REM ============================================

cd /d "%~dp0"

echo.
echo ========================================
echo  Clinical Trial Extractor
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo First time setup detected...
    echo Creating Python virtual environment...
    python -m venv venv
    
    echo Installing dependencies...
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo.
    echo Setup complete!
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env exists
if not exist ".env" (
    echo.
    echo ==========================================
    echo  FIRST TIME SETUP
    echo ==========================================
    echo.
    echo Please create a .env file with your settings.
    echo Opening .env.example for you...
    echo.
    if exist ".env.example" (
        notepad .env.example
    )
    echo.
    echo After setting your API keys, save as .env
    echo Then run this script again.
    echo.
    pause
    exit /b
)

REM Check if PostgreSQL is running
echo Checking PostgreSQL...
sc query postgresql-x64-15 >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: PostgreSQL service doesn't appear to be running.
    echo Starting PostgreSQL...
    net start postgresql-x64-15
    timeout /t 3 >nul
)

REM Check if database is initialized
echo Checking database...
python -c "from app import db, app; app.app_context().push(); db.create_all(); print('Database ready!')" 2>nul
if errorlevel 1 (
    echo Initializing database...
    flask init-db
)

REM Start the backend server
echo.
echo ========================================
echo  Starting Backend Server...
echo ========================================
echo.
echo Backend will be available at: http://localhost:5000
echo Frontend will open in your browser shortly...
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Open browser after 3 seconds
start "" cmd /c "timeout /t 3 >nul & start http://localhost:5000"

REM Start the Flask app
python app.py

pause