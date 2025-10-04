@echo off
REM ============================================
REM Clinical Trial Extractor - Easy Installation
REM ============================================

echo.
echo ========================================================
echo  CLINICAL TRIAL EXTRACTOR - EASY INSTALLATION
echo ========================================================
echo.
echo This script will help you install everything you need.
echo.
pause

REM Check Python
echo.
echo [Step 1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation!
    pause
    exit /b 1
)
python --version
echo Python is installed!

REM Check PostgreSQL
echo.
echo [Step 2/6] Checking PostgreSQL...
psql --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: PostgreSQL is not installed.
    echo.
    set /p install_pg="Would you like help installing PostgreSQL? (y/n): "
    if /i "%install_pg%"=="y" (
        echo.
        echo Please download and install PostgreSQL from:
        echo https://www.postgresql.org/download/windows/
        echo.
        echo After installation:
        echo 1. Remember the password you set for 'postgres' user
        echo 2. Make sure PostgreSQL service is running
        echo.
        pause
        start https://www.postgresql.org/download/windows/
        echo.
        echo Run this script again after PostgreSQL is installed.
        pause
        exit /b 0
    )
) else (
    psql --version
    echo PostgreSQL is installed!
)

REM Check Tesseract
echo.
echo [Step 3/6] Checking Tesseract OCR...
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Tesseract is not installed.
    echo.
    set /p install_tess="Would you like help installing Tesseract? (y/n): "
    if /i "%install_tess%"=="y" (
        echo.
        echo Please download and install Tesseract from:
        echo https://github.com/UB-Mannheim/tesseract/wiki
        echo.
        echo IMPORTANT: During installation, note the installation path
        echo (usually C:\Program Files\Tesseract-OCR)
        echo.
        pause
        start https://github.com/UB-Mannheim/tesseract/wiki
        echo.
        echo After installation, add Tesseract to your PATH or
        echo we'll configure it in the .env file.
        pause
    )
) else (
    tesseract --version | findstr "tesseract"
    echo Tesseract is installed!
)

REM Create virtual environment
echo.
echo [Step 4/6] Creating Python virtual environment...
if exist "venv\" (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo Virtual environment created!
)

REM Install Python packages
echo.
echo [Step 5/6] Installing Python packages...
echo This may take a few minutes...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo Python packages installed!

REM Create .env file
echo.
echo [Step 6/6] Setting up configuration...
if exist ".env" (
    echo Configuration file already exists.
) else (
    if exist ".env.example" (
        copy .env.example .env
        echo.
        echo Created .env file from template.
        echo IMPORTANT: You need to edit .env and add your OpenAI API key!
        echo.
        echo Opening .env file for you to edit...
        timeout /t 2 >nul
        notepad .env
    ) else (
        echo Creating default .env file...
        (
            echo # Clinical Trial Extractor Configuration
            echo OPENAI_API_KEY=your-api-key-here
            echo DATABASE_URL=postgresql://postgres:postgres@localhost/clinical_trials
            echo FLASK_ENV=development
        ) > .env
        echo.
        echo Created .env file.
        echo Opening for you to edit...
        timeout /t 2 >nul
        notepad .env
    )
)

REM Create database
echo.
echo Creating database...
set /p db_password="Enter your PostgreSQL password (default: postgres): "
if "%db_password%"=="" set db_password=postgres

set PGPASSWORD=%db_password%
psql -U postgres -c "CREATE DATABASE clinical_trials;" >nul 2>&1
if errorlevel 1 (
    echo Database already exists or creation failed.
    echo This is OK if the database already exists.
) else (
    echo Database created!
)

REM Initialize database schema
echo Initializing database schema...
python -c "from app import db, app; app.app_context().push(); db.create_all(); print('Database schema created!')"

REM Create desktop shortcut
echo.
set /p create_shortcut="Create desktop shortcut? (y/n): "
if /i "%create_shortcut%"=="y" (
    set SCRIPT="%TEMP%\create_shortcut.vbs"
    echo Set oWS = WScript.CreateObject("WScript.Shell") > %SCRIPT%
    echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\Clinical Trial Extractor.lnk" >> %SCRIPT%
    echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
    echo oLink.TargetPath = "%CD%\START_APP.bat" >> %SCRIPT%
    echo oLink.WorkingDirectory = "%CD%" >> %SCRIPT%
    echo oLink.Description = "Clinical Trial Data Extractor" >> %SCRIPT%
    echo oLink.Save >> %SCRIPT%
    cscript //nologo %SCRIPT%
    del %SCRIPT%
    echo Desktop shortcut created!
)

echo.
echo ========================================================
echo  INSTALLATION COMPLETE!
echo ========================================================
echo.
echo Next steps:
echo 1. Make sure you've added your OpenAI API key to .env
echo 2. Double-click "START_APP.bat" to run the application
echo 3. Or double-click the desktop shortcut if you created one
echo.
echo The app will open automatically in your browser!
echo.
pause