@echo off
REM Check if Python is installed
echo Checking for Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not added to PATH. Please install Python and try again.
    exit /b 1
)

REM Create a virtual environment if it doesn't already exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip to the latest version
echo Upgrading pip...
pip install --upgrade pip

REM Install required packages
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Please create it and try again.
    exit /b 1
)

REM Run the Python script
if exist Scripts/LEO_demo.py (
    echo Running your script...
    python Scripts/LEO_demo.py
) else (
    echo Scripts/LEO_demo.py not found. Please place the script in this directory and try again.
    exit /b 1
)

REM Deactivate the virtual environment
echo Deactivating virtual environment...
deactivate
