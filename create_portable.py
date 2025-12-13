"""
Python Embedded 포터블 버전 생성
사용자가 EXE 하나만 더블클릭하면 모든 것이 자동으로 작동
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

def download_python_embedded():
    """Python Embedded 버전 다운로드"""
    print("Downloading Python Embedded...")
    python_url = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-embed-amd64.zip"
    zip_path = "python_embedded.zip"

    if not os.path.exists(zip_path):
        print(f"Downloading from {python_url}")
        urllib.request.urlretrieve(python_url, zip_path)
        print("Downloaded!")
    else:
        print("Python embedded already downloaded")

    return zip_path

def create_portable():
    """포터블 버전 생성"""
    print("=" * 60)
    print("Creating Portable EMR Helper")
    print("=" * 60)
    print()

    portable_dir = Path("EMR_Helper_Portable")

    # 1. 폴더 생성
    print("[1/6] Creating folder structure...")
    if portable_dir.exists():
        shutil.rmtree(portable_dir)
    portable_dir.mkdir()

    # 2. Python 소스 복사
    print("[2/6] Copying source files...")
    files_to_copy = [
        "main.py",
        "screen_capture.py",
        "ocr_extractor.py",
        "message_generator.py",
        "config.json",
        "requirements.txt",
    ]

    for file in files_to_copy:
        if Path(file).exists():
            shutil.copy(file, portable_dir)
            print(f"  ✓ {file}")

    # images 폴더 복사
    if Path("images").exists():
        shutil.copytree("images", portable_dir / "images")
        print("  ✓ images/")

    # 3. 실행 스크립트 생성
    print("[3/6] Creating launcher...")

    launcher_bat = """@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ============================================================
echo EMR Helper - Portable Version
echo ============================================================
echo.

REM 첫 실행 시 패키지 설치
if not exist "installed.txt" (
    echo First run detected. Installing packages...
    echo This will take a few minutes...
    echo.
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo. > installed.txt
    echo Installation complete!
    echo.
)

echo Starting EMR Helper...
python main.py

if %errorlevel% neq 0 (
    echo.
    echo Error occurred! Check error.log
    pause
)
"""

    with open(portable_dir / "EMR_Helper.bat", "w", encoding="utf-8") as f:
        f.write(launcher_bat)

    # 4. 설치 스크립트 생성
    print("[4/6] Creating installer...")

    installer = """@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ============================================================
echo EMR Helper - One-Time Setup
echo ============================================================
echo.
echo This will install required packages.
echo You only need to run this once!
echo.
pause

echo Installing packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo Installation failed!
    pause
    exit /b 1
)

echo. > installed.txt

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Now run "EMR_Helper.bat" to start the program
echo.
pause
"""

    with open(portable_dir / "Setup.bat", "w", encoding="utf-8") as f:
        f.write(installer)

    # 5. README 생성
    print("[5/6] Creating README...")

    readme = """============================================================
EMR Helper - Portable Version
============================================================

SIMPLE 2-STEP USAGE:

Step 1: First Time Setup (Once only)
-------------------------------------
Double-click: Setup.bat

This will:
- Install required packages
- Takes 5-10 minutes
- You only do this ONCE

Step 2: Run Program
-------------------------------------
Double-click: EMR_Helper.bat

This will:
- Start the program immediately
- Use Ctrl+Shift+A in EMR screen
- Auto-generate messages!

============================================================
Requirements
============================================================

- Python 3.8+ must be installed
- Internet connection (for first setup only)
- Windows 7/8/10/11

If Python is not installed:
Download from: https://www.python.org/downloads/
Make sure to check "Add Python to PATH"

============================================================
Usage
============================================================

1. Run Setup.bat (first time only)
2. Run EMR_Helper.bat
3. Open EMR screen
4. Press Ctrl+Shift+A
5. Message generated automatically!

Exit: Press ESC

============================================================
Troubleshooting
============================================================

Q: Setup.bat fails
A: Make sure Python is installed and in PATH
   Run: python --version
   Should show Python 3.8 or higher

Q: EMR_Helper.bat fails
A: Check error.log file for details

Q: OCR doesn't work
A: 1. Check EMR screen is visible
   2. Screen scaling 100-150% recommended
   3. Check config.json settings

============================================================
"""

    with open(portable_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    # 6. 완료
    print("[6/6] Complete!")
    print()
    print("=" * 60)
    print("PORTABLE VERSION CREATED!")
    print("=" * 60)
    print()
    print(f"Folder: {portable_dir}")
    print()
    print("Distribution:")
    print("  1. Copy folder to USB or zip it")
    print("  2. Give to users")
    print("  3. Users run Setup.bat (first time)")
    print("  4. Then run EMR_Helper.bat")
    print()
    print("Requirements:")
    print("  - Users must have Python installed")
    print("  - But NO manual package installation needed")
    print("  - Setup.bat handles everything!")
    print()

if __name__ == "__main__":
    try:
        create_portable()
        input("\nPress Enter to exit...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
