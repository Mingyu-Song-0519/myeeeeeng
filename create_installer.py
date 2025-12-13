"""
올인원 설치 프로그램 생성
사용자가 설치.bat 하나만 실행하면 모든 것이 자동으로 설정됨
"""

import shutil
from pathlib import Path

def create_installer():
    print("=" * 60)
    print("Creating All-in-One Installer")
    print("=" * 60)
    print()

    # 배포 폴더 생성
    dist_folder = Path("EMR_Helper_Distribution")
    if dist_folder.exists():
        shutil.rmtree(dist_folder)
    dist_folder.mkdir()

    print("[1/4] Copying program files...")

    # 소스 파일 복사
    files = [
        "main.py",
        "screen_capture.py",
        "ocr_extractor.py",
        "message_generator.py",
        "config.json",
        "requirements.txt"
    ]

    for file in files:
        if Path(file).exists():
            shutil.copy(file, dist_folder)
            print(f"  ✓ {file}")

    # images 폴더 복사
    if Path("images").exists():
        shutil.copytree("images", dist_folder / "images")
        print(f"  ✓ images/")

    print("\n[2/4] Creating installer script...")

    # 메인 설치 스크립트
    installer_script = """@echo off
chcp 65001 > nul
cls
echo.
echo ============================================================
echo             EMR Helper - One-Click Installer
echo ============================================================
echo.
echo This installer will:
echo   1. Check if Python is installed
echo   2. Install required packages automatically
echo   3. Create desktop shortcut
echo   4. Ready to use!
echo.
echo This is a ONE-TIME setup. After this, just double-click
echo the desktop shortcut to run the program.
echo.
echo ============================================================
echo.
pause

REM Change to script directory
cd /d "%~dp0"

echo.
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python is not installed!
    echo.
    echo Please install Python first:
    echo   1. Go to: https://www.python.org/downloads/
    echo   2. Download Python 3.8 or newer
    echo   3. IMPORTANT: Check "Add Python to PATH" during installation
    echo   4. Run this installer again
    echo.
    pause
    exit /b 1
)
python --version
echo Python is installed!

echo.
echo [2/4] Installing required packages...
echo This will take 5-10 minutes. Please wait...
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Package installation failed!
    echo Please check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo [3/4] Creating desktop shortcut...

REM Get full path
set "INSTALL_DIR=%~dp0"
set "SHORTCUT=%USERPROFILE%\\Desktop\\EMR Helper.lnk"

REM Create shortcut using PowerShell
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%INSTALL_DIR%EMR_Helper.bat'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = 'EMR Auto Text Input Helper'; $s.Save()"

echo Desktop shortcut created!

echo.
echo [4/4] Creating completion marker...
echo Installation completed successfully! > installed.txt

echo.
echo ============================================================
echo                Installation Complete!
echo ============================================================
echo.
echo A shortcut "EMR Helper" has been created on your desktop.
echo.
echo HOW TO USE:
echo   1. Double-click "EMR Helper" on desktop
echo   2. Wait for program to load (first time takes 1-2 minutes)
echo   3. Open EMR screen
echo   4. Press Ctrl+Shift+A
echo   5. Message will be generated automatically!
echo.
echo To close program: Press ESC
echo.
echo ============================================================
echo.
pause
"""

    with open(dist_folder / "Install.bat", "w", encoding="utf-8") as f:
        f.write(installer_script)

    print("\n[3/4] Creating run script...")

    # 실행 스크립트
    run_script = """@echo off
chcp 65001 > nul
cd /d "%~dp0"

REM Check if installed
if not exist "installed.txt" (
    echo.
    echo ============================================================
    echo                  First Time Setup Required
    echo ============================================================
    echo.
    echo Please run "Install.bat" first!
    echo.
    echo This is a one-time setup that installs required packages.
    echo After installation, you can use this shortcut directly.
    echo.
    pause
    exit /b 1
)

REM Run program
python main.py

REM If error occurred
if %errorlevel% neq 0 (
    echo.
    echo An error occurred. Please check error.log
    pause
)
"""

    with open(dist_folder / "EMR_Helper.bat", "w", encoding="utf-8") as f:
        f.write(run_script)

    print("\n[4/4] Creating user guide...")

    # 사용자 가이드
    guide = """============================================================
                    EMR Helper - User Guide
============================================================

SUPER EASY 2-STEP PROCESS:

Step 1: INSTALL (One Time Only)
--------------------------------
Double-click: Install.bat

This will:
✓ Check Python installation
✓ Install required packages (5-10 minutes)
✓ Create desktop shortcut
✓ Done!

** You only do this ONCE! **


Step 2: USE (Every Time)
--------------------------------
Double-click: "EMR Helper" on desktop

Then:
1. Wait for program to load (first time: 1-2 min, later: instant)
2. Open EMR screen
3. Press Ctrl+Shift+A
4. Message generated automatically!
5. Press ESC to exit

That's it!

============================================================
                    System Requirements
============================================================

✓ Windows 7/8/10/11
✓ Python 3.8 or newer (installer will check)
✓ Internet connection (for initial setup only)
✓ ~500MB free space

If Python is not installed:
  1. Go to: https://www.python.org/downloads/
  2. Download latest version
  3. IMPORTANT: Check "Add Python to PATH"
  4. Install
  5. Run Install.bat

============================================================
                    Troubleshooting
============================================================

Q: Install.bat says "Python is not installed"
A: Install Python from python.org
   Make sure to check "Add Python to PATH"!

Q: Installation fails
A: Check internet connection
   Run as Administrator (right-click -> Run as administrator)

Q: Desktop shortcut doesn't work
A: Run EMR_Helper.bat directly from the folder

Q: Program closes immediately
A: Make sure you ran Install.bat first
   Check error.log for details

Q: OCR doesn't recognize text
A: - EMR screen must be visible and clear
   - Screen scaling: 100-150% recommended
   - Adjust settings in config.json

============================================================
                    Configuration
============================================================

Edit config.json to customize:
  - Hotkey (default: Ctrl+Shift+A)
  - Message template
  - OCR settings
  - Debug mode

============================================================
                    Support
============================================================

For issues or questions, check error.log file in the
program folder for detailed error messages.

============================================================
"""

    with open(dist_folder / "README.txt", "w", encoding="utf-8") as f:
        f.write(guide)

    print("\n" + "=" * 60)
    print("ALL-IN-ONE INSTALLER CREATED!")
    print("=" * 60)
    print()
    print(f"Distribution folder: {dist_folder}")
    print()
    print("Files created:")
    print("  ✓ Install.bat       <- Main installer")
    print("  ✓ EMR_Helper.bat    <- Program launcher")
    print("  ✓ README.txt        <- User guide")
    print("  ✓ All program files")
    print()
    print("=" * 60)
    print("HOW TO DISTRIBUTE:")
    print("=" * 60)
    print()
    print("1. ZIP the entire folder:")
    print(f"   {dist_folder}")
    print()
    print("2. Send to users")
    print()
    print("3. Users:")
    print("   - Extract ZIP")
    print("   - Run Install.bat (one time)")
    print("   - Use desktop shortcut forever!")
    print()
    print("=" * 60)
    print("USER EXPERIENCE:")
    print("=" * 60)
    print()
    print("User receives ZIP file")
    print("  └─> Extracts to folder")
    print("      └─> Double-clicks Install.bat")
    print("          └─> Waits 5-10 minutes")
    print("              └─> Desktop shortcut created!")
    print("                  └─> Double-clicks shortcut")
    print("                      └─> Program runs!")
    print()
    print("SUPER SIMPLE! Just 3 double-clicks total:")
    print("  1. Extract ZIP")
    print("  2. Run Install.bat")
    print("  3. Use desktop shortcut")
    print()

if __name__ == "__main__":
    try:
        create_installer()
        input("\nPress Enter to exit...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
