"""
EMR Helper - 배포용 패키징 스크립트
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def main():
    print("=" * 60)
    print("EMR Helper - Create Distribution Package")
    print("=" * 60)
    print("\nCreates auto-installer for end users")
    print("Users just run the installer and get desktop shortcut!\n")

    # Check if build exists
    dist_dir = Path("dist/EMR_Helper")
    if not dist_dir.exists() or not (dist_dir / "EMR_Helper.exe").exists():
        print("ERROR: Build not found!")
        print("Please run 'python build.py' first")
        input("\nPress Enter to exit...")
        return False

    input("Press Enter to continue...")

    print("\n[1/3] Creating package folder...")
    package_dir = Path("Package")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()

    print("\n[2/3] Copying program files...")
    shutil.copytree(dist_dir, package_dir / "EMR_Helper")
    print("Copied EMR_Helper/")

    print("\n[3/3] Creating installer script...")

    # Create installer
    installer_content = """@echo off
chcp 65001 > nul
echo ============================================================
echo EMR Helper - Auto Installer
echo ============================================================
echo.
echo This will install EMR Helper and create desktop shortcut
echo.
pause

cd /d "%~dp0"

REM Installation directory
set "INSTALL_DIR=%LOCALAPPDATA%\\EMR_Helper"

echo.
echo [1/3] Copying program files...
if exist "%INSTALL_DIR%" rmdir /s /q "%INSTALL_DIR%"
xcopy /e /i /y "EMR_Helper" "%INSTALL_DIR%"

echo [2/3] Creating desktop shortcut...
set "SHORTCUT=%USERPROFILE%\\Desktop\\EMR Helper.lnk"
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%INSTALL_DIR%\\EMR_Helper.exe'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = 'EMR Auto Text Input Helper'; $s.Save()"

echo [3/3] Complete!
echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Installation location: %INSTALL_DIR%
echo Desktop shortcut: EMR Helper
echo.
echo Double-click the desktop shortcut to run!
echo.
echo Usage:
echo   1. Run program (desktop icon)
echo   2. Open EMR screen
echo   3. Press Ctrl+Shift+A
echo   4. Message generated automatically!
echo.
pause
"""

    with open(package_dir / "Install.bat", "w", encoding="utf-8") as f:
        f.write(installer_content)

    # Create README
    readme_content = """============================================================
EMR Helper - Distribution Package
============================================================

For End Users (Easy Installation)
============================================================

1. Run "Install.bat"
   - Automatically installs the program
   - Creates desktop shortcut

2. Double-click "EMR Helper" icon on desktop

3. Use:
   - Open EMR screen
   - Press Ctrl+Shift+A
   - Message generated automatically!

Exit: Press ESC

============================================================
System Requirements
============================================================

- Windows 7/8/10/11
- No Python required
- No internet required
- Approx 500MB disk space

============================================================
Troubleshooting
============================================================

Q: Windows Defender blocks it
A: Click "More info" -> "Run anyway"
   (Safe program but not digitally signed)

Q: Program doesn't start
A: Run as administrator
   Right-click -> "Run as administrator"

Q: OCR doesn't work
A: 1. Check EMR screen is visible and clear
   2. Screen scaling 100-150% recommended
   3. Check config.json settings

Q: Shortcut doesn't work
A: Run directly from installation folder:
   C:\\Users\\[YourName]\\AppData\\Local\\EMR_Helper

============================================================
"""

    with open(package_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Success
    print("\n" + "=" * 60)
    print("PACKAGE CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nPackage folder: {package_dir}")
    print("\nContents:")
    print("  - Install.bat       <- Users run this")
    print("  - README.txt")
    print("  - EMR_Helper/       (program files)")

    print("\n" + "=" * 60)
    print("DISTRIBUTION STEPS:")
    print("=" * 60)
    print("\n1. Compress 'Package' folder to ZIP")
    print("2. Send ZIP file to users")
    print("3. Users extract and run 'Install.bat'")
    print("4. Desktop shortcut created automatically!")

    print("\nUsers only need to:")
    print("  - Extract ZIP")
    print("  - Run Install.bat")
    print("  - Click desktop shortcut")
    print("  - Done!")

    input("\nPress Enter to exit...")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
