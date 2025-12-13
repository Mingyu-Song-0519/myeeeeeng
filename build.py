"""
EMR Helper - 빌드 스크립트
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def print_step(step, total, message):
    """단계 출력"""
    print(f"\n[{step}/{total}] {message}")
    print("=" * 60)

def run_command(cmd, check=True):
    """명령어 실행"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if check and result.returncode != 0:
        print(f"Error: Command failed with code {result.returncode}")
        return False
    return True

def main():
    print("=" * 60)
    print("EMR Helper - Build Script")
    print("=" * 60)
    print("\nBuilding EXE for end users (no Python required)")
    print("This will take 10-20 minutes...\n")
    input("Press Enter to continue...")

    # 현재 디렉토리 확인
    os.chdir(Path(__file__).parent)

    # Step 1: Environment check
    print_step(1, 6, "Checking environment")
    if not run_command("python --version"):
        print("ERROR: Python is not installed")
        return False

    # Step 2: Clean previous builds
    print_step(2, 6, "Cleaning previous builds")
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Removed {folder}/")

    # Step 3: Install/upgrade packages
    print_step(3, 6, "Installing required packages")
    run_command("python -m pip install pyinstaller --upgrade")
    run_command("python -m pip install -r requirements.txt")

    # Step 4: Build EXE
    print_step(4, 6, "Building EXE (this takes time...)")
    success = run_command("python -m PyInstaller EMR_Helper.spec")

    if not success:
        print("\nBuild FAILED!")
        input("\nPress Enter to exit...")
        return False

    # Step 5: Copy additional files
    print_step(5, 6, "Copying additional files")

    dist_dir = Path("dist/EMR_Helper")
    if not dist_dir.exists():
        print(f"ERROR: {dist_dir} not found")
        return False

    # Copy config.json
    if not (dist_dir / "config.json").exists():
        shutil.copy("config.json", dist_dir)
        print("Copied config.json")

    # Copy images folder
    if not (dist_dir / "images").exists():
        shutil.copytree("images", dist_dir / "images")
        print("Copied images/")

    # Step 6: Create launcher
    print_step(6, 6, "Creating launcher script")

    launcher_content = """@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo Starting EMR Helper...
echo.
start "" "EMR_Helper.exe"
"""

    with open(dist_dir / "Run.bat", "w", encoding="utf-8") as f:
        f.write(launcher_content)

    # Create usage guide
    guide_content = """============================================================
EMR Helper - Usage Guide
============================================================

How to run:
  Double-click: EMR_Helper.exe
  or: Run.bat

How to use:
  1. Open EMR screen
  2. Press: Ctrl+Shift+A
  3. Message will be generated automatically!

Exit: Press ESC

Notes:
  - First run takes 1-2 minutes (loading OCR models)
  - No Python installation required
  - No internet required
  - Windows Defender may show warning:
    Click "More info" -> "Run anyway"

============================================================
"""

    with open(dist_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(guide_content)

    # Success!
    print("\n" + "=" * 60)
    print("BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput folder: {dist_dir}")
    print("\nFiles created:")
    print("  - EMR_Helper.exe")
    print("  - Run.bat")
    print("  - README.txt")
    print("  - config.json")
    print("  - images/")
    print("  - _internal/ (libraries)")

    print("\nHow to distribute:")
    print("  Copy the entire 'dist/EMR_Helper' folder")
    print("  Users can run EMR_Helper.exe directly")
    print("  No Python required!")

    print("\nTest it now:")
    print("  cd dist\\EMR_Helper")
    print("  EMR_Helper.exe")

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
