"""
가장 간단한 EXE 빌드 - 테스트용
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def main():
    print("=" * 60)
    print("Building Simple Test EXE")
    print("=" * 60)

    os.chdir(Path(__file__).parent)

    # Clean
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # Build simple test
    print("\nBuilding test_simple.py...")
    cmd = "python -m PyInstaller --onefile --console --name EMR_Helper_Test test_simple.py"

    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("Build successful!")
        print("=" * 60)
        print("\nTest it:")
        print("  cd dist")
        print("  EMR_Helper_Test.exe")
        print("\nIf this works, we'll build the full version.")
    else:
        print("\nBuild failed!")

    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
