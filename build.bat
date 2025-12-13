@echo off
chcp 65001 > nul
echo ================================================
echo EMR Helper - 실행 파일 빌드 스크립트
echo ================================================
echo.

echo [1/4] 기존 빌드 폴더 정리 중...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

echo.
echo [2/4] PyInstaller 확인 중...
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller가 설치되어 있지 않습니다. 설치 중...
    pip install pyinstaller
)

echo.
echo [3/4] EXE 파일 빌드 중... (시간이 걸릴 수 있습니다)
pyinstaller --clean ^
    --onefile ^
    --console ^
    --name "EMR_Helper" ^
    --add-data "config.json;." ^
    --add-data "images;images" ^
    --hidden-import "easyocr" ^
    --hidden-import "cv2" ^
    --hidden-import "mss" ^
    --hidden-import "pyautogui" ^
    --hidden-import "keyboard" ^
    --hidden-import "pyperclip" ^
    --hidden-import "win32gui" ^
    --collect-all easyocr ^
    --icon NONE ^
    main.py

if %errorlevel% neq 0 (
    echo.
    echo 오류: 빌드에 실패했습니다.
    pause
    exit /b 1
)

echo.
echo [4/4] 필수 파일 복사 중...
if not exist "dist\images" mkdir "dist\images"
copy "config.json" "dist\" >nul
xcopy "images\*.*" "dist\images\" /Y >nul

echo.
echo ================================================
echo 빌드 완료!
echo ================================================
echo.
echo 실행 파일 위치: dist\EMR_Helper.exe
echo.
echo 배포 방법:
echo 1. dist 폴더 전체를 복사
echo 2. EMR_Helper.exe를 실행
echo.
pause
