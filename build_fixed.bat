@echo off
chcp 65001 > nul
echo ================================================
echo EMR Helper - 수정된 빌드 스크립트
echo ================================================
echo.

cd /d "%~dp0"

echo [1/4] 기존 빌드 정리...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del /q *.spec
echo.

echo [2/4] PyInstaller 확인...
python -m pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller 설치 중...
    python -m pip install pyinstaller
)
echo.

echo [3/4] 빌드 중... (5-10분 소요)
echo.

python -m PyInstaller ^
    --clean ^
    --onedir ^
    --console ^
    --name EMR_Helper ^
    --hidden-import easyocr ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import PIL ^
    --hidden-import torch ^
    --hidden-import torchvision ^
    --collect-all easyocr ^
    --copy-metadata easyocr ^
    --copy-metadata torch ^
    main.py

if %errorlevel% neq 0 (
    echo.
    echo [오류] 빌드 실패
    echo.
    echo 해결 방법:
    echo 1. 명령 프롬프트에서: python main.py
    echo    (빌드 없이 바로 실행)
    echo.
    pause
    exit /b 1
)

echo.
echo [4/4] 파일 복사...

copy /y config.json dist\EMR_Helper\ >nul
xcopy /y /e images dist\EMR_Helper\images\ >nul
copy /y screen_capture.py dist\EMR_Helper\ >nul
copy /y ocr_extractor.py dist\EMR_Helper\ >nul
copy /y message_generator.py dist\EMR_Helper\ >nul

echo.
echo ================================================
echo 빌드 완료!
echo ================================================
echo.
echo 테스트 방법:
echo 1. 명령 프롬프트에서:
echo    cd /d D:\asanhelper\dist\EMR_Helper
echo    EMR_Helper.exe
echo.
pause
