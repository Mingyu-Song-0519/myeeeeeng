@echo off
chcp 65001 > nul
echo ================================================
echo EMR Helper - 포터블 버전 빌드 (권장)
echo ================================================
echo.
echo 이 빌드 방식은 폴더 형태로 배포되며 더 안정적입니다.
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
echo [3/4] 포터블 버전 빌드 중... (시간이 걸릴 수 있습니다)
pyinstaller --clean ^
    --onedir ^
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
    main.py

if %errorlevel% neq 0 (
    echo.
    echo 오류: 빌드에 실패했습니다.
    pause
    exit /b 1
)

echo.
echo [4/4] 실행 스크립트 생성 중...
(
echo @echo off
echo chcp 65001 ^> nul
echo start "" "EMR_Helper.exe"
) > "dist\EMR_Helper\실행.bat"

echo.
echo ================================================
echo 빌드 완료!
echo ================================================
echo.
echo 실행 파일 위치: dist\EMR_Helper\
echo.
echo 배포 방법:
echo 1. dist\EMR_Helper 폴더 전체를 USB나 다른 PC로 복사
echo 2. 폴더 안의 EMR_Helper.exe 또는 실행.bat을 실행
echo.
echo 참고:
echo - Python 설치 없이 바로 실행 가능
echo - 첫 실행 시 EasyOCR 모델 다운로드로 시간 소요
echo - config.json과 images 폴더가 같이 포함되어 있음
echo.
pause
