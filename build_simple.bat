@echo off
chcp 65001 > nul
echo ================================================
echo EMR Helper - 간단 빌드 (문제 해결용)
echo ================================================
echo.

echo [1/5] 현재 디렉토리 확인...
cd /d "%~dp0"
echo 현재 위치: %CD%
echo.

echo [2/5] 기존 빌드 폴더 정리...
if exist build (
    echo build 폴더 삭제 중...
    rmdir /s /q build
)
if exist dist (
    echo dist 폴더 삭제 중...
    rmdir /s /q dist
)
if exist *.spec (
    echo spec 파일 삭제 중...
    del /q *.spec
)
echo 정리 완료
echo.

echo [3/5] PyInstaller 설치 확인...
python -m pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller를 설치합니다...
    python -m pip install pyinstaller
    if %errorlevel% neq 0 (
        echo [오류] PyInstaller 설치 실패
        pause
        exit /b 1
    )
)
echo PyInstaller 확인 완료
echo.

echo [4/5] 실행 파일 빌드 중...
echo 이 작업은 5~10분 정도 소요될 수 있습니다.
echo.

python -m PyInstaller ^
    --clean ^
    --onedir ^
    --console ^
    --name EMR_Helper ^
    main.py

if %errorlevel% neq 0 (
    echo.
    echo [오류] 빌드에 실패했습니다.
    echo.
    echo 오류 해결 방법:
    echo 1. check_environment.bat을 실행하여 환경을 확인하세요
    echo 2. install.bat을 실행하여 필수 패키지를 설치하세요
    echo 3. Python 버전이 3.8 이상인지 확인하세요
    echo.
    pause
    exit /b 1
)

echo.
echo [5/5] 필수 파일 복사 중...

if not exist "dist\EMR_Helper" (
    echo [오류] dist\EMR_Helper 폴더가 생성되지 않았습니다.
    pause
    exit /b 1
)

echo config.json 복사 중...
copy /y "config.json" "dist\EMR_Helper\" >nul

echo images 폴더 복사 중...
if not exist "dist\EMR_Helper\images" mkdir "dist\EMR_Helper\images"
xcopy /y /e "images\*.*" "dist\EMR_Helper\images\" >nul

echo screen_capture.py 복사 중...
copy /y "screen_capture.py" "dist\EMR_Helper\" >nul

echo ocr_extractor.py 복사 중...
copy /y "ocr_extractor.py" "dist\EMR_Helper\" >nul

echo message_generator.py 복사 중...
copy /y "message_generator.py" "dist\EMR_Helper\" >nul

echo 실행 스크립트 생성 중...
(
echo @echo off
echo chcp 65001 ^> nul
echo echo EMR Helper 실행 중...
echo start "" "EMR_Helper.exe"
) > "dist\EMR_Helper\실행.bat"

echo.
echo ================================================
echo 빌드 완료!
echo ================================================
echo.
echo 생성된 폴더: dist\EMR_Helper
echo.
echo 테스트 방법:
echo 1. dist\EMR_Helper 폴더로 이동
echo 2. EMR_Helper.exe 또는 실행.bat 실행
echo.
echo 배포 방법:
echo dist\EMR_Helper 폴더 전체를 복사하여 사용
echo.
pause
