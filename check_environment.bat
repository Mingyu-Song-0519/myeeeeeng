@echo off
chcp 65001 > nul
echo ================================================
echo 환경 확인 스크립트
echo ================================================
echo.

echo [1] Python 설치 확인...
python --version
if %errorlevel% neq 0 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 설치하세요.
    goto :error
)
echo [OK] Python 설치됨
echo.

echo [2] pip 확인...
pip --version
if %errorlevel% neq 0 (
    echo [오류] pip를 찾을 수 없습니다.
    goto :error
)
echo [OK] pip 정상
echo.

echo [3] 필수 패키지 확인...
echo - pyinstaller 확인...
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [경고] pyinstaller가 설치되어 있지 않습니다.
    echo.
    echo 지금 설치하시겠습니까? (Y/N)
    set /p answer=선택:
    if /i "%answer%"=="Y" (
        pip install pyinstaller
    ) else (
        echo pyinstaller 설치를 건너뜁니다.
    )
) else (
    echo [OK] pyinstaller 설치됨
)

echo.
echo - easyocr 확인...
pip show easyocr >nul 2>&1
if %errorlevel% neq 0 (
    echo [경고] easyocr가 설치되어 있지 않습니다.
    echo install.bat을 먼저 실행하세요.
) else (
    echo [OK] easyocr 설치됨
)

echo.
echo - opencv-python 확인...
pip show opencv-python >nul 2>&1
if %errorlevel% neq 0 (
    echo [경고] opencv-python이 설치되어 있지 않습니다.
    echo install.bat을 먼저 실행하세요.
) else (
    echo [OK] opencv-python 설치됨
)

echo.
echo [4] 필수 파일 확인...
if not exist "config.json" (
    echo [오류] config.json 파일이 없습니다.
    goto :error
)
echo [OK] config.json 존재

if not exist "images" (
    echo [오류] images 폴더가 없습니다.
    goto :error
)
echo [OK] images 폴더 존재

if not exist "main.py" (
    echo [오류] main.py 파일이 없습니다.
    goto :error
)
echo [OK] main.py 존재

echo.
echo ================================================
echo 환경 확인 완료!
echo ================================================
echo.
echo 다음 단계:
echo 1. 패키지가 설치되지 않았다면: install.bat 실행
echo 2. 빌드 준비 완료: build_portable.bat 실행
echo.
pause
exit /b 0

:error
echo.
echo ================================================
echo 오류 발생!
echo ================================================
echo 위의 오류를 해결한 후 다시 시도하세요.
echo.
pause
exit /b 1
