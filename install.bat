@echo off
chcp 65001 > nul
echo ================================================
echo EMR 자동 텍스트 입력 도우미 - 설치 스크립트
echo ================================================
echo.

echo Python 버전 확인 중...
python --version
if %errorlevel% neq 0 (
    echo 오류: Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 Python을 다운로드하세요.
    pause
    exit /b 1
)

echo.
echo 필요한 패키지 설치 중...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo 오류: 패키지 설치에 실패했습니다.
    pause
    exit /b 1
)

echo.
echo ================================================
echo 설치가 완료되었습니다!
echo ================================================
echo.
echo 사용 방법:
echo 1. run.bat 파일을 실행하세요
echo 2. 또는 명령 프롬프트에서 'python main.py' 실행
echo.
pause
