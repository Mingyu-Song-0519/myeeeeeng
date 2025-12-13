@echo off
chcp 65001 > nul
echo ================================================
echo EMR Helper - 단일 EXE 파일 빌드
echo ================================================
echo.
echo 사용자가 EXE 파일 하나만 더블클릭하면 실행됩니다!
echo.
echo 주의: 첫 실행 시 압축 해제로 30초~1분 소요됩니다.
echo       파일 크기가 매우 클 수 있습니다 (500MB~2GB).
echo.
pause

cd /d "%~dp0"

echo.
echo [1/5] 환경 확인...
python --version
if %errorlevel% neq 0 (
    echo [오류] Python이 설치되어 있지 않습니다.
    pause
    exit /b 1
)
echo.

echo [2/5] 기존 빌드 정리...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo.

echo [3/5] PyInstaller 및 패키지 확인...
python -m pip install pyinstaller --upgrade
python -m pip install -r requirements.txt
echo.

echo [4/5] 단일 EXE 빌드 중...
echo 이 작업은 20-30분 정도 걸릴 수 있습니다.
echo 매우 큰 파일을 만들고 있습니다...
echo.
python -m PyInstaller EMR_Helper_onefile.spec

if %errorlevel% neq 0 (
    echo.
    echo [오류] 빌드 실패
    pause
    exit /b 1
)

echo.
echo [5/5] 사용 안내 파일 생성...
(
echo ============================================================
echo EMR Helper - 사용 방법
echo ============================================================
echo.
echo ** 이 파일 하나만 있으면 됩니다! **
echo.
echo 실행 방법:
echo   EMR_Helper.exe 더블클릭
echo.
echo 첫 실행 시:
echo   - 압축 해제 및 로딩에 1-2분 소요
echo   - 이후 실행부터는 빠릅니다
echo.
echo 사용 방법:
echo   1. EMR 화면 열기
echo   2. Ctrl+Shift+A 단축키
echo   3. 자동으로 메시지 생성!
echo.
echo 종료: ESC 키
echo.
echo 주의사항:
echo   - Python 설치 불필요
echo   - 인터넷 연결 불필요
echo   - Windows Defender 경고 시: 
echo     "추가 정보" -^> "실행" 클릭
echo.
echo ============================================================
) > "dist\EMR_Helper_사용방법.txt"

echo.
echo ================================================
echo 빌드 완료!
echo ================================================
echo.
echo 생성된 파일: dist\EMR_Helper.exe
echo 파일 크기: 
dir "dist\EMR_Helper.exe" | find "EMR_Helper.exe"
echo.
echo 배포 방법:
echo   dist\EMR_Helper.exe 파일 하나만 복사하면 됩니다!
echo   (EMR_Helper_사용방법.txt도 함께 배포 권장)
echo.
echo 테스트:
echo   cd dist
echo   EMR_Helper.exe
echo.
pause
