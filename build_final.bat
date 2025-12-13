@echo off
chcp 65001 > nul
echo ================================================
echo EMR Helper - 최종 EXE 빌드
echo ================================================
echo.
echo 초보자도 Python 없이 사용할 수 있는 EXE를 만듭니다.
echo 이 작업은 10-20분 정도 걸릴 수 있습니다.
echo.
pause

cd /d "%~dp0"

echo.
echo [1/6] 환경 확인 중...
echo.

python --version
if %errorlevel% neq 0 (
    echo [오류] Python이 설치되어 있지 않습니다.
    pause
    exit /b 1
)

echo.
echo [2/6] 기존 빌드 정리 중...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo 정리 완료
echo.

echo [3/6] 필수 패키지 설치 확인...
echo PyInstaller 확인...
python -m pip install pyinstaller --upgrade

echo.
echo 모든 의존성 패키지 확인...
python -m pip install -r requirements.txt
echo.

echo [4/6] EXE 빌드 중... (시간이 오래 걸립니다)
echo.
python -m PyInstaller EMR_Helper.spec

if %errorlevel% neq 0 (
    echo.
    echo [오류] 빌드 실패
    pause
    exit /b 1
)

echo.
echo [5/6] 추가 파일 복사 확인...
if not exist "dist\EMR_Helper\config.json" (
    echo config.json 복사...
    copy /y config.json dist\EMR_Helper\
)

if not exist "dist\EMR_Helper\images" (
    echo images 폴더 복사...
    xcopy /e /i images dist\EMR_Helper\images
)

echo.
echo [6/6] 실행 스크립트 생성...
(
echo @echo off
echo chcp 65001 ^> nul
echo cd /d "%%~dp0"
echo echo EMR Helper 실행 중...
echo echo.
echo start "" "EMR_Helper.exe"
) > "dist\EMR_Helper\실행.bat"

(
echo ============================================================
echo EMR Helper 사용 가이드
echo ============================================================
echo.
echo 실행 방법:
echo 1. EMR_Helper.exe 더블클릭
echo 2. 또는 실행.bat 더블클릭
echo.
echo 사용 방법:
echo 1. EMR 화면을 열어놓은 상태에서
echo 2. Ctrl+Shift+A 키를 누르면
echo 3. 자동으로 환자 정보를 추출하고 메시지를 생성합니다
echo.
echo 종료: ESC 키
echo.
echo 주의사항:
echo - 첫 실행 시 1-2분 정도 로딩 시간이 필요합니다
echo - Windows Defender 경고가 뜰 수 있습니다
echo   ^(추가 정보 -^> 실행 클릭^)
echo.
echo ============================================================
) > "dist\EMR_Helper\사용방법.txt"

echo.
echo ================================================
echo 빌드 완료!
echo ================================================
echo.
echo 생성된 폴더: dist\EMR_Helper
echo.
echo 이제 테스트해보세요:
echo 1. 명령 프롬프트에서:
echo    cd dist\EMR_Helper
echo    EMR_Helper.exe
echo.
echo 2. 또는 탐색기에서:
echo    dist\EMR_Helper 폴더 열기
echo    EMR_Helper.exe 더블클릭
echo.
echo 배포 방법:
echo dist\EMR_Helper 폴더 전체를 USB나 다른 PC로 복사
echo Python 설치 없이 바로 실행 가능!
echo.
pause
