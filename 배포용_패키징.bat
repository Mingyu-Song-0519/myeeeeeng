@echo off
chcp 65001 > nul
echo ================================================
echo EMR Helper - 배포용 자동 설치 패키지 생성
echo ================================================
echo.
echo 사용자가 한 번만 실행하면 자동으로 설치되고
echo 바탕화면에 바로가기가 생성됩니다!
echo.
pause

cd /d "%~dp0"

echo.
echo [1/4] 폴더 빌드 확인...
if not exist "dist\EMR_Helper\EMR_Helper.exe" (
    echo 먼저 build_final.bat을 실행하여 빌드하세요.
    pause
    exit /b 1
)

echo.
echo [2/4] 배포 패키지 폴더 생성...
if exist "배포용" rmdir /s /q "배포용"
mkdir "배포용"

echo.
echo [3/4] 자동 설치 스크립트 생성...

REM 자동 설치 스크립트
(
echo @echo off
echo chcp 65001 ^> nul
echo echo ============================================================
echo echo EMR Helper 자동 설치
echo echo ============================================================
echo echo.
echo echo 프로그램을 설치하고 바탕화면에 바로가기를 만듭니다.
echo echo.
echo pause
echo.
echo cd /d "%%~dp0"
echo.
echo REM 설치 폴더 결정
echo set "INSTALL_DIR=%%LOCALAPPDATA%%\EMR_Helper"
echo.
echo echo.
echo echo [1/3] 프로그램 파일 복사 중...
echo if exist "%%INSTALL_DIR%%" rmdir /s /q "%%INSTALL_DIR%%"
echo xcopy /e /i /y "EMR_Helper" "%%INSTALL_DIR%%"
echo.
echo echo [2/3] 바탕화면 바로가기 생성 중...
echo set "SHORTCUT=%%USERPROFILE%%\Desktop\EMR Helper.lnk"
echo powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%%SHORTCUT%%'^); $s.TargetPath = '%%INSTALL_DIR%%\EMR_Helper.exe'; $s.WorkingDirectory = '%%INSTALL_DIR%%'; $s.Description = 'EMR 자동 텍스트 입력 도우미'; $s.Save(^)"
echo.
echo echo [3/3] 완료!
echo echo.
echo echo ============================================================
echo echo 설치 완료!
echo echo ============================================================
echo echo.
echo echo 설치 위치: %%INSTALL_DIR%%
echo echo 바탕화면에 "EMR Helper" 바로가기가 생성되었습니다.
echo echo.
echo echo 바로가기를 더블클릭하여 실행하세요!
echo echo.
echo echo 사용 방법:
echo echo   1. 프로그램 실행
echo echo   2. EMR 화면에서 Ctrl+Shift+A
echo echo   3. 자동으로 메시지 생성!
echo echo.
echo pause
) > "배포용\EMR_Helper_설치.bat"

echo.
echo [4/4] 프로그램 파일 복사...
xcopy /e /i /y "dist\EMR_Helper" "배포용\EMR_Helper"

REM 사용 안내
(
echo ============================================================
echo EMR Helper - 배포 패키지
echo ============================================================
echo.
echo ** 초보자도 쉽게 사용할 수 있습니다! **
echo.
echo ============================================================
echo 설치 방법
echo ============================================================
echo.
echo 1. 이 폴더를 USB나 다른 PC로 복사
echo.
echo 2. "EMR_Helper_설치.bat" 더블클릭
echo    - 자동으로 설치됩니다
echo    - 바탕화면에 바로가기가 생성됩니다
echo.
echo 3. 바탕화면의 "EMR Helper" 아이콘 더블클릭
echo.
echo ============================================================
echo 사용 방법
echo ============================================================
echo.
echo 1. 프로그램 실행 (바탕화면 아이콘)
echo 2. EMR 화면 열기
echo 3. Ctrl+Shift+A 키 누르기
echo 4. 자동으로 환자 정보 추출 및 메시지 생성!
echo.
echo 종료: ESC 키
echo.
echo ============================================================
echo 문제 해결
echo ============================================================
echo.
echo Q: Windows Defender가 차단
echo A: "추가 정보" -^> "실행" 클릭
echo.
echo Q: 프로그램이 실행되지 않음
echo A: 설치 폴더에서 EMR_Helper.exe를 직접 실행
echo    (C:\Users\사용자명\AppData\Local\EMR_Helper)
echo.
echo Q: 단축키가 작동하지 않음  
echo A: 관리자 권한으로 실행
echo.
echo ============================================================
) > "배포용\README.txt"

echo.
echo ================================================
echo 배포 패키지 생성 완료!
echo ================================================
echo.
echo 생성된 폴더: 배포용\
echo.
echo 폴더 구조:
echo   배포용\
echo   ├── EMR_Helper_설치.bat  ^<- 사용자가 실행할 파일
echo   ├── README.txt
echo   └── EMR_Helper\          (프로그램 파일들^)
echo.
echo 배포 방법:
echo   1. "배포용" 폴더를 ZIP으로 압축
echo   2. 사용자에게 전달
echo   3. 사용자가 압축 해제 후 "EMR_Helper_설치.bat" 실행
echo   4. 바탕화면에 바로가기 생성됨!
echo.
pause
