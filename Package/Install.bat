@echo off
chcp 65001 > nul
echo ============================================================
echo EMR Helper - Auto Installer
echo ============================================================
echo.
echo This will install EMR Helper and create desktop shortcut
echo.
pause

cd /d "%~dp0"

REM Installation directory
set "INSTALL_DIR=%LOCALAPPDATA%\EMR_Helper"

echo.
echo [1/3] Copying program files...
if exist "%INSTALL_DIR%" rmdir /s /q "%INSTALL_DIR%"
xcopy /e /i /y "EMR_Helper" "%INSTALL_DIR%"

echo [2/3] Creating desktop shortcut...
set "SHORTCUT=%USERPROFILE%\Desktop\EMR Helper.lnk"
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%INSTALL_DIR%\EMR_Helper.exe'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = 'EMR Auto Text Input Helper'; $s.Save()"

echo [3/3] Complete!
echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Installation location: %INSTALL_DIR%
echo Desktop shortcut: EMR Helper
echo.
echo Double-click the desktop shortcut to run!
echo.
echo Usage:
echo   1. Run program (desktop icon)
echo   2. Open EMR screen
echo   3. Press Ctrl+Shift+A
echo   4. Message generated automatically!
echo.
pause
