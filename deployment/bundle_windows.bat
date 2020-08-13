@echo off
title Pupil Bundle Procedure
setlocal enableextensions 

goto :MAIN

:Bundle
setlocal
set app=%~1
set version=%~2

echo Bundling pupil_%app% %version%

cd deploy_%app%
pyinstaller --noconfirm --clean --log-level WARN bundle.spec
python finalize_bundle.py
set app_folder=pupil_%app%_windows_x64_%current_tag%
echo Finishing %app_folder%
move "dist\Pupil %app%" ..\%release_dir%\%app_folder%
cd ..

endlocal
exit /B 0

:MAIN
for /F "tokens=* USEBACKQ" %%F IN (`git describe --tags --long`) DO (
set current_tag=%%F
)
set release_dir=pupil_%current_tag%_windows_x64
echo release_dir:  %release_dir%
if not exist %release_dir% (
	mkdir %release_dir%
)

set PATH=%PATH%;%~dp0..\pupil_external
set PATH=%PATH%;C:\Python36\Lib\site-packages\scipy\.libs
set PATH=%PATH%;C:\Python36\Lib\site-packages\zmq

call :Bundle capture %current_tag%
call :Bundle service %current_tag%
call :Bundle player %current_tag%


python generate_msi_installer.py
rar a %release_dir%.msi.rar %release_dir%.msi

exit /B 0