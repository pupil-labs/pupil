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

python ..\pupil_src\shared_modules\pupil_detectors\build.py
python ..\pupil_src\shared_modules\cython_methods\build.py
python ..\pupil_src\shared_modules\calibration_routines\optimization_calibration\build.py

call :Bundle capture %current_tag%
call :Bundle service %current_tag%
call :Bundle player %current_tag%

cd %release_dir%
for /d %%d in (*) do (
	echo Adding %%d
	7z a -t7z ..\%release_dir%.7z %%d
)
cd ..

exit /B 0