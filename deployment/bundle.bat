@echo off
title Pupil Bundle Procedure
setlocal enableextensions 

for /F "tokens=* USEBACKQ" %%F IN (`git describe --tags --long`) DO (
set current_tag=%%F
)
set release_dir=pupil_%current_tag%_window_x64
echo release_dir:  %release_dir%
if not exist %release_dir% (
	mkdir %release_dir%
)

python ..\pupil_src\shared_modules\pupil_detectors\build.py
python ..\pupil_src\shared_modules\cython_methods\build.py
python ..\pupil_src\shared_modules\calibration_routines\optimization_calibration\build.py

cd deploy_capture
pyinstaller --noconfirm --clean --log-level WARN bundle.spec
python finalize_bundle.py
set capture_bundle=pupil_capture_windows_x64_%current_tag%
echo Finishing %capture_bundle%
move "dist\Pupil Capture" ..\%release_dir%\%capture_bundle%
cd ..

cd deploy_service
pyinstaller --noconfirm --clean --log-level WARN bundle.spec
python finalize_bundle.py
set capture_bundle=pupil_capture_windows_x64_%current_tag%
echo Finishing %capture_bundle%
move "dist\Pupil Service" ..\%release_dir%\%capture_bundle%
cd ..

cd deploy_player
pyinstaller --noconfirm --clean --log-level WARN bundle.spec
python finalize_bundle.py
set capture_bundle=pupil_capture_windows_x64_%current_tag%
echo Finishing %capture_bundle%
move "dist\Pupil Player" ..\%release_dir%\%capture_bundle%
cd ..

"C:\Program Files\7-Zip\7z.exe" a -t7z %release_dir%.7z %release_dir%