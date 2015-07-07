python ../pupil_src/shared_modules/pyx_compiler.py
pyinstaller --noconfirm --clean bundle.spec
python finalize_bundle.py
