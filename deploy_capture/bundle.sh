rm -r dist
pyinstaller --noconfirm --clean bundle.spec
python finalize_bundle.py
