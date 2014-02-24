version=$(python git_version.py)
os=$(uname -s)
platform=$(uname -p)
echo $version,$os,$platform
out_name="pupil_capture_${version}_${os}_${platform}.zip"
rm -r Pupil_Capture
rm $out_name
pyinstaller --noconfirm --clean bundle.spec
python finalize_bundle.py
mv dist Pupil_Capture
zip -r $out_name  "Pupil_Capture"
