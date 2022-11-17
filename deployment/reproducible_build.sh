#! /bin/bash -xe
export PYTHONHASHSEED=42
platform="$(uname -s)"
case "${platform}" in
    Linux*)     machine=linux;;
    Darwin*)    machine=macos;;
    *)          machine="UNKNOWN:${platform}"
esac
release_dir="pupil_$(git describe --tags --long)_${machine}_x64"
echo "Creating bundle at $release_dir"
pyinstaller pupil_core.spec --noconfirm --log-level DEBUG --distpath $release_dir

ln -s /Applications/ $release_dir/Applications
size_in_k=$(du -sk $release_dir | cut -f1)
let "size_in_b = $size_in_k * 1024"
hdiutil create \
    -volname "Install Pupil $current_tag" \
    -srcfolder $release_dir \
    -format UDZO \
    -size "$size_in_b"b \
    $release_dir.dmg
