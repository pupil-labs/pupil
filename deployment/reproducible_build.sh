#! /bin/bash -xe
export PYTHONHASHSEED=42
platform="$(uname -s)"
case "${platform}" in
    Linux*)     machine=linux;;
    Darwin*)    machine=macos;;
    *)          machine="UNKNOWN:${platform}"
esac
release_dir="pupil_core_$(git describe --tags --long)_${machine}_x64"
echo "+ Creating bundle at $release_dir"
pyinstaller pupil_core.spec --noconfirm --log-level DEBUG --distpath $release_dir
