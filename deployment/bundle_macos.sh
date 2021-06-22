#!/usr/local/bin/zsh

pl_codesign () {
    sign="Developer ID Application: Pupil Labs UG (haftungsbeschrankt) (R55K9ESN6B)"
    codesign \
        --all-architectures \
        --force \
        --strict=all \
        --options runtime \
        --entitlements entitlements.plist \
        --continue \
        --verify \
        --verbose=4 \
        -s "$sign" \
        --deep "$@"
}

# get most major.minor tag, without trailing count
current_tag=$(git describe --tags --long)
release_dir=$(echo "pupil_${current_tag}_macos_x64")
echo "release_dir:  ${release_dir}"
mkdir ${release_dir}

ext=app

# bundle Pupil Capture
printf "\n##########\nBundling Pupil Capture\n##########\n\n"
cd deploy_capture
./bundle.sh
mv dist/*.$ext ../$release_dir
cd ..

# bundle Pupil Service
printf "\n##########\nBundling Pupil Service\n##########\n\n"
cd deploy_service
./bundle.sh
mv dist/*.$ext ../$release_dir
cd ..

# bundle Pupil Player
printf "\n##########\nBundling Pupil Player\n##########\n\n"
cd deploy_player
./bundle.sh
mv dist/*.$ext ../$release_dir
cd ..

printf "\n##########\nSigning applications\n##########\n"
pl_codesign $release_dir/*.$ext/Contents/Resources/**/.dylibs/*.dylib
pl_codesign $release_dir/*.$ext

printf "\n##########\nCreating dmg file\n##########\n"
ln -s /Applications/ $release_dir/Applications
size_in_k=$(du -sk $release_dir | cut -f1)
let "size_in_b = $size_in_k * 1024"
hdiutil create \
    -volname "Install Pupil $current_tag" \
    -srcfolder $release_dir \
    -format UDZO \
    -size "$size_in_b"b \
    $release_dir.dmg

printf "\n##########\nSigning dmg file\n##########\n"
pl_codesign "$release_dir.dmg"
