#!/bin/bash

# get most major.minor tag, without trailing count
current_tag=$(git describe --tags --long)

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    release_dir=$(echo "pupil_${current_tag}_linux_x64")
    ext=*.deb
elif [[ "$OSTYPE" == "darwin"* ]]; then
    release_dir=$(echo "pupil_${current_tag}_macos_x64")
    ext=*.NOTANEXTENSION
fi
echo "release_dir:  ${release_dir}"
mkdir ${release_dir}

# bundle Pupil Capture
printf "\n##########\nBundling Pupil Capture\n##########\n\n"
cd deploy_capture
./bundle.sh
mv *.$ext ../$release_dir

# bundle Pupil Service
printf "\n##########\nBundling Pupil Service\n##########\n\n"
cd ../deploy_service
./bundle.sh
mv *.$ext ../$release_dir

# bundle Pupil Player
printf "\n##########\nBundling Pupil Player\n##########\n\n"
cd ../deploy_player
./bundle.sh
mv *.$ext ../$release_dir

cd ..

if [[ "$OSTYPE" == "darwin"* ]]; then
    printf "\n########## Moving applications into $release_dir"
    mv **/*.app $release_dir

    printf "\n########## Creating dmg file"
    ln -s /Applications/ $release_dir/Applications
    dir_size = $(du -m -d 0 $release_dir | cut -f1)
    hdiutil create \
        -volname 'Install Pupil' \
        -srcfolder $release_dir \
        -format UDZO \
        -megabytes $dir_size \
        $release_dir.dmg
    
    printf "\n########## Signing dmg file"
    sign = "Developer ID Application: Pupil Labs UG (haftungsbeschrankt) (R55K9ESN6B)"
    codesign \
        --force \
        --verify \
        --verbose \
        --s $sign \
        --deep \
        $release_dir.dmg
else
    printf "\n##########\nzipping release\n##########\n\n"
    zip -r $release_dir.zip $release_dir
fi
