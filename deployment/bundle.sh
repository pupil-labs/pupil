#!/bin/bash

# get most major.minor tag, without trailing count
current_tag=$(git describe --tags --long)

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    release_dir=$(echo "pupil_${current_tag}_linux_x64")
    ext=*.deb
elif [[ "$OSTYPE" == "darwin"* ]]; then
    release_dir=$(echo "pupil_${current_tag}_macos_x64")
    ext=*.dmg
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
printf "\n##########\nzipping release\n##########\n\n"
zip -r $release_dir.zip $release_dir
