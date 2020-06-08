#!/bin/bash

# get most major.minor tag, without trailing count
current_tag=$(git describe --tags --long)

release_dir=$(echo "pupil_${current_tag}_linux_x64")
echo "release_dir:  ${release_dir}"
mkdir ${release_dir}

ext=deb

# bundle Pupil Capture
printf "\n##########\nBundling Pupil Capture\n##########\n\n"
cd deploy_capture
./bundle.sh
mv *.$ext ../$release_dir
cd ..

# bundle Pupil Service
printf "\n##########\nBundling Pupil Service\n##########\n\n"
cd deploy_service
./bundle.sh
mv *.$ext ../$release_dir
cd ..

# bundle Pupil Player
printf "\n##########\nBundling Pupil Player\n##########\n\n"
cd deploy_player
./bundle.sh
mv *.$ext ../$release_dir
cd ..

printf "\n##########\nzipping release\n##########\n\n"
zip -r $release_dir.zip $release_dir
