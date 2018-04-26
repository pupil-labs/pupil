#!/bin/sh

# create release dir from latest tag
current_tag=$(git describe --tags | awk -F"-" '{print $1"."$2}')
release_dir=$(echo "pupil_${current_tag}_linux_x64")
echo "release_dir:  ${release_dir}"
mkdir ${release_dir}

# build dependencies
printf "\n##########\nBuilding pupil detector\n##########\n\n"
python3 ../pupil_src/shared_modules/pupil_detectors/build.py

printf "\n##########\nBuilding cython modules\n##########\n\n"
python3 ../pupil_src/shared_modules/cython_methods/build.py

printf "\n##########\nBuilding calibration methods\n##########\n\n"
python3 ../pupil_src/shared_modules/calibration_routines/optimization_calibration/build.py

# bundle Pupil Capture
printf "\n##########\nBundling Pupil Capture\n##########\n\n"
cd deploy_capture
./bundle.sh
mv *.deb ../$release_dir

# bundle Pupil Service
printf "\n##########\nBundling Pupil Service\n##########\n\n"
cd ../deploy_service
./bundle.sh
mv *.deb ../$release_dir

# bundle Pupil Player
printf "\n##########\nBundling Pupil Player\n##########\n\n"
cd ../deploy_player
./bundle.sh
mv *.deb ../$release_dir

cd ..
printf "\n##########\nzipping release\n##########\n\n"
zip -r $release_dir.zip $release_dir