rm *.dmg
rm *.deb
cd deploy_capture
./bundle.sh
mv *.deb ../
mv *.dmg ../
cd ../deploy_service
./bundle.sh
mv *.deb ../
mv *.dmg ../
cd ../deploy_player
./bundle.sh
mv *.deb ../
mv *.dmg ../
cd ../