if [ -f "/etc/modprobe.d/pupil_uvc_cam.conf" ]
then
echo "Your system had be patched with this patch before. I will undo this patch now." 
sudo rm  "/etc/modprobe.d/pupil_uvc_cam.conf"
else
echo "I will mod your camera driver to be less strict with badwidth allocation. If you want to undo just re-run this script."
# make the change for this session
sudo rmmod uvcvideo
sudo modprobe uvcvideo quirks=0x80
# same patch but permanent for the future.
sudo sh -c 'echo "options uvcvideo quirks=0x80" > /etc/modprobe.d/pupil_uvc_cam.conf'
fi
echo "Ok, Done. Press any key to exit."
read any_key

