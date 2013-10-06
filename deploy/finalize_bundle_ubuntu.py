import sys,os
import write_version_file
import shutil

distribtution_dir = 'dist'
pupil_capture_dir =  os.path.join(distribtution_dir, 'pupil_capture')
pupil_capture_exc = os.path.join(pupil_capture_dir,'pupil_capture')
v4l2_exc = os.path.join(pupil_capture_dir,'v4l2-ctl')


os.chmod(pupil_capture_exc,0775)
os.chmod(v4l2_exc,0775)
print "gave pupil_capture and v4l2 excecutables proper rights"

shutil.copy('make_shortcut.sh',os.path.join(distribtution_dir,'make_shortcut.sh'))
print "copied a small script that creates a shortcut for the user into distribtution_dir"
os.chmod(os.path.join(distribtution_dir,'make_shortcut.sh'),0775)
print "gave that file excetion rights"

print "starting version stript:"
write_version_file.main(pupil_capture_dir)
print "created version file in dist folder"



