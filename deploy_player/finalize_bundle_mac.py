import shutil
import write_version_file
print "starting version stript:"
write_version_file.main('dist/Pupil Player.app/Contents/MacOS')
print "created version file in dist folder"
