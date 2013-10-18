import shutil
import write_version_file
print "starting version stript:"
write_version_file.main('dist/Pupil Capture.app/Contents/MacOS')
print "created version file in dist folder"

print "copy starter app"
shutil.copytree('run_pupil_capture_from_mac_terminal.app', 'dist/run_pupil_capture_from_mac_terminal.app')