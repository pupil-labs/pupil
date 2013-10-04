import os,sys

modules_path = os.path.join(os.path.abspath(__file__).rsplit(os.path.sep,2)[0],'pupil_src','shared_modules')
sys.path.append(modules_path)
print modules_path
from git_version import get_tag_commit

version = get_tag_commit()
print "Current version of Pupil: ",version

with open("dist/pupil_capture/_version_string_",'w') as f:
	f.write(version)
	print 'Wrote version into: "dist/pupil_capture/_version_string_" '

try:
	with open("dist/pupil_capture.app/_version_string_",'w') as f:
		f.write(version)
		print 'Wrote version into: "dist/pupil_capture.app/_version_string_" '
except:
	print"I guess you are not bundling a macos app."

print "done"