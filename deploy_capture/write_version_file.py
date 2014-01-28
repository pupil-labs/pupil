import os,sys

def main(target_dir):
	modules_path = os.path.join(os.path.abspath(__file__).rsplit(os.path.sep,2)[0],'pupil_src','shared_modules')
	sys.path.append(modules_path)
	from git_version import get_tag_commit

	version = get_tag_commit()
	print "Current version of Pupil: ",version

	with open(os.path.join(target_dir,'_version_string_'),'w') as f:
		f.write(version)
		print 'Wrote version into: %s' %os.path.join(target_dir,'_version_string_')

def get_version():
	modules_path = os.path.join(os.path.abspath(__file__).rsplit(os.path.sep,2)[0],'pupil_src','shared_modules')
	sys.path.append(modules_path)
	from git_version import get_tag_commit
	return get_tag_commit()

if __name__ == '__main__':
	main("dist/pupil_capture")