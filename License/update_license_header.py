import fnmatch
import os
import re

# load license_header.txt
license_txt = file('license_header.txt', 'r').read()
pattern = re.compile('\([*]\)~(.+?)~\([*]\)', re.DOTALL|re.MULTILINE)

print "%s" %(pattern.findall(license_txt))

# choose files types to include
# choose directories to exclude from search
includes = ['*.py', '*.c'] 
excludes = ['.git', 'atb', 'glfw', 'src_video', 'v4l2_ctl', 'data', 'License'] 
# header_pattern = re.compile('(?:\n[\t ]*)\("{3}|\*)(.*?)\("{3}|*/)')

# find out the cwd and change to the top level Pupil folder
cwd = os.getcwd()
pupil_dir = os.path.join(*os.path.split(cwd)[:-1])

# transform glob patterns to regular expressions
includes = r'|'.join([fnmatch.translate(x) for x in includes])
excludes = r'|'.join([fnmatch.translate(x) for x in excludes]) or r'$.'


def get_files(start_dir, includes, excludes):
	# use os.walk to recursively dig down into the Pupil directory
	for root, dirs, files in os.walk(start_dir):
		
		# exclude directories specified in the excludes list
		# serial directories like data and data_001 data_00n are excluded with regex pattern
		dirs[:] = [d for d in dirs if not re.match(excludes, d)]

		# only select files in the includes list
		files = [os.path.join(root, f) for f in files]
		files = [f for f in files if re.match(includes, f)]

	return files

def write_header(files, license_txt):
	# Add a license/docstring header to selected files
	for f in files:
		# read original and copy to data for pre-pending
		with file(f, 'r') as original: 
			data = original.read()

		# add license header and append file data
		with file(f, 'w') as modified: 
			#if header already exists
			modified.write(license_txt + data)
			modified.close()

def remove_header(file, pattern):
	# find doc string or multiline comment in top of file and remove it
	pass
