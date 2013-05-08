import fnmatch
import os
import re

# load license_header.txt
license_txt = file('license_header.txt', 'r').read()
pattern = re.compile('\([*]\)~(.+?)~\([*]\)', re.DOTALL|re.MULTILINE)


# choose files names/types to include
# choose directories and file names/types to exclude from search
includes = ['*.py', '*.c']
excludes = ['.git*','src_video', 'uvcc', 'data*', 'License']
# header_pattern = re.compile('(?:\n[\t ]*)\("{3}|\*)(.*?)\("{3}|*/)')

# find out the cwd and change to the top level Pupil folder
cwd = os.getcwd()
pupil_dir = os.path.join(*os.path.split(cwd)[:-1])

# transform glob patterns to regular expressions
includes = r'|'.join([fnmatch.translate(x) for x in includes])
excludes = r'|'.join([fnmatch.translate(x) for x in excludes]) or r'$.'


def get_files(start_dir, includes, excludes):
	# use os.walk to recursively dig down into the Pupil directory
	match_files = []
	for root, dirs, files in os.walk(start_dir):
		if not re.search(excludes, root):
			files = [f for f in files if re.search(includes, f) and not re.search(excludes, f)]
			files = [os.path.join(root, f) for f in files]
			match_files += files
	return match_files

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
if __name__ == '__main__':
	# print "%s" %(pattern.findall(license_txt))
	print get_files("../",includes,excludes)