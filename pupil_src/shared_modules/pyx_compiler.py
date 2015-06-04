'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

'''
This is out cython auto compiler. It is only used when running from src and exluded in the bundles.
'build_extensions' will look though the whole src dir tree and compile all pyx files as extenion modules.
These can then be imported as if they where pure python modules.
'''

from distutils.core import setup
from distutils.command import clean
from Cython.Build import cythonize
import os,sys
import logging


def build_extension_inplace(dir,ext):
    cwd = os.getcwd()
    os.chdir(dir)
    #compile them as extention modules inplace.
    try:
        os.mkdir('build') #need to manually make the being able to call --clean multiple times.
    except OSError:
        pass
    setup(
        ext_modules = cythonize(ext),
        script_args=['build_ext','--inplace','clean']
    )
    os.chdir(cwd)


def build_all_extensions():
    #find all pyx src files
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    for root, dirs, files in os.walk(pupil_base_dir):
        for file in files:
            if file.endswith(".pyx"):
                 build_extension_inplace(root, file)

def build_extensions():
    if True:
        #silence the output
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull,'w')
        try:
            build_all_extensions()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
    else:
        build_all_extensions()


if __name__ == '__main__':
    build_extensions()