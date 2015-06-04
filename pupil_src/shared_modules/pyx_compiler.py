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
from Cython.Build import cythonize
import os


def build_extensions():
    #find all pyx src files
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    pyx_files = []
    for root, dirs, files in os.walk(pupil_base_dir):
        for file in files:
            if file.endswith(".pyx"):
                 pyx_files.append(os.path.join(root, file))

    #compile them as extention modules inplace.
    setup(
        ext_modules = cythonize(pyx_files),
        script_args=['build_ext','--inplace','-q']
    )


if __name__ == '__main__':
    build_extensions()