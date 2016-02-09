'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

dependencies = []

extensions = [
    Extension(
        name="pycpplog",
        sources=['pycpplog.pyx'  ],
        include_dirs = [ ],
        libraries = [ ],
        #library_dirs = ['/usr/local/lib'],
        extra_link_args=[], #'-WL,-R/usr/local/lib'
        extra_compile_args=['-std=c++11','-w','-O2'], #-w hides warnings
        depends= dependencies,
        language="c++")
]

setup(
    name="pycpplog",
    version="0.1",
    url="https://github.com/pupil-labs/pycpplog",
    author='Pupil Labs',
    author_email='info@pupil-labs.com',
    license='GNU',
    ext_modules=cythonize(extensions)
)

