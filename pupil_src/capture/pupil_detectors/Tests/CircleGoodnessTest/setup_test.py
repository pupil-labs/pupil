'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

# monkey-patch for parallel compilation
def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    # parallel code
    N=4 # number of parallel compilations
    import multiprocessing.pool
    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N).imap(_single_compile,objects))
    return objects
import distutils.ccompiler
distutils.ccompiler.CCompiler.compile=parallelCCompile

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

import sys , os
 # Make all pupil shared_modules available to this Python session.
pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_detectors', 1)[0] + 'pupil_detectors'
pupil_test_dir = os.path.abspath(__file__).rsplit('CircleGoodnessTest', 1)[0]
print pupil_test_dir
include_path = [  pupil_base_dir,pupil_test_dir ]

shared_cpp_include_path = '../../shared_cpp/include'
singleeyefitter_include_path = 'singleeyefitter/'

extensions = [
    Extension(

        name="CircleGoodnessTest",
        sources=['CircleGoodnessTest.pyx' , '../../singleeyefitter/utils.cpp'],
        include_dirs = [ '/usr/local/include/eigen3' ,  pupil_base_dir, pupil_test_dir  ],
        libraries = ['ceres'],
        # library_dirs = ['/usr/local/lib'],
        extra_link_args=[], #'-WL,-R/usr/local/lib'
        extra_compile_args=["-std=c++11",'-w'], #-w hides warnings
        language="c++"),

]

setup(
    name="CircleGoodness Test",
    version="0.1",
    url="https://github.com/pupil-labs/pupil",
    author='Pupil Labs',
    author_email='info@pupil-labs.com',
    license='GNU',
    ext_modules=cythonize(extensions,include_path = include_path )
)
