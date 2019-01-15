"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        name="cython_methods.methods",
        sources=["methods.pyx"],
        include_dirs=[np.get_include()],
        libraries=[],
        library_dirs=[],
        extra_link_args=[],
        extra_compile_args=["-D_USE_MATH_DEFINES", "-std=c++11", "-w", "-O2"],
        extra_objects=[],
        depends=[],
        language="c++",
    )
]

if __name__ == "__main__":
    setup(
        name="cython_methods",
        version="0.1",
        url="https://github.com/pupil-labs/pupil",
        author="Pupil Labs",
        author_email="info@pupil-labs.com",
        license="GNU",
        ext_modules=cythonize(extensions, quiet=True, nthreads=8),
    )
