
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


np.get_include()
extensions = [
    Extension(
        # configured to Andrew Xia's ubuntu installation location
        name="eye_model_3d",
        sources=['eye_model_3d.pyx','SingleEyeFitter.cpp','utils.cpp'],#,'cvx.cpp'], #I don't need cvx.
        include_dirs = [ '.', '/usr/local/include/eigen3','/usr/local/include/ceres'
        #'usr/local/include/spii',#'/home/ceres-solver',
        ],
        libraries = ['spii','opencv_highgui','opencv_core','opencv_imgproc','ceres'],
        # library_dirs = ['/usr/local/lib'],
        extra_link_args=[], #'-WL,-R/usr/local/lib'
        extra_compile_args=["-std=c++11",'-w'], #-w hides warnings
        language="c++")
]

setup(
    name="eye_model_3d",
    version="0.1",
    url="https://github.com/pupil-labs/pupil",
    author='Pupil Labs',
    author_email='info@pupil-labs.com',
    license='GNU',
    ext_modules=cythonize(extensions)
)
