
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


np.get_include()
extensions = [
    Extension(
        name="eye_model_3d",
        sources=['eye_model_3d.pyx','SingleEyeFitter.cpp','utils.cpp','cvx.cpp'],
        include_dirs = [ '.', '/usr/local/include/eigen3','/usr/local/include/ceres','/usr/local/include/boost','usr/local/include/opencv2',' /usr/local/include/spii'],
        libraries = ['spii','opencv_core','opencv_highgui','ceres'],
        library_dirs = [],
        extra_link_args=[],
        extra_compile_args=["-std=c++11"],
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
