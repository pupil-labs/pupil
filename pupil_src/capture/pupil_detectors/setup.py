
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

extensions = [
    Extension(
        # configured to Andrew Xia's ubuntu installation location
        name="detector_2d",
        sources=['detector_2d.pyx','singleeyefitter/cvx.cpp','singleeyefitter/utils.cpp','singleeyefitter/detectorUtils.cpp'], #I don't need cvx.
        include_dirs = [ np.get_include() , '/usr/local/include/eigen3'],
        libraries = ['opencv_highgui','opencv_core','opencv_imgproc'],
        # library_dirs = ['/usr/local/lib'],
        extra_link_args=[], #'-WL,-R/usr/local/lib'
        extra_compile_args=["-std=c++11",'-w'], #-w hides warnings
        language="c++"),
     Extension(
        # configured to Andrew Xia's ubuntu installation location
        name="detector_3d",
        sources=['detector_3d.pyx','singleeyefitter/cvx.cpp','singleeyefitter/utils.cpp','singleeyefitter/detectorUtils.cpp', 'singleeyefitter/SingleEyeFitter.cpp'], #I don't need cvx.
        include_dirs = [ np.get_include() , '/usr/local/include/eigen3'],
        libraries = ['opencv_highgui','opencv_core','opencv_imgproc'],
        # library_dirs = ['/usr/local/lib'],
        extra_link_args=[], #'-WL,-R/usr/local/lib'
        extra_compile_args=["-std=c++11",'-w'], #-w hides warnings
        language="c++"),
    # Extension(
    #     # configured to Andrew Xia's ubuntu installation location
    #     name="eye_model_3d",
    #     sources=['eye_model_3d.pyx','singleeyefitter/SingleEyeFitter.cpp','singleeyefitter/utils.cpp'],#,'cvx.cpp'], #I don't need cvx.
    #     include_dirs = [ 'singleeyefitter/', '/usr/local/include/eigen3','/usr/local/include/ceres' , np.get_include()
    #     #'usr/local/include/spii',#'/home/ceres-solver',
    #     ],
    #     libraries = ['spii','opencv_highgui','opencv_core','opencv_imgproc','ceres'],
    #     # library_dirs = ['/usr/local/lib'],
    #     extra_link_args=[], #'-WL,-R/usr/local/lib'
    #     extra_compile_args=["-std=c++11",'-w'], #-w hides warnings
    #     language="c++")
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
