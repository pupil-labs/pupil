
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
import os, platform



dependencies = []
# include all header files, to recognize changes
for dirpath, dirnames, filenames in os.walk("singleeyefitter"):
    for filename in [f for f in filenames if f.endswith(".h")]:
        dependencies.append( os.path.join(dirpath, filename) )

shared_cpp_include_path = '../../shared_cpp/include'
singleeyefitter_include_path = 'singleeyefitter/'

extensions = [
    Extension(
        name="detector_2d",
        sources=['detector_2d.pyx','singleeyefitter/ImageProcessing/cvx.cpp','singleeyefitter/utils.cpp','singleeyefitter/detectorUtils.cpp' ],
        include_dirs = [ np.get_include() , '/usr/local/include/eigen3','/usr/include/eigen3', shared_cpp_include_path , singleeyefitter_include_path, '/usr/local/opt/opencv3/include'],
        libraries = ['opencv_highgui','opencv_core','opencv_imgproc' , 'boost_python'],
        library_dirs = ['/usr/local/opt/opencv3/lib'],
        extra_link_args=[], #'-WL,-R/usr/local/lib'
        extra_compile_args=["-std=c++11",'-w','-O2'], #-w hides warnings
        depends= dependencies,
        language="c++"),
     Extension(
        name="detector_3d",
        sources=['detector_3d.pyx','singleeyefitter/ImageProcessing/cvx.cpp','singleeyefitter/utils.cpp','singleeyefitter/detectorUtils.cpp', 'singleeyefitter/EyeModelFitter.cpp','singleeyefitter/EyeModel.cpp'],
        include_dirs = [ np.get_include() , '/usr/local/include/eigen3','/usr/include/eigen3', '/usr/local/opt/opencv3/include', shared_cpp_include_path , singleeyefitter_include_path],
        libraries = ['opencv_highgui','opencv_core','opencv_imgproc', 'opencv_video', 'ceres', 'boost_python' ],
        library_dirs = ['/usr/local/opt/opencv3/lib'],
        extra_link_args=[], #'-WL,-R/usr/local/lib'
        extra_compile_args=["-std=c++11",'-w','-O2'], #-w hides warnings
        depends= dependencies,
        language="c++"),
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

