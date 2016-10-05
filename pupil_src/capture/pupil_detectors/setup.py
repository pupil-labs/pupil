'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

# # monkey-patch for parallel compilation
# def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
#     # those lines are copied from distutils.ccompiler.CCompiler directly
#     macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
#     cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
#     # parallel code
#     N=4 # number of parallel compilations
#     import multiprocessing.pool
#     def _single_compile(obj):
#         try: src, ext = build[obj]
#         except KeyError: return
#         self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
#     # convert to list, imap is evaluated on-demand
#     list(multiprocessing.pool.ThreadPool(N).imap(_single_compile,objects))
#     return objects
# import distutils.ccompiler
# distutils.ccompiler.CCompiler.compile=parallelCCompile

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

# opencv3 - highgui module has been split into parts: imgcodecs, videoio, and highgui itself
opencv_libraries = ['opencv_core', 'opencv_highgui', 'opencv_videoio', 'opencv_imgcodecs', 'opencv_imgproc', 'opencv_video']
opencv_library_dir = '/usr/local/opt/opencv3/lib'
opencv_include_dir = '/usr/local/opt/opencv3/include'

if(not os.path.isfile(opencv_library_dir+'/libopencv_core.so')):
    ros_dists = ['kinetic', 'jade', 'indigo']
    for ros_dist in ros_dists:
        ros_candidate_path = '/opt/ros/'+ros_dist+'/lib'
        if(os.path.isfile(ros_candidate_path+'/libopencv_core3.so')):
            opencv_library_dir = ros_candidate_path
            opencv_include_dir = '/opt/ros/'+ros_dist+'/include/opencv-3.1.0-dev'
            opencv_libraries = [lib + '3' for lib in opencv_libraries]
            break

extensions = [
    Extension(
        name="detector_2d",
        sources=['detector_2d.pyx','singleeyefitter/ImageProcessing/cvx.cpp','singleeyefitter/utils.cpp','singleeyefitter/detectorUtils.cpp' ],
        include_dirs = [ np.get_include() , '/usr/local/include/eigen3','/usr/include/eigen3', shared_cpp_include_path , singleeyefitter_include_path, opencv_include_dir ],
        libraries = ['boost_python']+opencv_libraries,
        library_dirs = [opencv_library_dir],
        extra_link_args=[], #'-WL,-R/usr/local/lib'
        extra_compile_args=["-std=c++11",'-w','-O2'], #-w hides warnings
        depends= dependencies,
        language="c++"),
     Extension(
        name="detector_3d",
        sources=['detector_3d.pyx','singleeyefitter/ImageProcessing/cvx.cpp','singleeyefitter/utils.cpp','singleeyefitter/detectorUtils.cpp', 'singleeyefitter/EyeModelFitter.cpp','singleeyefitter/EyeModel.cpp'],
        include_dirs = [ np.get_include() , '/usr/local/include/eigen3','/usr/include/eigen3', shared_cpp_include_path , singleeyefitter_include_path, opencv_include_dir ],
        libraries = ['ceres', 'boost_python']+opencv_libraries,
        library_dirs = [opencv_library_dir],
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

