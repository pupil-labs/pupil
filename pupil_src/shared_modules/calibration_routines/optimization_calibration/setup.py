"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

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
import os, sys, platform


dependencies = []
# include all header files, to recognize changes
for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".h")]:
        dependencies.append(os.path.join(dirpath, filename))

root_dir = ''
include_dirs = [os.path.join(root_dir, '..', '..', '..', 'shared_cpp', 'include'),
                os.path.join(root_dir, '..', '..', 'pupil_detectors', 'singleeyefitter'),
                np.get_include()]


# opencv3 - highgui module has been split into parts: imgcodecs, videoio, and highgui itself
opencv_libraries = [
    "opencv_core",
    "opencv_highgui",
    "opencv_videoio",
    "opencv_imgcodecs",
    "opencv_imgproc",
    "opencv_video",
]

if platform.system() == "Windows":
    # Find the path where dependencies are installed.
    usr_local = None
    if 'VCPKG_PREFIX' in os.environ:
        usr_local = os.environ['VCPKG_PREFIX']
    elif 'CONDA_PREFIX' in os.environ:
        usr_local = os.path.join(os.environ['CONDA_PREFIX'], 'Library')
    else:
        test_paths = os.environ['PYTHONPATH']
        for t_p in test_paths.split(';'):
            if any([_.startswith('opencv_core') for _ in os.listdir(t_p)]):
                usr_local = os.path.dirname(t_p)
                break
    if usr_local is None:
        raise EnvironmentError("Could not find library directory."
                               "Set environment variable for VCPKG_PREFIX or use conda.")

    usr_local = os.path.abspath(usr_local)
    include_dirs.append(os.path.join(usr_local, 'include'))
    lib_dir = os.path.join(usr_local, 'lib')
    library_dirs = [lib_dir]
    libs = []
    # Get a list of OpenCV libraries.
    opencv_match = [_ for _ in os.listdir(lib_dir) if _.startswith('opencv_core')][0]
    opencv_term = opencv_match[11:]
    opencv_libs = [_ + opencv_term for _ in opencv_libraries]
    # Get a list of required boost libraries (only boost_python?)
    boost_libs = [_ for _ in os.listdir(lib_dir) if _.startswith('boost_python')]
    # Collect list of required libraries.
    xtra_obj = opencv_libs + boost_libs + ['ceres.lib', 'glog.lib']

else:
    opencv_library_dirs = [
        "/usr/local/opt/opencv/lib",  # old opencv brew (v3)
        "/usr/local/opt/opencv@3/lib",  # new opencv@3 brew
        "/usr/local/lib",  # new opencv brew (v4)
    ]
    opencv_include_dirs = [
        "/usr/local/opt/opencv/include",  # old opencv brew (v3)
        "/usr/local/opt/opencv@3/include",  # new opencv@3 brew
        "/usr/local/include/opencv4",  # new opencv brew (v4)
    ]
    opencv_core_found = any(
        os.path.isfile(path + "/libopencv_core.so") for path in opencv_library_dirs
    )
    if not opencv_core_found:
        ros_dists = ["kinetic", "jade", "indigo"]
        for ros_dist in ros_dists:
            ros_candidate_path = "/opt/ros/" + ros_dist + "/lib"
            if os.path.isfile(ros_candidate_path + "/libopencv_core3.so"):
                opencv_library_dirs = [ros_candidate_path]
                opencv_include_dirs = [
                    "/opt/ros/" + ros_dist + "/include/opencv-3.1.0-dev"
                ]
                opencv_libraries = [lib + "3" for lib in opencv_libraries]
                break
    include_dirs.extend([
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ])
    include_dirs.extend(opencv_include_dirs)
    python_version = sys.version_info
    if platform.system() == "Linux":
        # boost_python-py34
        boost_lib = "boost_python-py" + str(python_version[0]) + str(python_version[1])
    else:
        boost_lib = "boost_python" + str(python_version[0]) + str(python_version[1])
    libs = ["ceres", boost_lib] + opencv_libraries
    xtra_obj = []
    library_dirs = opencv_library_dirs


extensions = [
    Extension(
        name="calibration_routines.optimization_calibration.calibration_methods",
        sources=["calibration_methods.pyx"],
        include_dirs=include_dirs,
        libraries=libs,
        library_dirs=library_dirs,
        extra_link_args=[],  # '-WL,-R/usr/local/lib'
        extra_compile_args=[
            "-D_USE_MATH_DEFINES",
            "-std=c++11",
            "-w",
            "-O2",
        ],  # -w hides warnings
        extra_objects=xtra_obj,
        depends=dependencies,
        language="c++",
    )
]

if __name__ == "__main__":
    setup(
        name="calibration_routines.optimization_calibration",
        version="0.1",
        url="https://github.com/pupil-labs/pupil",
        author="Pupil Labs",
        author_email="info@pupil-labs.com",
        license="GNU",
        ext_modules=cythonize(extensions, quiet=True, nthreads=8),
    )
