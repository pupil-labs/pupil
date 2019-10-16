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
import os, sys, platform, shutil, glob


dependencies = []
# include all header files, to recognize changes
for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".h")]:
        dependencies.append(os.path.join(dirpath, filename))

shared_cpp_include_path = "../../../shared_cpp/include"
singleeyefitter_include_path = "../../pupil_detectors/singleeyefitter"
extra_compile_args = ["-D_USE_MATH_DEFINES", "-std=c++11", "-w", "-O2"]

if platform.system() == "Windows":
    libs = []
    library_dirs = []

    # Check if the setup.py is running in a Conda environment
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    if is_conda:
        # Deduct the include and library dir dynamically
        include_dir = os.path.join(sys.prefix, "Library", "include")
        lib_dir = os.path.join(sys.prefix, "Library", "lib")
        if not os.path.exists(include_dir) or not os.path.exists(lib_dir):
            raise EnvironmentError("Unable to locate Anaconda include or lib directory")

        # When installing Eigen via Conda it is installed in a folder called 'eigen3'. Deal with that, if appropriate.
        eigen_dir = os.path.join(include_dir, "Eigen")
        custom_eigen_dir = os.path.join(include_dir, "eigen3")
        if not os.path.exists(eigen_dir) and os.path.exists(custom_eigen_dir):
            """
            # Creating a junction might be the prettiest way - but it need extended rights.
            import ctypes
            kernel_dll = ctypes.windll.LoadLibrary("kernel32.dll")
            if kernel_dll.CreateSymbolicLinkA(eigen_dir, custom_eigen_dir, 1) == 0:
                raise EnvironmentError(f"Unable tio create junction between {eigen_dir} and {custom_eigen_dir}.")
            """

            # Use the ugly variant and simply copy the directory
            shutil.copytree(src=custom_eigen_dir, dst=eigen_dir)

        # Change mode to prevent mismatch with the Conda libraries
        extra_compile_args.append("-MD")

        # Finally build lib_spec dynamically
        lib_spec = [
            [np.get_include(), ""],
            [include_dir, ""],
            [
                os.path.join(include_dir, "opencv2"),
                *glob.glob(os.path.join(lib_dir, "opencv_*.lib")),
            ],
            [os.path.join(include_dir, "Eigen"), ""],
            [os.path.join(include_dir, "ceres"), os.path.join(lib_dir, "ceres.lib")],
            [os.path.join(include_dir, "glog"), os.path.join(lib_dir, "glog.lib")],
        ]
    else:
        lib_spec = [
            [np.get_include(), ""],
            [
                "C:\\work\\opencv\\build\\include",
                "C:\\work\\opencv\\build\\x64\\vc14\\lib\\opencv_world345.lib",
            ],
            ["C:\\work\\ceres-windows\\Eigen", ""],
            [
                "C:\\work\\ceres-windows\\ceres-solver\\include",
                "C:\\work\\ceres-windows\\x64\\Release\\ceres_static.lib",
            ],
            [
                "C:\\work\\ceres-windows\\glog\\src\\windows",
                "C:\\work\\ceres-windows\\x64\\Release\\libglog_static.lib",
            ],
            ["C:\\work\\ceres-windows", ""],
        ]

    include_dirs = [spec[0] for spec in lib_spec]
    include_dirs.append(shared_cpp_include_path)
    include_dirs.append(singleeyefitter_include_path)

    # Handle multiple library entries, for example in the case of OpenCV
    xtra_obj = []
    for spec in lib_spec:
        xtra_obj.extend(spec[1:])

else:
    # opencv3 - highgui module has been split into parts: imgcodecs, videoio, and highgui itself
    opencv_libraries = [
        "opencv_core",
        "opencv_highgui",
        "opencv_videoio",
        "opencv_imgcodecs",
        "opencv_imgproc",
        "opencv_video",
    ]

    # explicit lib and include dirs for homebrew installed opencv
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
    include_dirs = [
        np.get_include(),
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
        shared_cpp_include_path,
        singleeyefitter_include_path,
    ] + opencv_include_dirs
    libs = ["ceres"] + opencv_libraries
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
        extra_compile_args=extra_compile_args,  # -w hides warnings
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
