'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2017  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the files COPYING and COPYING.LESSER, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

def build_cpp_extension():
    import subprocess as sp
    import os
    src_loc = os.path.dirname(os.path.realpath(__file__))
    cwd = os.getcwd()
    sp.call("cd %s && python setup.py build_ext --inplace && cd %s"%(src_loc,cwd),shell=True)

if __name__ == '__main__':
    build_cpp_extension()

