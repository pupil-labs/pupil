'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''


def build_cpp_extension():
    import subprocess as sp
    import os, sys
    src_loc = os.path.dirname(os.path.realpath(__file__))
    install_loc = os.path.split(os.path.split(src_loc)[0])[0]
    cwd = os.getcwd()
    build_cmd = "cd {0} && {1} setup.py install --install-lib={2} && {1} setup.py clean && cd {3}"
    sp.call(build_cmd.format(src_loc, sys.executable, install_loc, cwd), shell=True)


if __name__ == '__main__':
    build_cpp_extension()
