'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2017  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the files COPYING and COPYING.LESSER, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
if __name__ == '__main__':
    import subprocess as sp
    sp.call("python setup_test.py build_ext --inplace",shell=True)
print "BUILD COMPLETE ______________________"
