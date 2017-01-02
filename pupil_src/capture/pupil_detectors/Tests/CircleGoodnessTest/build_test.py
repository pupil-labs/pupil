'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
if __name__ == '__main__':
    import subprocess as sp
    sp.call("python setup_test.py build_ext --inplace",shell=True)
print "BUILD COMPLETE ______________________"
