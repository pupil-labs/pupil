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
    python_includes = sp.check_output("python-config --includes",shell=True ).strip('\n')
    python_ldflags= sp.check_output("python-config --ldflags",shell=True ).strip('\n')
    shared_cpp_include_path = '-I../../shared_cpp/include/ '
    boost_ldflags = " -lboost_python"

    s = "g++ -std=c++11 " + shared_cpp_include_path + python_includes + python_ldflags +boost_ldflags + " loggerTest.cpp -o loggerTest"
    sp.call(s,shell=True)

    print "BUILD COMPLETE ______________________"
    sp.call("./loggerTest",shell=True)
    sp.call("rm loggerTest",shell=True)
